"""
Relationship Extractor for Knowledge Graph Construction.

Extracts relationships between entities using:
- Dependency parsing
- Pattern-based extraction
- LLM-based inference
- Co-reference resolution
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

import spacy
from spacy.tokens import Doc, Token, Span
from loguru import logger

from .entity_extractor import Entity, EntityExtractionResult
from ..core.config import get_settings


class RelationType(str, Enum):
    """Standard relationship types."""
    # Employment and affiliation
    WORKS_FOR = "WORKS_FOR"
    EMPLOYED_BY = "EMPLOYED_BY"
    MEMBER_OF = "MEMBER_OF"

    # Location relationships
    LOCATED_IN = "LOCATED_IN"
    HEADQUARTERS_IN = "HEADQUARTERS_IN"
    BORN_IN = "BORN_IN"

    # Ownership and creation
    OWNS = "OWNS"
    CREATED_BY = "CREATED_BY"
    INVENTED_BY = "INVENTED_BY"
    FOUNDED_BY = "FOUNDED_BY"

    # References and citations
    REFERENCES = "REFERENCES"
    CITES = "CITES"
    MENTIONS = "MENTIONS"

    # Hierarchical relationships
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"

    # Temporal relationships
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    OCCURS_ON = "OCCURS_ON"

    # General relationships
    RELATED_TO = "RELATED_TO"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    USES = "USES"
    IMPLEMENTS = "IMPLEMENTS"


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    source_entity: Entity
    target_entity: Entity
    relation_type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    evidence: Optional[str] = None
    source: str = "dependency"  # dependency, pattern, llm

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "source": self.source_entity.text,
            "source_type": self.source_entity.type,
            "target": self.target_entity.text,
            "target_type": self.target_entity.type,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "properties": self.properties,
            "context": self.context,
            "evidence": self.evidence,
            "extraction_source": self.source
        }

    def __hash__(self):
        """Make relationship hashable for deduplication."""
        return hash((
            self.source_entity.text.lower(),
            self.target_entity.text.lower(),
            self.relation_type
        ))

    def __eq__(self, other):
        """Check equality for deduplication."""
        if not isinstance(other, Relationship):
            return False
        return (
            self.source_entity.text.lower() == other.source_entity.text.lower() and
            self.target_entity.text.lower() == other.target_entity.text.lower() and
            self.relation_type == other.relation_type
        )


@dataclass
class RelationshipExtractionResult:
    """Result of relationship extraction."""
    relationships: List[Relationship]
    document_id: str
    statistics: Dict[str, int] = field(default_factory=dict)


class RelationshipExtractor:
    """
    Comprehensive relationship extraction using multiple techniques.

    Features:
    - Dependency parsing for grammatical relationships
    - Pattern-based extraction using spaCy matchers
    - LLM-based relationship inference
    - Co-reference resolution
    - Cross-sentence relationships
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        use_llm: bool = False,
        max_distance: int = 5,
        confidence_threshold: float = 0.7,
        enable_coref: bool = False
    ):
        """
        Initialize relationship extractor.

        Args:
            spacy_model: spaCy model name
            use_llm: Whether to use LLM for enhanced extraction
            max_distance: Maximum sentence distance for relationships
            confidence_threshold: Minimum confidence for inclusion
            enable_coref: Enable co-reference resolution
        """
        self.use_llm = use_llm
        self.max_distance = max_distance
        self.confidence_threshold = confidence_threshold
        self.enable_coref = enable_coref
        self.settings = get_settings()

        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model for relationship extraction: {spacy_model}")
        except OSError:
            logger.warning(f"Model {spacy_model} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        # Initialize pattern matcher
        self._initialize_patterns()

        # Initialize LLM client if needed
        self.llm_client = None
        if use_llm:
            self._initialize_llm()

    def _initialize_patterns(self) -> None:
        """Initialize relationship extraction patterns."""
        from spacy.matcher import DependencyMatcher

        self.dep_matcher = DependencyMatcher(self.nlp.vocab)

        # Pattern 1: Subject-Verb-Object (e.g., "John works for Microsoft")
        pattern_svo = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": "nsubj"}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}}
            }
        ]
        self.dep_matcher.add("SVO", [pattern_svo])

        # Pattern 2: Prepositional phrases (e.g., "CEO of Apple")
        pattern_prep = [
            {
                "RIGHT_ID": "head",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
            },
            {
                "LEFT_ID": "head",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {"DEP": "prep"}
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "pobj",
                "RIGHT_ATTRS": {"DEP": "pobj"}
            }
        ]
        self.dep_matcher.add("PREP", [pattern_prep])

        logger.info("Initialized relationship extraction patterns")

    def _initialize_llm(self) -> None:
        """Initialize LLM client for enhanced extraction."""
        try:
            if self.settings.llm.provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=self.settings.llm.api_key)
                logger.info("Initialized OpenAI client for relationship extraction")
            elif self.settings.llm.provider == "anthropic":
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=self.settings.llm.api_key)
                logger.info("Initialized Anthropic client for relationship extraction")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.use_llm = False

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        document_id: str
    ) -> RelationshipExtractionResult:
        """
        Extract relationships from text given entities.

        Args:
            text: Source text
            entities: List of extracted entities
            document_id: Document identifier

        Returns:
            RelationshipExtractionResult
        """
        if not entities:
            return RelationshipExtractionResult(
                relationships=[],
                document_id=document_id,
                statistics={"total": 0}
            )

        # Process text with spaCy
        doc = self.nlp(text)

        # Map entities to their spans in the doc
        entity_spans = self._map_entities_to_doc(entities, doc)

        # Extract relationships using different methods
        relationships = []

        # 1. Dependency-based extraction
        dep_rels = self._extract_dependency_relationships(doc, entity_spans)
        relationships.extend(dep_rels)

        # 2. Pattern-based extraction
        pattern_rels = self._extract_pattern_relationships(doc, entity_spans)
        relationships.extend(pattern_rels)

        # 3. Proximity-based relationships
        prox_rels = self._extract_proximity_relationships(entity_spans, doc)
        relationships.extend(prox_rels)

        # 4. LLM-based extraction (if enabled)
        if self.use_llm and len(entities) < 20:  # Only for smaller entity sets
            llm_rels = self._extract_llm_relationships(text, entities, document_id)
            relationships.extend(llm_rels)

        # Deduplicate relationships
        relationships = self._deduplicate_relationships(relationships)

        # Filter by confidence
        relationships = [
            rel for rel in relationships
            if rel.confidence >= self.confidence_threshold
        ]

        # Generate statistics
        statistics = self._generate_statistics(relationships)

        result = RelationshipExtractionResult(
            relationships=relationships,
            document_id=document_id,
            statistics=statistics
        )

        logger.debug(f"Extracted {len(relationships)} relationships from document {document_id}")
        return result

    def _map_entities_to_doc(
        self,
        entities: List[Entity],
        doc: Doc
    ) -> List[Tuple[Entity, Span]]:
        """Map entities to their corresponding spans in the spaCy doc."""
        entity_spans = []

        for entity in entities:
            # Find the span in the doc
            span = doc.char_span(
                entity.start_char,
                entity.end_char,
                alignment_mode="expand"
            )

            if span:
                entity_spans.append((entity, span))

        return entity_spans

    def _extract_dependency_relationships(
        self,
        doc: Doc,
        entity_spans: List[Tuple[Entity, Span]]
    ) -> List[Relationship]:
        """Extract relationships using dependency parsing."""
        relationships = []

        # Use dependency matcher
        matches = self.dep_matcher(doc)

        for match_id, token_ids in matches:
            pattern_name = self.nlp.vocab.strings[match_id]

            # Get the tokens
            tokens = [doc[token_id] for token_id in token_ids]

            if pattern_name == "SVO":
                # Subject-Verb-Object pattern
                verb_token = tokens[0]
                subj_token = tokens[1]
                obj_token = tokens[2]

                # Find entities that match subject and object
                subj_entity = self._find_entity_at_token(subj_token, entity_spans)
                obj_entity = self._find_entity_at_token(obj_token, entity_spans)

                if subj_entity and obj_entity:
                    # Determine relationship type from verb
                    rel_type = self._verb_to_relation_type(verb_token.lemma_)

                    relationship = Relationship(
                        source_entity=subj_entity,
                        target_entity=obj_entity,
                        relation_type=rel_type,
                        confidence=0.8,
                        evidence=f"{subj_token.text} {verb_token.text} {obj_token.text}",
                        source="dependency"
                    )
                    relationships.append(relationship)

            elif pattern_name == "PREP":
                # Prepositional phrase pattern
                head_token = tokens[0]
                prep_token = tokens[1]
                pobj_token = tokens[2]

                head_entity = self._find_entity_at_token(head_token, entity_spans)
                obj_entity = self._find_entity_at_token(pobj_token, entity_spans)

                if head_entity and obj_entity:
                    # Determine relationship type from preposition
                    rel_type = self._prep_to_relation_type(prep_token.text, head_entity, obj_entity)

                    relationship = Relationship(
                        source_entity=head_entity,
                        target_entity=obj_entity,
                        relation_type=rel_type,
                        confidence=0.75,
                        evidence=f"{head_token.text} {prep_token.text} {pobj_token.text}",
                        source="dependency"
                    )
                    relationships.append(relationship)

        return relationships

    def _extract_pattern_relationships(
        self,
        doc: Doc,
        entity_spans: List[Tuple[Entity, Span]]
    ) -> List[Relationship]:
        """Extract relationships using predefined patterns."""
        relationships = []

        # Define common patterns
        patterns = [
            # "X founded Y"
            (r"founded|established|created", "FOUNDED_BY"),
            # "X acquired Y"
            (r"acquired|bought|purchased", "OWNS"),
            # "X located in Y"
            (r"located|based|situated", "LOCATED_IN"),
            # "X works for Y"
            (r"works for|employed by", "WORKS_FOR"),
        ]

        text = doc.text

        # Check each pair of entities
        for i, (ent1, span1) in enumerate(entity_spans):
            for j, (ent2, span2) in enumerate(entity_spans):
                if i >= j:
                    continue

                # Get text between entities
                start = min(span1.end_char, span2.end_char)
                end = max(span1.start_char, span2.start_char)

                if end - start > 100:  # Skip if too far apart
                    continue

                between_text = text[start:end].lower()

                # Check patterns
                for pattern, rel_type in patterns:
                    import re
                    if re.search(pattern, between_text):
                        relationship = Relationship(
                            source_entity=ent1,
                            target_entity=ent2,
                            relation_type=rel_type,
                            confidence=0.7,
                            evidence=between_text,
                            source="pattern"
                        )
                        relationships.append(relationship)

        return relationships

    def _extract_proximity_relationships(
        self,
        entity_spans: List[Tuple[Entity, Span]],
        doc: Doc
    ) -> List[Relationship]:
        """Extract relationships based on entity proximity."""
        relationships = []

        # Group entities by sentence
        sent_entities = defaultdict(list)
        for entity, span in entity_spans:
            # Find which sentence this entity belongs to
            for sent_idx, sent in enumerate(doc.sents):
                if span.start >= sent.start and span.end <= sent.end:
                    sent_entities[sent_idx].append((entity, span))
                    break

        # Create proximity relationships for entities in same sentence
        for sent_idx, ent_list in sent_entities.items():
            if len(ent_list) < 2:
                continue

            for i, (ent1, span1) in enumerate(ent_list):
                for j, (ent2, span2) in enumerate(ent_list):
                    if i >= j:
                        continue

                    # Create a generic RELATED_TO relationship
                    relationship = Relationship(
                        source_entity=ent1,
                        target_entity=ent2,
                        relation_type="RELATED_TO",
                        confidence=0.5,  # Lower confidence for proximity
                        properties={"proximity": "same_sentence"},
                        source="dependency"
                    )
                    relationships.append(relationship)

        return relationships

    def _extract_llm_relationships(
        self,
        text: str,
        entities: List[Entity],
        document_id: str
    ) -> List[Relationship]:
        """Extract relationships using LLM."""
        if not self.llm_client:
            return []

        # Prepare entity list for prompt
        entity_list = "\n".join([
            f"- {ent.text} ({ent.type})"
            for ent in entities
        ])

        prompt = f"""Identify relationships between the following entities in the text.

Entities:
{entity_list}

Text:
{text[:1500]}  # Limit text length

For each relationship, return JSON in this format:
{{
  "source": "entity text",
  "target": "entity text",
  "relation_type": "WORKS_FOR|LOCATED_IN|OWNS|CREATED_BY|etc",
  "confidence": 0.0-1.0,
  "evidence": "supporting text"
}}

Return only a JSON array of relationships, no other text."""

        try:
            if self.settings.llm.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.settings.llm.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting relationships between entities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
            elif self.settings.llm.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.settings.llm.model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.content[0].text
            else:
                return []

            # Parse JSON response
            rels_data = json.loads(content)

            relationships = []
            # Create entity lookup
            entity_map = {ent.text.lower(): ent for ent in entities}

            for rel_data in rels_data:
                source_ent = entity_map.get(rel_data["source"].lower())
                target_ent = entity_map.get(rel_data["target"].lower())

                if source_ent and target_ent:
                    relationship = Relationship(
                        source_entity=source_ent,
                        target_entity=target_ent,
                        relation_type=rel_data["relation_type"],
                        confidence=rel_data.get("confidence", 0.75),
                        evidence=rel_data.get("evidence"),
                        source="llm"
                    )
                    relationships.append(relationship)

            logger.debug(f"LLM extracted {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.warning(f"LLM relationship extraction failed: {e}")
            return []

    def _find_entity_at_token(
        self,
        token: Token,
        entity_spans: List[Tuple[Entity, Span]]
    ) -> Optional[Entity]:
        """Find entity that contains the given token."""
        for entity, span in entity_spans:
            if token.i >= span.start and token.i < span.end:
                return entity
        return None

    def _verb_to_relation_type(self, verb_lemma: str) -> str:
        """Map verb to relationship type."""
        verb_map = {
            "work": "WORKS_FOR",
            "employ": "EMPLOYED_BY",
            "own": "OWNS",
            "create": "CREATED_BY",
            "found": "FOUNDED_BY",
            "invent": "INVENTED_BY",
            "locate": "LOCATED_IN",
            "live": "LOCATED_IN",
            "mention": "MENTIONS",
            "reference": "REFERENCES",
            "cite": "CITES",
            "use": "USES",
            "implement": "IMPLEMENTS",
        }
        return verb_map.get(verb_lemma, "RELATED_TO")

    def _prep_to_relation_type(
        self,
        prep: str,
        source_entity: Entity,
        target_entity: Entity
    ) -> str:
        """Map preposition to relationship type based on entity types."""
        prep = prep.lower()

        # Location prepositions
        if prep in ["in", "at", "near"]:
            if target_entity.type == "LOCATION":
                return "LOCATED_IN"

        # Possession/ownership
        if prep == "of":
            if source_entity.type == "PERSON":
                return "WORKS_FOR"
            else:
                return "PART_OF"

        # General
        if prep == "for":
            return "WORKS_FOR"

        if prep == "by":
            return "CREATED_BY"

        return "RELATED_TO"

    def _deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Deduplicate relationships, keeping highest confidence."""
        if not relationships:
            return []

        # Sort by confidence
        relationships.sort(key=lambda r: r.confidence, reverse=True)

        # Remove duplicates
        seen = set()
        unique_rels = []

        for rel in relationships:
            key = (
                rel.source_entity.text.lower(),
                rel.target_entity.text.lower(),
                rel.relation_type
            )

            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        return unique_rels

    def _generate_statistics(
        self,
        relationships: List[Relationship]
    ) -> Dict[str, int]:
        """Generate statistics about extracted relationships."""
        stats = {
            "total": len(relationships),
            "by_type": defaultdict(int),
            "by_source": defaultdict(int),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0
        }

        for rel in relationships:
            stats["by_type"][rel.relation_type] += 1
            stats["by_source"][rel.source] += 1

            if rel.confidence >= 0.8:
                stats["high_confidence"] += 1
            elif rel.confidence >= 0.6:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1

        stats["by_type"] = dict(stats["by_type"])
        stats["by_source"] = dict(stats["by_source"])

        return stats
