"""
Entity Extractor for Knowledge Graph Construction.

Extracts named entities from documents using multiple approaches:
- spaCy NER for base entity extraction
- LLM-based extraction for complex/domain-specific entities
- Custom entity type support
- Batch processing for efficiency
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import asyncio

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from loguru import logger

from ..core.config import get_settings


class EntityType(str, Enum):
    """Standard entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    FACILITY = "FACILITY"
    GPE = "GPE"  # Geopolitical entity
    NORP = "NORP"  # Nationalities or religious or political groups

    # Custom domain-specific types
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    METHOD = "METHOD"
    METRIC = "METRIC"
    DATASET = "DATASET"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    type: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    source: str = "spacy"  # spacy, llm, or custom

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "text": self.text,
            "type": self.type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "context": self.context,
            "source": self.source
        }

    def __hash__(self):
        """Make entity hashable for deduplication."""
        return hash((self.text.lower(), self.type, self.start_char, self.end_char))

    def __eq__(self, other):
        """Check equality for deduplication."""
        if not isinstance(other, Entity):
            return False
        return (
            self.text.lower() == other.text.lower() and
            self.type == other.type and
            self.start_char == other.start_char and
            self.end_char == other.end_char
        )


@dataclass
class EntityExtractionResult:
    """Result of entity extraction."""
    entities: List[Entity]
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)


class EntityExtractor:
    """
    Comprehensive entity extraction using spaCy and optional LLM enhancement.

    Features:
    - Named Entity Recognition with spaCy
    - Custom entity patterns
    - LLM-based extraction for complex entities
    - Confidence scoring
    - Multi-language support
    - Batch processing
    """

    _SPACY_MODEL_CACHE: Dict[str, Language] = {}
    _SPACY_MODEL_ERRORS: Dict[str, str] = {}
    _SPACY_LOAD_SIGNATURES: Dict[str, int] = {}

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        use_llm: bool = False,
        confidence_threshold: float = 0.7,
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32
    ):
        """
        Initialize entity extractor.

        Args:
            spacy_model: spaCy model name
            use_llm: Whether to use LLM for enhanced extraction
            confidence_threshold: Minimum confidence for entity inclusion
            custom_patterns: Custom entity patterns for rule-based extraction
            batch_size: Batch size for processing
        """
        self.spacy_model_name = spacy_model
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.settings = get_settings()

        # Load spaCy model
        self.available = True
        self._spacy_unavailable_reason: Optional[str] = None

        self.nlp, load_error = self._load_spacy_model(
            spacy_model,
            use_cache=not bool(custom_patterns),
        )
        if self.nlp is None:
            guidance = (
                f"spaCy model '{spacy_model}' is not installed. "
                f"Run `python -m spacy download {spacy_model}` to enable entity extraction."
            )
            logger.warning(guidance)
            self.available = False
            self._spacy_unavailable_reason = load_error
        else:
            logger.info(f"Loaded spaCy model: {spacy_model}")

        # Add custom components
        if custom_patterns:
            self._add_custom_patterns(custom_patterns)

        # Initialize LLM client if needed
        self.llm_client = None
        if use_llm:
            self._initialize_llm()

    @classmethod
    def _load_spacy_model(
        cls,
        model_name: str,
        *,
        use_cache: bool = True,
    ) -> Tuple[Optional[Language], Optional[str]]:
        """Load spaCy model with optional process-level caching."""
        loader_signature = id(spacy.load)

        if use_cache:
            cached_signature = cls._SPACY_LOAD_SIGNATURES.get(model_name)
            if cached_signature is not None and cached_signature != loader_signature:
                cls._SPACY_MODEL_CACHE.pop(model_name, None)
                cls._SPACY_MODEL_ERRORS.pop(model_name, None)
                cls._SPACY_LOAD_SIGNATURES.pop(model_name, None)

            if model_name in cls._SPACY_MODEL_CACHE:
                return cls._SPACY_MODEL_CACHE[model_name], None

            if model_name in cls._SPACY_MODEL_ERRORS:
                return None, cls._SPACY_MODEL_ERRORS[model_name]

        try:
            model = spacy.load(model_name)
        except OSError as err:
            if use_cache:
                cls._SPACY_MODEL_ERRORS[model_name] = str(err)
                cls._SPACY_LOAD_SIGNATURES[model_name] = loader_signature
            return None, str(err)

        if use_cache:
            cls._SPACY_MODEL_CACHE[model_name] = model
            cls._SPACY_LOAD_SIGNATURES[model_name] = loader_signature
        return model, None

    def _add_custom_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Add custom entity patterns to spaCy pipeline."""
        from spacy.matcher import Matcher

        if not self.available or self.nlp is None:
            logger.warning("Cannot add custom patterns because spaCy model is unavailable")
            return

        if "custom_entity_matcher" not in self.nlp.pipe_names:
            matcher = Matcher(self.nlp.vocab)

            for pattern in patterns:
                matcher.add(pattern["label"], [pattern["pattern"]])

            # Add to pipeline
            @Language.component("custom_entity_matcher")
            def custom_entity_component(doc):
                matches = matcher(doc)
                new_ents = []
                for match_id, start, end in matches:
                    span = Span(doc, start, end, label=match_id)
                    new_ents.append(span)
                doc.ents = list(doc.ents) + new_ents
                return doc

            self.nlp.add_pipe("custom_entity_matcher", last=True)
            logger.info(f"Added {len(patterns)} custom entity patterns")

    def _initialize_llm(self) -> None:
        """Initialize LLM client for enhanced extraction."""
        try:
            if self.settings.llm.provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=self.settings.llm.api_key)
                logger.info("Initialized OpenAI client for entity extraction")
            elif self.settings.llm.provider == "anthropic":
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=self.settings.llm.api_key)
                logger.info("Initialized Anthropic client for entity extraction")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.use_llm = False

    def extract_entities(
        self,
        text: str,
        document_id: str,
        extract_attributes: bool = True,
        include_context: bool = True,
        context_window: int = 50
    ) -> EntityExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from
            document_id: Document identifier
            extract_attributes: Whether to extract entity attributes
            include_context: Whether to include context around entities
            context_window: Number of characters before/after entity for context

        Returns:
            EntityExtractionResult with extracted entities
        """
        if not self.available or self.nlp is None:
            logger.warning(
                "Entity extraction requested but spaCy model is unavailable; returning no entities"
            )
            return EntityExtractionResult(
                entities=[],
                document_id=document_id,
                metadata={"disabled": True, "reason": getattr(self, "_spacy_unavailable_reason", None)},
                statistics={}
            )

        # Process with spaCy
        doc = self.nlp(text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            # Get context if requested
            context = None
            if include_context:
                start_ctx = max(0, ent.start_char - context_window)
                end_ctx = min(len(text), ent.end_char + context_window)
                context = text[start_ctx:end_ctx]

            # Extract attributes
            attributes = {}
            if extract_attributes:
                attributes = self._extract_entity_attributes(ent, doc)

            entity = Entity(
                text=ent.text,
                type=self._normalize_entity_type(ent.label_),
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=self._calculate_confidence(ent),
                attributes=attributes,
                context=context,
                source="spacy"
            )

            if entity.confidence >= self.confidence_threshold:
                entities.append(entity)

        # Enhance with LLM if enabled
        if self.use_llm and len(entities) < 50:  # Only for smaller entity sets
            llm_entities = self._extract_with_llm(text, document_id)
            entities.extend(llm_entities)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Generate statistics
        statistics = self._generate_statistics(entities)

        result = EntityExtractionResult(
            entities=entities,
            document_id=document_id,
            statistics=statistics
        )

        logger.debug(f"Extracted {len(entities)} entities from document {document_id}")
        return result

    def extract_entities_batch(
        self,
        texts: List[Tuple[str, str]],  # (text, document_id)
        extract_attributes: bool = True,
        include_context: bool = True
    ) -> List[EntityExtractionResult]:
        """
        Extract entities from multiple texts in batch.

        Args:
            texts: List of (text, document_id) tuples
            extract_attributes: Whether to extract attributes
            include_context: Whether to include context

        Returns:
            List of EntityExtractionResult
        """
        if not self.available or self.nlp is None:
            logger.warning(
                "Batch entity extraction requested but spaCy model is unavailable; returning empty results"
            )
            return [
                EntityExtractionResult(
                    entities=[],
                    document_id=doc_id,
                    metadata={"disabled": True, "reason": getattr(self, "_spacy_unavailable_reason", None)},
                    statistics={}
                )
                for _, doc_id in texts
            ]

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Process batch with spaCy
            docs = list(self.nlp.pipe(
                [text for text, _ in batch],
                batch_size=self.batch_size
            ))

            # Extract entities from each doc
            for doc, (text, doc_id) in zip(docs, batch):
                entities = []

                for ent in doc.ents:
                    context = None
                    if include_context:
                        start_ctx = max(0, ent.start_char - 50)
                        end_ctx = min(len(text), ent.end_char + 50)
                        context = text[start_ctx:end_ctx]

                    attributes = {}
                    if extract_attributes:
                        attributes = self._extract_entity_attributes(ent, doc)

                    entity = Entity(
                        text=ent.text,
                        type=self._normalize_entity_type(ent.label_),
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=self._calculate_confidence(ent),
                        attributes=attributes,
                        context=context,
                        source="spacy"
                    )

                    if entity.confidence >= self.confidence_threshold:
                        entities.append(entity)

                entities = self._deduplicate_entities(entities)
                statistics = self._generate_statistics(entities)

                results.append(EntityExtractionResult(
                    entities=entities,
                    document_id=doc_id,
                    statistics=statistics
                ))

        logger.info(f"Batch extracted entities from {len(texts)} documents")
        return results

    def _extract_entity_attributes(
        self,
        entity: Span,
        doc: Doc
    ) -> Dict[str, Any]:
        """Extract additional attributes for an entity."""
        attributes = {}

        # Get dependency information
        if entity.root.dep_:
            attributes["dependency"] = entity.root.dep_

        # Get head token
        if entity.root.head:
            attributes["head"] = entity.root.head.text

        # Get entity sentiment (if available)
        if hasattr(entity, "sentiment"):
            attributes["sentiment"] = entity.sentiment

        # For persons, try to extract titles
        if entity.label_ == "PERSON":
            # Look for titles before the name
            if entity.start > 0:
                prev_token = doc[entity.start - 1]
                if prev_token.text.lower() in ["mr", "mrs", "ms", "dr", "prof", "president"]:
                    attributes["title"] = prev_token.text

        # For organizations, try to extract organization type
        if entity.label_ == "ORG":
            org_types = ["company", "corporation", "inc", "ltd", "university", "institute"]
            for token in entity:
                if token.text.lower() in org_types:
                    attributes["org_type"] = token.text.lower()
                    break

        return attributes

    def _extract_with_llm(
        self,
        text: str,
        document_id: str
    ) -> List[Entity]:
        """Extract entities using LLM for complex/domain-specific entities."""
        if not self.llm_client:
            return []

        # Prepare prompt
        prompt = f"""Extract named entities from the following text. Focus on domain-specific entities like technical concepts, methods, datasets, and metrics.

Return a JSON array of entities with this format:
{{
  "text": "entity text",
  "type": "CONCEPT|TECHNOLOGY|METHOD|METRIC|DATASET",
  "confidence": 0.0-1.0
}}

Text:
{text[:2000]}  # Limit text length for LLM

Return only the JSON array, no other text."""

        try:
            if self.settings.llm.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.settings.llm.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting domain-specific named entities."},
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
            entities_data = json.loads(content)

            entities = []
            for ent_data in entities_data:
                # Find entity position in text
                start_char = text.find(ent_data["text"])
                if start_char == -1:
                    continue

                entity = Entity(
                    text=ent_data["text"],
                    type=ent_data["type"],
                    start_char=start_char,
                    end_char=start_char + len(ent_data["text"]),
                    confidence=ent_data.get("confidence", 0.8),
                    source="llm"
                )

                if entity.confidence >= self.confidence_threshold:
                    entities.append(entity)

            logger.debug(f"LLM extracted {len(entities)} additional entities")
            return entities

        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []

    def _normalize_entity_type(self, label: str) -> str:
        """Normalize entity type labels."""
        # Map spaCy labels to our standard types
        label_map = {
            "PERSON": "PERSON",
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "ORGANIZATION": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "LOCATION": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "WORK_OF_ART": "WORK_OF_ART",
            "LAW": "LAW",
            "LANGUAGE": "LANGUAGE",
            "FAC": "FACILITY",
            "FACILITY": "FACILITY",
            "NORP": "NORP"
        }

        return label_map.get(label, label)

    def _calculate_confidence(self, entity: Span) -> float:
        """Calculate confidence score for an entity."""
        # Base confidence from spaCy
        confidence = 0.8

        # Adjust based on entity length (very short entities are less reliable)
        if len(entity.text) <= 2:
            confidence *= 0.7

        # Adjust based on capitalization (proper nouns are more reliable)
        if entity.text[0].isupper():
            confidence *= 1.1

        # Adjust based on entity type (some types are more reliable)
        reliable_types = {"DATE", "MONEY", "PERCENT", "TIME"}
        if entity.label_ in reliable_types:
            confidence *= 1.2

        return min(confidence, 1.0)

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities, keeping the one with highest confidence.

        Handles:
        - Exact duplicates
        - Overlapping spans
        - Case variations
        """
        if not entities:
            return []

        # Sort by confidence (descending)
        entities.sort(key=lambda e: e.confidence, reverse=True)

        # Remove exact duplicates and repeated mentions
        seen_spans = set()
        seen_text_type = set()
        unique_entities = []

        for entity in entities:
            # Create a normalized key
            span_key = (entity.text.lower(), entity.type, entity.start_char, entity.end_char)
            text_key = (entity.text.lower(), entity.type)

            if span_key in seen_spans:
                continue

            seen_spans.add(span_key)

            # Only keep first occurrence per text/type combination
            if text_key in seen_text_type:
                continue

            seen_text_type.add(text_key)
            unique_entities.append(entity)

        # Remove overlapping entities (keep higher confidence)
        final_entities = []
        for entity in unique_entities:
            # Check if this entity overlaps with any already added
            overlaps = False
            for added_entity in final_entities:
                if self._entities_overlap(entity, added_entity):
                    overlaps = True
                    break

            if not overlaps:
                final_entities.append(entity)

        return final_entities

    def _entities_overlap(self, ent1: Entity, ent2: Entity) -> bool:
        """Check if two entities overlap in their spans."""
        return not (ent1.end_char <= ent2.start_char or ent2.end_char <= ent1.start_char)

    def _generate_statistics(self, entities: List[Entity]) -> Dict[str, int]:
        """Generate statistics about extracted entities."""
        stats = {
            "total_entities": len(entities),
            "by_type": defaultdict(int),
            "by_source": defaultdict(int),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0
        }

        for entity in entities:
            stats["by_type"][entity.type] += 1
            stats["by_source"][entity.source] += 1

            if entity.confidence >= 0.9:
                stats["high_confidence"] += 1
            elif entity.confidence >= 0.7:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1

        # Convert defaultdicts to regular dicts
        stats["by_type"] = dict(stats["by_type"])
        stats["by_source"] = dict(stats["by_source"])

        return stats

    def add_custom_entity_type(
        self,
        entity_type: str,
        patterns: List[str],
        case_sensitive: bool = False
    ) -> None:
        """
        Add a custom entity type with patterns.

        Args:
            entity_type: Custom entity type name
            patterns: List of pattern strings to match
            case_sensitive: Whether matching should be case-sensitive
        """
        if not self.available or self.nlp is None:
            logger.warning("Cannot add custom entity type because spaCy model is unavailable")
            return

        from spacy.matcher import PhraseMatcher

        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER" if not case_sensitive else "TEXT")
        patterns_list = [self.nlp.make_doc(pattern) for pattern in patterns]
        matcher.add(entity_type, patterns_list)

        @Language.component(f"custom_{entity_type}_matcher")
        def custom_type_component(doc):
            matches = matcher(doc)
            new_ents = []
            for match_id, start, end in matches:
                span = Span(doc, start, end, label=entity_type)
                new_ents.append(span)
            doc.ents = list(doc.ents) + new_ents
            return doc

        if f"custom_{entity_type}_matcher" not in self.nlp.pipe_names:
            self.nlp.add_pipe(f"custom_{entity_type}_matcher", last=True)
            logger.info(f"Added custom entity type: {entity_type}")

    def get_entity_types(self) -> Set[str]:
        """Get all available entity types."""
        base_types = {e.value for e in EntityType}
        return base_types
