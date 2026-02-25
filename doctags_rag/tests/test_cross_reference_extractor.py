"""Unit tests for cross_reference_extractor.py.

Pure unit tests — no external dependencies, no fixtures.
"""

import pytest
from contextprime.processing.cross_reference_extractor import extract_cross_references, CrossRef


class TestExtractCrossReferences:
    """Tests for the extract_cross_references function."""

    def test_article_pattern_basic(self):
        refs = extract_cross_references("c1", "Article 6 applies.", "d1")
        assert len(refs) >= 1
        article_ref = next((r for r in refs if r.ref_type == "article"), None)
        assert article_ref is not None
        assert article_ref.target_label == "article_6"

    def test_article_pattern_with_paragraph(self):
        refs = extract_cross_references("c1", "See Article 17(3).", "d1")
        labels = [r.target_label for r in refs]
        assert "article_17(3)" in labels

    def test_article_pursuant_to(self):
        refs = extract_cross_references("c1", "pursuant to Article 4 of the regulation", "d1")
        assert any(r.ref_type == "article" for r in refs)

    def test_article_as_defined_in(self):
        refs = extract_cross_references("c1", "as defined in Article 4", "d1")
        assert any(r.ref_type == "article" for r in refs)

    def test_article_subject_to(self):
        refs = extract_cross_references("c1", "subject to Article 83 of this regulation", "d1")
        assert any(r.ref_type == "article" for r in refs)

    def test_section_pattern(self):
        refs = extract_cross_references("c1", "Section 12.3 states the requirement.", "d1")
        assert any(r.ref_type == "section" and r.target_label == "section_12.3" for r in refs)

    def test_schedule_pattern(self):
        refs = extract_cross_references("c1", "Schedule 2 applies to this provision.", "d1")
        assert any(r.ref_type == "schedule" for r in refs)

    def test_annex_roman_numeral(self):
        refs = extract_cross_references("c1", "Annex I requirements must be met.", "d1")
        assert any(r.ref_type == "schedule" for r in refs)

    def test_paragraph_pattern(self):
        refs = extract_cross_references("c1", "as set out in paragraph 3(a) below.", "d1")
        assert any(r.ref_type == "paragraph" for r in refs)

    def test_deduplication(self):
        refs = extract_cross_references("c1", "Article 6 and Article 6 both apply.", "d1")
        article_6_refs = [r for r in refs if r.target_label == "article_6"]
        assert len(article_6_refs) == 1

    def test_empty_content_returns_empty_list(self):
        refs = extract_cross_references("c1", "", "d1")
        assert refs == []

    def test_no_refs_returns_empty_list(self):
        refs = extract_cross_references(
            "c1",
            "The controller shall implement appropriate technical and organisational measures.",
            "d1",
        )
        assert refs == []

    def test_crossref_fields_populated(self):
        refs = extract_cross_references("c1", "Article 6 applies.", "d1")
        ref = next(r for r in refs if r.target_label == "article_6")
        assert ref.source_chunk_id == "c1"
        assert ref.doc_id == "d1"
        assert ref.ref_type == "article"
