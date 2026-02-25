"""
Demo script for document processing pipeline.

Demonstrates the complete workflow:
1. Processing single documents
2. Batch processing
3. Different output formats
4. Statistics and analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextprime.processing import (
    create_pipeline,
    PipelineConfig,
    DocTagsConverter,
)
from loguru import logger


def demo_single_file():
    """Demonstrate processing a single file."""
    logger.info("=" * 80)
    logger.info("Demo: Processing Single File")
    logger.info("=" * 80)

    # Path to sample document
    sample_file = Path(__file__).parent.parent / "data" / "samples" / "sample_text.txt"

    if not sample_file.exists():
        logger.error(f"Sample file not found: {sample_file}")
        return

    # Create pipeline with custom configuration
    pipeline = create_pipeline(
        chunk_size=800,
        chunk_overlap=150,
        ocr_engine='paddleocr'
    )

    # Process the file
    logger.info(f"Processing: {sample_file.name}")

    def progress_callback(stage, progress):
        logger.info(f"Stage: {stage}, Progress: {progress:.1%}")

    result = pipeline.process_file(sample_file, progress_callback=progress_callback)

    # Display results
    if result.success:
        logger.success(f"Processing completed in {result.processing_time:.2f}s")

        logger.info(f"\nParsed Document:")
        logger.info(f"  - Elements: {len(result.parsed_doc.elements)}")
        logger.info(f"  - Text length: {len(result.parsed_doc.text)} chars")
        logger.info(f"  - Parser: {result.parsed_doc.metadata.get('parser')}")

        logger.info(f"\nDocTags:")
        logger.info(f"  - Total tags: {len(result.doctags_doc.tags)}")
        logger.info(f"  - Title: {result.doctags_doc.title}")

        # Show tag type distribution
        tag_types = {}
        for tag in result.doctags_doc.tags:
            tag_type = tag.tag_type.value
            tag_types[tag_type] = tag_types.get(tag_type, 0) + 1

        logger.info(f"  - Tag types: {tag_types}")

        logger.info(f"\nChunks:")
        logger.info(f"  - Total chunks: {len(result.chunks)}")

        # Show first few chunks
        logger.info(f"\nFirst 3 chunks:")
        for i, chunk in enumerate(result.chunks[:3]):
            logger.info(f"\n  Chunk {i + 1}:")
            logger.info(f"    ID: {chunk.chunk_id}")
            logger.info(f"    Length: {len(chunk.content)} chars")
            logger.info(f"    Context: {chunk.context}")
            logger.info(f"    Preview: {chunk.content[:100]}...")

        # Show different output formats
        logger.info("\n" + "=" * 80)
        logger.info("Output Formats")
        logger.info("=" * 80)

        # Markdown
        markdown = DocTagsConverter.to_markdown(result.doctags_doc)
        logger.info(f"\nMarkdown output ({len(markdown)} chars):")
        logger.info(markdown[:500] + "...")

        # HTML
        html = DocTagsConverter.to_html(result.doctags_doc)
        logger.info(f"\nHTML output ({len(html)} chars):")
        logger.info(html[:500] + "...")

        # Plain text
        text = DocTagsConverter.to_text(result.doctags_doc)
        logger.info(f"\nPlain text output ({len(text)} chars):")
        logger.info(text[:500] + "...")

    else:
        logger.error(f"Processing failed: {result.error}")


def demo_batch_processing():
    """Demonstrate batch processing of multiple files."""
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Batch Processing")
    logger.info("=" * 80)

    # Get all sample files
    samples_dir = Path(__file__).parent.parent / "data" / "samples"
    sample_files = list(samples_dir.glob("*.*"))

    if not sample_files:
        logger.error("No sample files found")
        return

    logger.info(f"Found {len(sample_files)} sample files")

    # Create pipeline
    pipeline = create_pipeline(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Progress callback
    def progress_callback(processed, total):
        logger.info(f"Progress: {processed}/{total} files processed")

    # Process batch
    results = pipeline.process_batch(sample_files, progress_callback=progress_callback)

    # Display statistics
    stats = pipeline.get_statistics(results)

    logger.info("\n" + "=" * 80)
    logger.info("Batch Processing Statistics")
    logger.info("=" * 80)

    logger.info(f"\nOverall:")
    logger.info(f"  - Total files: {stats['total']}")
    logger.info(f"  - Successful: {stats['successful']}")
    logger.info(f"  - Failed: {stats['failed']}")
    logger.info(f"  - Success rate: {stats['success_rate']:.1%}")

    if stats['successful'] > 0:
        logger.info(f"\nProcessing:")
        logger.info(f"  - Total chunks: {stats['total_chunks']}")
        logger.info(f"  - Avg chunks/doc: {stats['avg_chunks_per_doc']:.1f}")
        logger.info(f"  - Total time: {stats['total_processing_time']:.2f}s")
        logger.info(f"  - Avg time/doc: {stats['avg_processing_time']:.2f}s")

    logger.info(f"\nFile types:")
    for file_type, count in stats.get('file_types', {}).items():
        logger.info(f"  - {file_type}: {count}")

    # Show individual results
    logger.info("\n" + "=" * 80)
    logger.info("Individual Results")
    logger.info("=" * 80)

    for result in results:
        status = "✓" if result.success else "✗"
        logger.info(f"\n{status} {result.file_path.name}")
        if result.success:
            logger.info(f"  - Chunks: {len(result.chunks)}")
            logger.info(f"  - Time: {result.processing_time:.2f}s")
        else:
            logger.info(f"  - Error: {result.error}")


def demo_save_outputs():
    """Demonstrate saving intermediate outputs."""
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Saving Outputs")
    logger.info("=" * 80)

    sample_file = Path(__file__).parent.parent / "data" / "samples" / "sample_markdown.md"

    if not sample_file.exists():
        logger.error(f"Sample file not found: {sample_file}")
        return

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)

    # Create pipeline with output saving enabled
    config = PipelineConfig(
        chunk_size=1000,
        chunk_overlap=200,
        save_intermediate=True,
        save_json=True,
        save_markdown=True,
        output_dir=output_dir
    )

    from contextprime.processing import DocumentProcessingPipeline
    pipeline = DocumentProcessingPipeline(config)

    # Process file
    logger.info(f"Processing: {sample_file.name}")
    result = pipeline.process_file(sample_file)

    if result.success:
        logger.success("Processing completed")

        # Check output files
        doc_output_dir = output_dir / sample_file.stem
        logger.info(f"\nOutput saved to: {doc_output_dir}")

        if doc_output_dir.exists():
            output_files = list(doc_output_dir.glob("*"))
            logger.info(f"Output files:")
            for file in output_files:
                size = file.stat().st_size / 1024  # KB
                logger.info(f"  - {file.name} ({size:.1f} KB)")
        else:
            logger.warning("Output directory not created")
    else:
        logger.error(f"Processing failed: {result.error}")


def demo_structure_analysis():
    """Demonstrate structure analysis of DocTags."""
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Structure Analysis")
    logger.info("=" * 80)

    sample_file = Path(__file__).parent.parent / "data" / "samples" / "sample_text.txt"

    if not sample_file.exists():
        logger.error(f"Sample file not found: {sample_file}")
        return

    pipeline = create_pipeline()
    result = pipeline.process_file(sample_file)

    if not result.success:
        logger.error(f"Processing failed: {result.error}")
        return

    doctags_doc = result.doctags_doc

    # Analyze structure
    logger.info(f"\nDocument: {doctags_doc.title}")
    logger.info(f"Document ID: {doctags_doc.doc_id}")

    # Build hierarchy tree
    logger.info("\nDocument Structure:")

    def print_tag_tree(tags, parent_id=None, indent=0):
        """Recursively print tag tree."""
        for tag in tags:
            if tag.parent_id == parent_id:
                indent_str = "  " * indent
                type_str = tag.tag_type.value
                content_preview = tag.content[:50].replace('\n', ' ')

                logger.info(f"{indent_str}- [{type_str}] {content_preview}...")

                # Print children
                print_tag_tree(tags, tag.tag_id, indent + 1)

    # Find root tag
    root_tags = [t for t in doctags_doc.tags if t.parent_id is None]
    for root in root_tags:
        print_tag_tree(doctags_doc.tags, root.tag_id, 0)

    # Section analysis
    logger.info("\nSections:")
    from contextprime.processing.doctags_processor import DocTagType

    sections = [t for t in doctags_doc.tags if t.tag_type == DocTagType.SECTION]
    for section in sections:
        logger.info(f"  - {section.content} (level {section.level})")

    # Hierarchy info
    logger.info(f"\nHierarchy:")
    logger.info(f"  - Sections: {len(doctags_doc.hierarchy.get('sections', {}))}")


def main():
    """Run all demos."""
    logger.info("Document Processing Pipeline Demo")
    logger.info("=" * 80)

    try:
        # Demo 1: Single file processing
        demo_single_file()

        # Demo 2: Batch processing
        demo_batch_processing()

        # Demo 3: Save outputs
        demo_save_outputs()

        # Demo 4: Structure analysis
        demo_structure_analysis()

        logger.success("\n" + "=" * 80)
        logger.success("All demos completed successfully!")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
