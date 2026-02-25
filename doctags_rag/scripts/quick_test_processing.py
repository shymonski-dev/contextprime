"""
Quick test script to verify processing pipeline works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextprime.processing import (
    create_pipeline,
    FileTypeDetector,
    TextCleaner,
)


def test_basic_functionality():
    """Test basic functionality of processing components."""
    print("Testing Contextprime Processing Pipeline")
    print("=" * 60)

    # Test 1: File type detection
    print("\n1. Testing File Type Detection...")
    test_path = Path(__file__)
    file_type = FileTypeDetector.detect_file_type(test_path)
    print(f"   ✓ Detected file type: {file_type}")

    # Test 2: Text cleaning
    print("\n2. Testing Text Cleaning...")
    dirty_text = "Hello\r\n\r\n\r\nWorld   with   spaces"
    clean_text = TextCleaner.clean_text(dirty_text, aggressive=True)
    print(f"   ✓ Cleaned text: '{clean_text}'")

    # Test 3: Pipeline creation
    print("\n3. Testing Pipeline Creation...")
    try:
        pipeline = create_pipeline(
            chunk_size=500,
            chunk_overlap=100
        )
        print(f"   ✓ Pipeline created successfully")
        print(f"   - Chunk size: {pipeline.config.chunk_size}")
        print(f"   - Chunk overlap: {pipeline.config.chunk_overlap}")
    except Exception as e:
        print(f"   ✗ Pipeline creation failed: {e}")
        return False

    # Test 4: Process sample file
    print("\n4. Testing Document Processing...")
    sample_file = Path(__file__).parent.parent / "data" / "samples" / "sample_text.txt"

    if not sample_file.exists():
        print(f"   ⚠ Sample file not found: {sample_file}")
        print("   Creating a simple test file...")

        # Create a simple test file
        test_content = """# Test Document

## Introduction

This is a test document for verating the processing pipeline.

## Section 1

Some content in section 1.

## Section 2

Some content in section 2.

- Item 1
- Item 2
- Item 3

## Conclusion

This concludes the test document.
"""
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(test_content, encoding='utf-8')
        print(f"   ✓ Created test file: {sample_file}")

    try:
        result = pipeline.process_file(sample_file)

        if result.success:
            print(f"   ✓ Processing successful!")
            print(f"   - Processing time: {result.processing_time:.2f}s")
            print(f"   - Elements parsed: {len(result.parsed_doc.elements)}")
            print(f"   - DocTags created: {len(result.doctags_doc.tags)}")
            print(f"   - Chunks created: {len(result.chunks)}")

            # Show first chunk
            if result.chunks:
                first_chunk = result.chunks[0]
                print(f"\n   First chunk preview:")
                print(f"   - ID: {first_chunk.chunk_id}")
                print(f"   - Length: {len(first_chunk.content)} chars")
                print(f"   - Content: {first_chunk.content[:100]}...")
        else:
            print(f"   ✗ Processing failed: {result.error}")
            return False

    except Exception as e:
        print(f"   ✗ Processing error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
