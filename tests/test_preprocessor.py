"""
Tests for document preprocessing.
"""

import pytest
import asyncio
from src.preprocessor.text_processor import TextProcessor
from src.preprocessor.document_parser import DocumentParser
import os

def test_clean_text():
    """Test text cleaning functionality."""
    processor = TextProcessor()
    
    # Test with various text artifacts
    text = "This  is  a  test\n\n\nwith  extra spaces\x00and\tcontrol\rcharacters."
    cleaned = processor.clean_text(text)
    
    assert "This is a test with extra spaces and control characters." == cleaned
    assert "\x00" not in cleaned  # Null bytes removed
    assert "  " not in cleaned  # Multiple spaces collapsed

def test_segment_into_paragraphs():
    """Test paragraph segmentation."""
    processor = TextProcessor()
    
    text = """Paragraph one.

Paragraph two.

Paragraph three with
multiple lines.
"""
    
    paragraphs = processor.segment_into_paragraphs(text)
    
    assert len(paragraphs) == 3
    assert "Paragraph one." in paragraphs[0]
    assert "Paragraph two." in paragraphs[1]
    assert "multiple lines" in paragraphs[2]

def test_hierarchical_section_segmentation(sample_contract):
    """Test hierarchical section segmentation."""
    processor = TextProcessor(use_hierarchical_segmentation=True)
    
    sections = processor.segment_document(sample_contract)
    
    # Check basics
    assert len(sections) > 5
    
    # Find section numbers
    section_1 = None
    section_1_1 = None
    section_2 = None
    
    for section in sections:
        if section.section_number == "1":
            section_1 = section
        elif section.section_number == "1.1":
            section_1_1 = section
        elif section.section_number == "2":
            section_2 = section
    
    # Check hierarchy
    assert section_1 is not None
    assert section_1_1 is not None
    assert section_2 is not None
    assert section_1_1.parent_id == section_1.section_id
    
    # Check section types
    assert "SERVICE" in section_1.title.upper()
    assert "TERM" in section_2.title.upper()

@pytest.mark.asyncio
async def test_document_parser(sample_contract_file):
    """Test document parser."""
    parser = DocumentParser()
    
    # Test with file content
    with open(sample_contract_file, 'rb') as f:
        file_content = f.read()
    
    parsed_text = await parser.parse_file(file_content, os.path.basename(sample_contract_file))
    
    # Check that key content was preserved
    assert "SERVICES AGREEMENT" in parsed_text
    assert "Effective Date" in parsed_text
    assert "ABC Company" in parsed_text
    assert "TERM AND TERMINATION" in parsed_text