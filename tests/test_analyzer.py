"""
Tests for Contract Analyzer.
"""

import pytest
import asyncio
from src.contract_analyzer import ContractAnalyzer

@pytest.mark.asyncio
async def test_analyze_document_text(analyzer, sample_contract):
    """Test analyzing a document from text."""
    # Analyze sample contract
    result = await analyzer.analyze_document(text=sample_contract)
    
    # Check basic results
    assert result.status == "success"
    assert result.contract_id is not None
    assert len(result.sections) > 0
    assert result.processing_time_ms > 0
    
    # Check metadata
    assert result.metadata.word_count > 0
    assert "service" in result.metadata.detected_contract_type.lower()
    assert result.metadata.contract_type_confidence > 0.0
    
    # Check sections
    assert any("SERVICES" in s.title.upper() for s in result.sections)
    assert any("TERM" in s.title.upper() for s in result.sections)
    assert any("FEE" in s.title.upper() for s in result.sections)
    
    # Check risks
    assert len(result.risks) > 0
    risk_categories = set(risk.risk_category for risk in result.risks)
    assert len(risk_categories) > 0
    
    # Look for expected risks
    auto_renewal = any("auto" in risk.risk_name.lower() and "renew" in risk.risk_name.lower() for risk in result.risks)
    term_without_notice = any("termination" in risk.risk_name.lower() and "notice" in risk.risk_name.lower() for risk in result.risks)
    
    assert auto_renewal
    assert term_without_notice
    
    # Check summary
    assert len(result.summary.overall_summary) > 50
    assert len(result.summary.key_points) > 0
    
    # Check extracted data
    assert result.extracted_data is not None

@pytest.mark.asyncio
async def test_error_handling(analyzer):
    """Test error handling in analyzer."""
    # Test with empty text
    result = await analyzer.analyze_document(text="")
    
    assert result.status == "error"
    assert len(result.errors) > 0
    
    # Test with invalid file content
    result = await analyzer.analyze_document(file_content=b"invalid", filename="invalid.xyz")
    
    assert result.status == "error"
    assert len(result.errors) > 0