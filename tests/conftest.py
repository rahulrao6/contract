"""
Fixtures for Contract Analyzer tests.
"""

import os
import pytest
import tempfile
from typing import Dict, List, Any, Optional

from src.config import Config
from src.contract_analyzer import ContractAnalyzer
from src.models.risk_detector import RiskDetectionModel
from src.models.summarizer import ContractSummarizer
from src.models.classifier import ContractClassifier
from src.models.extractor import ContractExtractor
from src.preprocessor.document_parser import DocumentParser
from src.preprocessor.text_processor import TextProcessor
from src.patterns.risk_patterns import RiskPatternLibrary
from src.security.privacy_manager import PrivacyManager

# Sample contract text for testing
SAMPLE_CONTRACT = """
SERVICES AGREEMENT

This Services Agreement (the "Agreement") is made and entered into as of January 1, 2023 (the "Effective Date"), 
by and between ABC Company, Inc., a Delaware corporation ("Company"), and XYZ Services LLC, a California limited 
liability company ("Provider").

1. SERVICES

1.1 Services. Provider shall provide to Company the services described in Exhibit A (the "Services").

1.2 Performance Standards. Provider shall perform the Services in a professional manner consistent with industry standards.

2. TERM AND TERMINATION

2.1 Term. This Agreement shall commence on the Effective Date and continue for a period of one (1) year, 
unless earlier terminated as provided herein (the "Initial Term"). This Agreement shall automatically renew 
for successive one-year terms unless either party provides written notice of non-renewal at least thirty (30) days
prior to the end of the then-current term.

2.2 Termination. Company may terminate this Agreement immediately without notice in the event of Provider's breach 
of Sections 3, 4, or 5.

3. FEES AND PAYMENT

3.1 Fees. Company shall pay Provider the fees set forth in Exhibit B (the "Fees"). Provider reserves the right to 
increase fees at any time upon 30 days' notice.

3.2 Payment Terms. Provider shall invoice Company monthly for the Fees. Company shall pay all undisputed amounts
within thirty (30) days of receipt of an invoice.
"""

@pytest.fixture
def config():
    """Fixture for Config."""
    # Create a test-specific config
    return Config()

@pytest.fixture
def analyzer(config):
    """Fixture for ContractAnalyzer."""
    return ContractAnalyzer(config)

@pytest.fixture
def sample_contract():
    """Fixture for sample contract text."""
    return SAMPLE_CONTRACT

@pytest.fixture
def sample_contract_file():
    """Fixture for sample contract file."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp.write(SAMPLE_CONTRACT.encode('utf-8'))
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

@pytest.fixture
def document_parser():
    """Fixture for DocumentParser."""
    return DocumentParser()

@pytest.fixture
def text_processor():
    """Fixture for TextProcessor."""
    return TextProcessor()

@pytest.fixture
def risk_library():
    """Fixture for RiskPatternLibrary."""
    return RiskPatternLibrary()

@pytest.fixture
def privacy_manager():
    """Fixture for PrivacyManager."""
    return PrivacyManager()