"""
Configuration settings for the Contract Analyzer system.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class Config:
    """Configuration settings for the system."""
    
    # Version
    VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # Model configurations
    LEGAL_BERT_MODEL: str = "nlpaueb/legal-bert-base-uncased"
    SUMMARIZER_MODEL: str = "facebook/bart-large-cnn"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NER_MODEL: str = "dslim/bert-base-NER"
    CUAD_MODEL: str = "contract-ai/CUAD_v1"
    
    # Processing settings
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 8
    MAX_DOCUMENT_SIZE_MB: int = 25
    MAX_SUMMARY_LENGTH: int = 300
    MIN_SUMMARY_LENGTH: int = 75
    EMBEDDING_DIMENSION: int = 768

    MIN_RISK_TEXT_LENGTH: int = 10  # Lowered from 20; adjust as needed

    
    # Device settings
    USE_CUDA: bool = torch.cuda.is_available()
    DEVICE: str = "cuda" if USE_CUDA else "cpu"
    
    # Feature flags
    ENABLE_HIERARCHICAL_SEGMENTATION: bool = True
    ENABLE_ADVANCED_RISK_DETECTION: bool = True
    ENABLE_PII_DETECTION: bool = True
    ENABLE_CUAD_ANALYSIS: bool = True
    ENABLE_CROSS_REFERENCE_DETECTION: bool = True
    ENABLE_SEMANTIC_SEARCH: bool = True
    
    # Risk detection thresholds
    RISK_CONFIDENCE_THRESHOLD: float = 0.6


    
    # Contract types for classification
    CONTRACT_TYPES: Dict[str, List[str]] = field(default_factory=lambda: {
        "lease": ["lease", "rent", "tenant", "landlord", "property", "premises", "rental", "leased", "occupancy"],
        "employment": ["employment", "salary", "employee", "employer", "hire", "compensation", "benefits", "termination of employment"],
        "nda": ["confidential", "non-disclosure", "confidentiality", "proprietary", "trade secret", "disclosure of information"],
        "service": ["service", "provider", "client", "deliverable", "statement of work", "services provided", "service level"],
        "subscription": ["subscription", "recurring", "monthly", "annual", "billing cycle", "renewal", "subscription term"],
        "license": ["license", "licensor", "licensee", "intellectual property", "royalty", "permitted use", "license grant"],
        "purchase": ["purchase", "buyer", "seller", "goods", "delivery", "price", "payment terms", "invoice"],
        "loan": ["loan", "lender", "borrower", "principal", "interest rate", "repayment", "default", "term loan"]
    })
    
    # Section type indicators
    SECTION_TYPE_INDICATORS: Dict[str, List[str]] = field(default_factory=lambda: {
        "covenant": ["agree", "covenant", "shall", "must", "undertake", "obligation", "duty", "perform"],
        "representation": ["represent", "warranty", "warrants", "certify", "acknowledge", "confirm", "declare"],
        "definition": ["mean", "definition", "defined", "interpret", "shall mean", "refers to", "defined term"],
        "termination": ["terminate", "termination", "cancel", "end", "expiration", "discontinue", "cease"],
        "payment": ["payment", "fee", "cost", "price", "pay", "compensation", "rate", "invoice", "billing"],
        "liability": ["liability", "indemnify", "indemnification", "hold harmless", "damages", "remedy"]
    })
    
    # API configuration
    API_TITLE: str = "Contract Analyzer API"
    API_DESCRIPTION: str = "Advanced API for analyzing legal contracts"
    API_VERSION: str = "v1.0"
    
    def __post_init__(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)