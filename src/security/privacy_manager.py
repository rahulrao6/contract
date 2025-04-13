"""
Privacy manager for handling PII detection and security.
"""

import re
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class PrivacyManager:
    """Manager for PII detection and security."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize privacy manager.
        
        Args:
            encryption_key: Optional encryption key for text encryption
        """
        # Initialize encryption if key provided
        self.encryption_key = encryption_key
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            self.cipher_suite = None
            
        # PII detection patterns
        self._init_pii_patterns()
            
    def _init_pii_patterns(self):
        """Initialize patterns for PII detection."""
        self.pii_patterns = {
            # Email with verification for TLD length and common patterns
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            
            # Phone with international and various formats
            "phone": re.compile(r'\b(?:\+\d{1,3}[ -]?)?(?:\(\d{1,4}\)|\d{1,4})[ -]?\d{1,4}[ -]?\d{1,9}\b'),
            
            # SSN with or without dashes
            "ssn": re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
            
            # Credit card with or without spaces/dashes
            "credit_card": re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
            
            # Address with improved street name recognition
            "address": re.compile(
                r'\b\d+\s+([A-Za-z]+\s+)+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way|Place|Pl|Terrace|Ter)\b',
                re.IGNORECASE
            ),
            
            # Full name patterns with titles
            "name": re.compile(r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),
            
            # Date of birth in various formats
            "dob": re.compile(r'\b(?:(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}|(?:19|20)\d{2}[/.-](?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01]))\b'),
            
            # Passport numbers - simplified pattern
            "passport": re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
            
            # Bank account / financial identification
            "financial_id": re.compile(r'\b(?:account|routing|ACH).{1,10}?(?:number|#).{1,10}?\d{4,17}\b', re.IGNORECASE)
        }
        
        # Legal entity patterns
        self.legal_entity_patterns = {
            # Company identifiers like EIN, Company numbers
            "company_id": re.compile(r'\b(?:EIN|Tax ID|Company Number|Registration Number)[\s:]+\d{2}[-]?\d{7}\b', re.IGNORECASE),
            
            # Contract numbers
            "contract_number": re.compile(r'\b(?:Contract|Agreement|License)\s+(?:No\.|Number|#)\s*[\w-]+\b', re.IGNORECASE),
            
            # Case citation
            "case_citation": re.compile(r'\b\d+\s+[A-Za-z\.]+\s+\d+\s+\(\d{4}\)\b')
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect personal identifiable information in text.
        
        Args:
            text: Text to analyze for PII
            
        Returns:
            Dictionary of PII types and instances
        """
        results = {}
        
        if not text:
            return results
        
        # Process text in chunks for more efficient processing
        chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
        
        for pii_type, pattern in self.pii_patterns.items():
            all_matches = []
            
            for chunk in chunks:
                matches = pattern.findall(chunk)
                if matches:
                    if isinstance(matches[0], tuple):
                        matches = [m[0] for m in matches]
                    all_matches.extend(matches)
            
            # Deduplicate matches
            if all_matches:
                results[pii_type] = list(set(all_matches))
                
        return results
    
    def detect_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Detect legal entities in text.
        
        Args:
            text: Text to analyze for legal entities
            
        Returns:
            Dictionary of entity types and instances
        """
        results = {}
        
        if not text:
            return results
        
        for entity_type, pattern in self.legal_entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[entity_type] = list(set(matches))
                
        return results
    
    def redact_pii(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Redact personal identifiable information from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted_text, detected_pii)
        """
        pii_detected = self.detect_pii(text)
        redacted_text = text
        
        for pii_type, instances in pii_detected.items():
            for instance in instances:
                redaction_id = hashlib.md5(instance.encode()).hexdigest()[:8]
                redaction = f"[REDACTED-{pii_type}-{redaction_id}]"
                redacted_text = redacted_text.replace(instance, redaction)
                
        return redacted_text, pii_detected
    
    def encrypt_text(self, text: str) -> str:
        """
        Encrypt text for secure storage.
        
        Args:
            text: Text to encrypt
            
        Returns:
            Encrypted text
        """
        if not self.cipher_suite:
            logger.warning("Encryption requested but no encryption key provided")
            return text
        
        try:
            encrypted = self.cipher_suite.encrypt(text.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            return "[ENCRYPTION ERROR]"

    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt text for processing.
        
        Args:
            encrypted_text: Encrypted text to decrypt
            
        Returns:
            Decrypted text
        """
        if not self.cipher_suite:
            logger.warning("Decryption requested but no encryption key provided")
            return encrypted_text
        
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_text.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            return "[DECRYPTION ERROR]"