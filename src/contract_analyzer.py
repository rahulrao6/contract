"""
Contract Analyzer - main module for legal contract analysis.
"""

import os
import json
import uuid
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re  # Required for regex matching

from src.config import Config
from src.models.risk_detector import RiskDetectionModel
from src.models.classifier import ContractClassifier
from src.models.summarizer import ContractSummarizer
from src.models.extractor import ContractExtractor
from src.preprocessor.document_parser import DocumentParser
from src.preprocessor.text_processor import TextProcessor, Section
from src.patterns.risk_patterns import RiskPatternLibrary
from src.patterns.clause_patterns import ClausePatternLibrary
from src.patterns.legal_terms import LegalTermLibrary
from src.security.privacy_manager import PrivacyManager
from src.utils.helpers import create_unique_id

logger = logging.getLogger(__name__)

# Data classes to hold analysis results
@dataclass
class DocumentMetadata:
    """Metadata about an analyzed document."""
    file_name: str
    file_type: str
    file_size_kb: float
    page_count: int
    language: str = "en"
    word_count: int = 0
    processing_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    detected_contract_type: Optional[str] = None
    contract_type_confidence: float = 0.0
    extracted_dates: List[str] = field(default_factory=list)
    extracted_parties: List[str] = field(default_factory=list)
    contains_pii: bool = False
    pii_types: List[str] = field(default_factory=list)

@dataclass
class RiskItem:
    """Represents a risk identified in a contract."""
    risk_id: str
    section_id: str
    risk_name: str
    risk_level: str
    risk_category: str
    risk_description: str
    original_text: str = ""
    context: str = ""
    confidence: float = 1.0
    detection_method: str = "rule_based"
    suggested_changes: str = ""
    suggested_questions: List[str] = field(default_factory=list)
    standard_assessment: str = "unusual"
    cross_references: List[str] = field(default_factory=list)

@dataclass
class KeyPoint:
    """Key point from a contract."""
    point_id: str
    category: str
    description: str
    importance: str = "medium"
    context: str = ""
    extracted_value: str = ""

@dataclass
class ContractSummary:
    """Summary of contract analysis."""
    overall_summary: str
    key_points: List[KeyPoint]
    section_summaries: List[Dict[str, Any]]
    contract_basics: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """Complete result of contract analysis."""
    contract_id: str
    metadata: DocumentMetadata
    sections: List[Section]
    risks: List[RiskItem]
    summary: ContractSummary
    extracted_data: Dict[str, Any]
    cross_references: Dict[str, List[str]]
    processing_time_ms: int
    has_pii: bool = False
    status: str = "success"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ContractAnalyzer:
    """Main class for comprehensive contract analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the contract analyzer.
        
        Args:
            config: Configuration settings, or use defaults.
        """
        self.config = config or Config()
        
        # Initialize core components with extended fallback logic
        self.document_parser = DocumentParser(max_file_size_mb=self.config.MAX_DOCUMENT_SIZE_MB)
        self.text_processor = TextProcessor(use_hierarchical_segmentation=self.config.ENABLE_HIERARCHICAL_SEGMENTATION)
        self.risk_library = RiskPatternLibrary()
        self.clause_library = ClausePatternLibrary()
        self.term_library = LegalTermLibrary()
        self.privacy_manager = PrivacyManager()
        
        # Lazy-loaded models
        self._risk_model = None
        self._classifier_model = None
        self._summarizer_model = None
        self._extractor_model = None
        
        logger.info("Contract Analyzer initialized with config: %s", 
                    {k: v for k, v in vars(self.config).items() 
                     if not k.startswith('_') and k.isupper()})
    
    @property
    def risk_model(self) -> RiskDetectionModel:
        """Lazy load risk detection model."""
        if self._risk_model is None:
            self._risk_model = RiskDetectionModel(
                model_name=self.config.LEGAL_BERT_MODEL,
                cache_dir=self.config.CACHE_DIR,
                device=self.config.DEVICE
            )
        return self._risk_model
    
    @property
    def classifier_model(self) -> ContractClassifier:
        """Lazy load contract classifier model."""
        if self._classifier_model is None:
            self._classifier_model = ContractClassifier(
                model_name=self.config.LEGAL_BERT_MODEL,
                cache_dir=self.config.CACHE_DIR,
                device=self.config.DEVICE
            )
        return self._classifier_model
    
    @property
    def summarizer_model(self) -> ContractSummarizer:
        """Lazy load summarizer model."""
        if self._summarizer_model is None:
            self._summarizer_model = ContractSummarizer(
                model_name=self.config.SUMMARIZER_MODEL,
                cache_dir=self.config.CACHE_DIR,
                device=self.config.DEVICE
            )
        return self._summarizer_model
    
    @property
    def extractor_model(self) -> ContractExtractor:
        """Lazy load information extractor model."""
        if self._extractor_model is None:
            self._extractor_model = ContractExtractor(
                model_name=self.config.LEGAL_BERT_MODEL,
                cache_dir=self.config.CACHE_DIR,
                device=self.config.DEVICE
            )
        return self._extractor_model
    
    async def analyze_document(self, 
                               file_content: Optional[bytes] = None, 
                               filename: Optional[str] = None,
                               text: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a contract document.
        
        Args:
            file_content: Binary file content.
            filename: Name of the file.
            text: Plain text content (alternative to file).
            
        Returns:
            Complete analysis results.
        """
        start_time = time.time()
        contract_id = str(uuid.uuid4())
        warnings = []
        errors = []
        
        try:
            # Extract text from file or use provided text
            if file_content and filename:
                raw_text = await self.document_parser.parse_file(file_content, filename)
                file_name = filename
            elif text:
                raw_text = text
                file_name = "document.txt"
            else:
                raise ValueError("Either file_content and filename, or text must be provided")
                
            # Clean and normalize text
            normalized_text = self.text_processor.clean_text(raw_text)
            
            # Handle PII detection and redaction
            if self.config.ENABLE_PII_DETECTION:
                redacted_text, pii_result = self.privacy_manager.redact_pii(normalized_text)
                has_pii = bool(pii_result)
                pii_types = list(pii_result.keys()) if pii_result else []
                if has_pii:
                    processing_text = redacted_text
                    logger.info(f"PII detected in document: {pii_types}")
                else:
                    processing_text = normalized_text
            else:
                has_pii = False
                pii_types = []
                processing_text = normalized_text
            
            # Extract document metadata with extended fallback checking
            metadata = self._extract_metadata(file_name, normalized_text, has_pii, pii_types)
            
            # Segment document into sections using robust segmentation methods
            sections = self.text_processor.segment_document(processing_text)
            
            # Extract cross-references if enabled
            cross_references = self._extract_cross_references(sections) if self.config.ENABLE_CROSS_REFERENCE_DETECTION else {}
            
            # Enhance each section with detected legal entities
            for section in sections:
                entities = self.privacy_manager.detect_legal_entities(section.text)
                section.extracted_entities.update(entities)
            
            # Run rule-based risk analysis with fallback to ML-based detection if enabled
            risks = self._analyze_risks(sections)
            
            # Generate overall summary from important sections
            summary = self._generate_summary(processing_text, sections, metadata.detected_contract_type)
            
            # Extract structured information using both ML and regex-based methods
            extracted_data = self._extract_information(processing_text)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return AnalysisResult(
                contract_id=contract_id,
                metadata=metadata,
                sections=sections,
                risks=risks,
                summary=summary,
                extracted_data=extracted_data,
                cross_references=cross_references,
                processing_time_ms=processing_time_ms,
                has_pii=has_pii,
                status="success",
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}", exc_info=True)
            processing_time_ms = int((time.time() - start_time) * 1000)
            return AnalysisResult(
                contract_id=contract_id,
                metadata=DocumentMetadata(
                    file_name=filename or "unknown.txt",
                    file_type="unknown",
                    file_size_kb=0,
                    page_count=0
                ),
                sections=[],
                risks=[],
                summary=ContractSummary(
                    overall_summary="Analysis failed",
                    key_points=[],
                    section_summaries=[],
                    contract_basics={},
                    meta={}
                ),
                extracted_data={},
                cross_references={},
                processing_time_ms=processing_time_ms,
                status="error",
                errors=[str(e)]
            )
    
    def _extract_metadata(self, file_name: str, text: str, has_pii: bool, pii_types: List[str]) -> DocumentMetadata:
        """
        Extract metadata from document with detailed checks.
        
        Args:
            file_name: Name of the file.
            text: Normalized text.
            has_pii: Whether PII was detected.
            pii_types: Types of PII detected.
            
        Returns:
            Document metadata.
        """
        metadata = DocumentMetadata(
            file_name=file_name,
            file_type=file_name.split('.')[-1] if '.' in file_name else "txt",
            file_size_kb=round(len(text.encode('utf-8')) / 1024, 2),
            page_count=text.count("\f") + 1,
            language="en",
            word_count=len(text.split()),
            contains_pii=has_pii,
            pii_types=pii_types
        )
        
        # Detect key dates and parties using regex patterns
        dates_pattern = r'(?:dated|effective|as of|dated as of)\s+(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:day of\s+)?\w+,?\s+\d{4})'
        parties_pattern = r'(?:between|by and between|among)\s+([^,]+?)\s+(?:and|,)\s+([^,\.]+)'
        dates = re.findall(dates_pattern, text, re.IGNORECASE)
        parties = []
        for match in re.finditer(parties_pattern, text, re.IGNORECASE):
            if match.groups():
                for group in match.groups():
                    if group.strip() and len(group.strip()) > 3:
                        parties.append(group.strip())
        metadata.extracted_dates = list(set(dates))[:5]
        metadata.extracted_parties = list(set(parties))[:5]
        
        # Classify contract type based on keyword counts
        type_scores = {}
        for ctype, keywords in self.config.CONTRACT_TYPES.items():
            score = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower())) for keyword in keywords)
            type_scores[ctype] = score
        if max(type_scores.values(), default=0) > 0:
            most_likely_type = max(type_scores.items(), key=lambda x: x[1])[0]
            type_confidence = min(max(type_scores[most_likely_type] / (len(self.config.CONTRACT_TYPES[most_likely_type]) * 2), 0.5), 0.95)
        else:
            most_likely_type = "general"
            type_confidence = 0.5
        metadata.detected_contract_type = most_likely_type
        metadata.contract_type_confidence = type_confidence
        
        return metadata
    
    def _extract_cross_references(self, sections: List[Section]) -> Dict[str, List[str]]:
        """
        Extract cross-references between sections.
        
        Args:
            sections: List of document sections.
            
        Returns:
            Dictionary mapping section IDs to referenced section IDs.
        """
        cross_references = {}
        section_map = {section.section_id: section for section in sections}
        section_numbers = {}
        for section in sections:
            if section.section_number:
                section_numbers[section.section_number] = section.section_id
        ref_patterns = [
            r'(?:pursuant to|as provided in|as set forth in|in accordance with|as defined in|subject to|as described in)\s+(?:Section|Article|Paragraph|Clause)\s+(\d+(?:\.\d+)*)',
            r'(?:Section|Article|Paragraph|Clause)\s+(\d+(?:\.\d+)*)\s+(?:hereof|above|below)',
            r'(?:see|refer to)\s+(?:Section|Article|Paragraph|Clause)\s+(\d+(?:\.\d+)*)'
        ]
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in ref_patterns]
        for section in sections:
            refs = []
            for pattern in compiled_patterns:
                matches = pattern.findall(section.text)
                refs.extend(matches)
            if refs:
                target_sections = []
                for ref in refs:
                    if ref in section_numbers:
                        target_id = section_numbers[ref]
                        if target_id != section.section_id:
                            target_sections.append(target_id)
                if target_sections:
                    cross_references[section.section_id] = target_sections
        return cross_references
    
    def _analyze_risks(self, sections: List[Section]) -> List[RiskItem]:
        """
        Analyze risks in contract sections using both rule-based and ML methods.
        
        Args:
            sections: List of document sections.
            
        Returns:
            List of identified risk items.
        """
        risks = []
        
        # Rule-based detection
        for i, section in enumerate(sections):
            if len(section.text.split()) < 10 or section.is_header:
                continue
            section_risks = self.risk_library.detect_risks(section.text)
            for risk in section_risks:
                risk_item = RiskItem(
                    risk_id=f"risk-{len(risks):03d}",
                    section_id=section.section_id,
                    risk_name=risk['risk_name'],
                    risk_level=risk['risk_level'],
                    risk_category=risk['risk_category'],
                    risk_description=risk['explanation'],
                    original_text=risk.get('matched_text', ''),
                    context=risk.get('context', ''),
                    confidence=risk.get('confidence', 1.0),
                    detection_method='rule_based',
                    suggested_changes=risk.get('suggested_alternative', '')
                )
                risks.append(risk_item)
        
        # ML-based risk detection (if enabled)
        if self.config.ENABLE_ADVANCED_RISK_DETECTION and len(sections) > 0:
            try:
                section_texts = [s.text for s in sections if len(s.text.split()) >= 10 and not s.is_header]
                section_ids = [s.section_id for s in sections if len(s.text.split()) >= 10 and not s.is_header]
                if section_texts:
                    risk_predictions = self.risk_model.predict_risks(
                        section_texts, 
                        threshold=self.config.RISK_CONFIDENCE_THRESHOLD
                    )
                    for prediction in risk_predictions:
                        section_idx = prediction['text_idx']
                        if section_idx < len(section_ids):
                            section_id = section_ids[section_idx]
                            risk_name = prediction['risk_name']
                            if any(r.section_id == section_id and r.risk_name.lower() == risk_name.lower() for r in risks):
                                continue
                            risk_item = RiskItem(
                                risk_id=f"risk-{len(risks):03d}",
                                section_id=section_id,
                                risk_name=risk_name,
                                risk_level=prediction['risk_level'],
                                risk_category=prediction['risk_category'],
                                risk_description=f"ML detected potential {risk_name} risk in this section.",
                                context=section_texts[section_idx][:300] + "...",
                                confidence=prediction['confidence'],
                                detection_method='ml'
                            )
                            risks.append(risk_item)
            except Exception as e:
                logger.warning(f"Error in ML risk detection: {str(e)}")
        return risks
    
    def _generate_summary(self, text: str, sections: List[Section], contract_type: str) -> ContractSummary:
        """
        Generate a summary of the contract using both section summaries and full-text summarization.
        
        Args:
            text: Full document text.
            sections: List of document sections.
            contract_type: Detected contract type.
            
        Returns:
            ContractSummary object.
        """
        important_sections = [s for s in sections if s.section_type in ["covenant", "representation", "liability", 
                                                                        "termination", "payment", "definition"] or len(s.text.split()) > 100]
        if len(important_sections) > 10:
            important_sections = sorted(important_sections, key=lambda s: len(s.text.split()), reverse=True)[:10]
        section_summaries = []
        if important_sections:
            section_texts = [section.text for section in important_sections]
            summaries = self.summarizer_model.summarize(
                section_texts, 
                max_length=self.config.MAX_SUMMARY_LENGTH,
                min_length=self.config.MIN_SUMMARY_LENGTH
            )
            for section, summary in zip(important_sections, summaries):
                if summary.strip():
                    section_summaries.append({
                        "section_id": section.section_id,
                        "title": section.title,
                        "summary": summary
                    })
        overall_summary = "Contract analysis could not generate a summary."
        try:
            if len(text) > 10000:
                combined_text = " ".join([f"{s['title']}: {s['summary']}" for s in section_summaries])
                if combined_text:
                    summary_result = self.summarizer_model.summarize(
                        [combined_text],
                        max_length=self.config.MAX_SUMMARY_LENGTH,
                        min_length=self.config.MIN_SUMMARY_LENGTH
                    )
                    if summary_result and summary_result[0]:
                        overall_summary = summary_result[0]
            else:
                summary_result = self.summarizer_model.summarize(
                    [text[:10000]],
                    max_length=self.config.MAX_SUMMARY_LENGTH,
                    min_length=self.config.MIN_SUMMARY_LENGTH
                )
                if summary_result and summary_result[0]:
                    overall_summary = summary_result[0]
            if contract_type != "general" and contract_type not in overall_summary.lower():
                overall_summary = f"This {contract_type} agreement " + overall_summary[0].lower() + overall_summary[1:]
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            overall_summary = f"This appears to be a {contract_type} agreement. The system was unable to generate a detailed summary."
        
        key_points = self._generate_key_points(text, sections)
        contract_basics = {
            "type": contract_type,
            "sections_count": len(sections),
            "estimated_length": "short" if len(text) < 5000 else "medium" if len(text) < 20000 else "long"
        }
        return ContractSummary(
            overall_summary=overall_summary,
            key_points=key_points,
            section_summaries=section_summaries,
            contract_basics=contract_basics,
            meta={"summary_generation_time": datetime.utcnow().isoformat()}
        )
    
    def _generate_key_points(self, text: str, sections: List[Section]) -> List[KeyPoint]:
        """
        Generate key points from contract text.
        
        Args:
            text: Full document text.
            sections: List of document sections.
            
        Returns:
            List of KeyPoint objects.
        """
        key_points = []
        point_id_counter = 0
        
        extracted_info = self._extract_information(text)
        
        for term_type, info in extracted_info.items():
            if info and info.get('value'):
                key_points.append(KeyPoint(
                    point_id=f"point-{point_id_counter:03d}",
                    category=term_type,
                    description=f"{term_type.replace('_', ' ').title()}: {info['value']}",
                    importance="high",
                    extracted_value=info['value']
                ))
                point_id_counter += 1
        
        important_sections = [s for s in sections if s.section_type in ["covenant", "representation", "termination", 
                                                                        "payment", "liability"]]
        for section in important_sections[:5]:
            key_points.append(KeyPoint(
                point_id=f"point-{point_id_counter:03d}",
                category="section",
                description=f"Important {section.section_type} clause: {section.title}",
                importance="medium",
                context=section.text[:100] + "..." if len(section.text) > 100 else section.text,
                extracted_value=section.title
            ))
            point_id_counter += 1
        
        return key_points
    
    def _extract_information(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract structured information from the contract.
        
        Args:
            text: Document text.
            
        Returns:
            Dictionary with extraction results.
        """
        extraction_types = [
            "parties", "effective_date", "termination_date", 
            "governing_law", "venue", "notice_period"
        ]
        try:
            extracted_info = self.extractor_model.extract_information(text, extraction_types)
        except Exception as e:
            logger.error(f"Error in information extraction: {str(e)}")
            extracted_info = {ext_type: {"value": "", "confidence": 0.0, "method": "none"} 
                              for ext_type in extraction_types}
        return extracted_info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Contract Analyzer on a sample file")
    parser.add_argument("file", help="Path to contract file")
    args = parser.parse_args()
    analyzer = ContractAnalyzer()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(analyzer.analyze_document(file_content=open(args.file, 'rb').read(), filename=args.file))
    print(json.dumps(asdict(result), indent=2))
