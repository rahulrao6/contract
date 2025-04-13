"""
Text processor for contract analysis.
"""

import regex
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unidecode import unidecode  # For normalizing unicode text

logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a section of a contract."""
    section_id: str
    title: str
    text: str
    section_type: str
    is_header: bool = False
    section_number: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    level: int = 0
    position: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)

class TextProcessor:
    """Enhanced text processor for contract analysis."""
    
    def __init__(self, use_hierarchical_segmentation: bool = True):
        """
        Initialize text processor.
        
        Args:
            use_hierarchical_segmentation: Whether to use hierarchical segmentation
        """
        self.use_hierarchical_segmentation = use_hierarchical_segmentation
        
        # Section patterns with enhanced precision and coverage
        self.section_patterns = {
            "numbered": regex.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$'),
            "roman": regex.compile(r'^([IVX]+)\.?\s+(.+)$'),
            "alphabetical": regex.compile(r'^([A-Z])\.?\s+(.+)$'),
            "titled": regex.compile(r'^([A-Z][A-Z\s]+)[\.\:]\s*(.*)$'),
            "article": regex.compile(r'^(?:ARTICLE|Section)\s+(\d+|[IVX]+)(?:\.|\:|\s+\-)\s*(.+)$', regex.IGNORECASE),
            "schedule": regex.compile(r'^(?:SCHEDULE|APPENDIX|EXHIBIT)\s+([A-Z\d]+)(?:\.|\:|\s+\-)\s*(.+)?$', regex.IGNORECASE)
        }
        
        # Legal term indicators for section type classification
        self.section_type_indicators = {
            "covenant": [
                "agree", "covenant", "shall", "must", "undertake", "obligation", "duty", "perform", "commit", 
                "required to", "responsible for", "will provide"
            ],
            "representation": [
                "represent", "warranty", "warrants", "represents", "certify", "acknowledge", "confirm", 
                "declare", "guarantee", "assurance"
            ],
            "condition": [
                "condition", "subject to", "contingent", "dependent", "provided that", "only if", 
                "unless", "until", "when", "prerequisite"
            ],
            "termination": [
                "terminate", "termination", "cancel", "end", "expiration", "discontinue", "cease", 
                "rescind", "revoke", "void", "invalidate"
            ],
            "payment": [
                "payment", "fee", "cost", "price", "dollar", "pay", "compensation", "rate", "invoice", 
                "billing", "expense", "reimbursement"
            ],
            "liability": [
                "liability", "indemnify", "indemnification", "hold harmless", "responsible for", 
                "damages", "remedy", "claim", "reimburse", "defend", "settle"
            ],
            "definition": [
                "mean", "definition", "define", "interpret", "construe", "refer to", "shall mean", 
                "defined term", "for purposes of"
            ],
            "reference": [
                "refer to", "pursuant to", "accordance with", "subject to", "described in", 
                "set forth in", "specified in", "provided in"
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize contract text.
        
        Args:
            text: Raw contract text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Normalize unicode characters
        text = unidecode(text)
        
        # Remove null characters and other control characters except newlines and tabs
        text = regex.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)
        
        # Normalize whitespace but preserve paragraph structure
        text = regex.sub(r' +', ' ', text)
        text = regex.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR and formatting issues
        text = regex.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Fix missing spaces between words
        
        # Remove page numbers and headers/footers that might appear in formatted documents
        text = regex.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = regex.sub(r'\n\s*Page\s+\d+\s+of\s+\d+\s*\n', '\n', regex.IGNORECASE)  # Page X of Y format
        
        return text.strip()
    
    def segment_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Cleaned contract text
            
        Returns:
            List of paragraphs
        """
        if not text:
            return []
            
        # Split by double newlines to preserve paragraph structure
        raw_paragraphs = regex.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        paragraphs = []
        for p in raw_paragraphs:
            cleaned = p.strip()
            # Only keep paragraphs with meaningful content
            if len(cleaned) > 10 and not regex.match(r'^\s*[\d.]+\s*$', cleaned):  # Skip standalone numbers
                paragraphs.append(cleaned)
                
        return paragraphs
    
    def segment_document(self, text: str) -> List[Section]:
        """
        Segment document into sections.
        
        Args:
            text: Cleaned contract text
            
        Returns:
            List of Section objects
        """
        if self.use_hierarchical_segmentation:
            return self.hierarchical_section_segmentation(text)
        else:
            return self.flat_section_segmentation(text)
    
    def hierarchical_section_segmentation(self, text: str) -> List[Section]:
        """
        Segment document into hierarchical sections.
        
        Args:
            text: Cleaned contract text
            
        Returns:
            List of Section objects with hierarchical relationships
        """
        paragraphs = self.segment_into_paragraphs(text)
        sections: List[Section] = []
        
        current_hierarchy = {0: None}  # Level -> parent_section_id
        current_level = 0
        parent_id = None
        
        for idx, p in enumerate(paragraphs):
            # Skip very short paragraphs that are likely artifacts
            if len(p.strip()) < 5:
                continue
                
            is_header = False
            section_title = None
            section_number = None
            detected_level = 0
            
            # Check against section patterns
            for pattern_name, pattern in self.section_patterns.items():
                match = pattern.match(p.strip())
                if match:
                    is_header = True
                    section_number = match.group(1) if match.groups() else None
                    section_title = match.group(2) if len(match.groups()) > 1 else p.strip()
                    
                    # Determine hierarchy level based on pattern and numbering
                    if pattern_name == "numbered":
                        # Hierarchical numbering like 1.2.3
                        num_parts = section_number.count('.') + 1
                        detected_level = num_parts
                    elif pattern_name == "article":
                        detected_level = 1
                    elif pattern_name == "titled" or pattern_name == "schedule":
                        detected_level = 0  # Top level
                    elif pattern_name == "alphabetical":
                        detected_level = 2  # Usually subsections
                    elif pattern_name == "roman":
                        detected_level = 1  # Usually main sections
                        
                    break
            
            # If no match with patterns, try heuristic detection
            if not is_header and len(p.split()) <= 10:
                # Check for all-caps short paragraphs (likely headers)
                if p.isupper() and len(p) > 3:
                    is_header = True
                    section_title = p
                    detected_level = 0
                # Check for lines ending with colon (possible headers)
                elif p.strip().endswith(':') and len(p.split()) <= 5:
                    is_header = True
                    section_title = p.rstrip(':')
                    detected_level = 1
            
            # Determine section type based on content analysis
            section_type = self._determine_section_type(p)
            
            # Create section ID
            section_id = f"sec-{idx:03d}"
            
            # Update hierarchy tracking
            if is_header:
                # For headers, we update the hierarchy
                current_level = detected_level
                
                # Find parent ID based on the hierarchical level
                if detected_level == 0:
                    parent_id = None  # Top level has no parent
                else:
                    # Find parent at the level above
                    parent_levels = [level for level in current_hierarchy.keys() if level < detected_level]
                    if parent_levels:
                        parent_level = max(parent_levels)
                        parent_id = current_hierarchy.get(parent_level)
                    else:
                        parent_id = None
                
                # Update current hierarchy at this level and remove all deeper levels
                current_hierarchy[detected_level] = section_id
                for level in list(current_hierarchy.keys()):
                    if level > detected_level:
                        del current_hierarchy[level]
            else:
                # Regular content inherits the current hierarchy
                parent_id = current_hierarchy.get(current_level)
            
            # Calculate position information
            position = {
                "start_char": len("\n\n".join(paragraphs[:idx])) + 2*idx if idx > 0 else 0,
                "end_char": len("\n\n".join(paragraphs[:idx+1])) + 2*idx,
                "paragraph_idx": idx
            }
            
            # Create the section object
            section = Section(
                section_id=section_id,
                title=section_title if section_title else f"Section {idx + 1}",
                text=p,
                section_type=section_type,
                is_header=is_header,
                section_number=section_number,
                parent_id=parent_id,
                children_ids=[],
                level=detected_level,
                position=position
            )
            
            sections.append(section)
            
        # Update children_ids for parent-child relationships
        section_map = {section.section_id: section for section in sections}
        for section in sections:
            if section.parent_id and section.parent_id in section_map:
                section_map[section.parent_id].children_ids.append(section.section_id)
                
        return sections
    
    def flat_section_segmentation(self, text: str) -> List[Section]:
        """
        Segment document into flat sections.
        
        Args:
            text: Cleaned contract text
            
        Returns:
            List of Section objects without hierarchical relationships
        """
        paragraphs = self.segment_into_paragraphs(text)
        sections = []
        
        current_position = 0
        for idx, p in enumerate(paragraphs):
            is_header = False
            section_title = None
            section_number = None
            
            # Check against section patterns
            for pattern_name, pattern in self.section_patterns.items():
                match = pattern.match(p.strip())
                if match:
                    is_header = True
                    if len(match.groups()) > 1:
                        section_number = match.group(1)
                        section_title = match.group(2)
                    else:
                        section_title = match.group(1)
                    break
            
            # Fallback header detection
            if not is_header:
                if p.isupper() and len(p.split()) <= 5:
                    is_header = True
                    section_title = p
                
            # Determine section type
            section_type = self._determine_section_type(p)
            
            # Calculate position
            length = len(p)
            position = {
                "start_char": current_position,
                "end_char": current_position + length,
                "paragraph_idx": idx
            }
            
            # Create section ID
            section_id = f"sec-{idx:03d}"
            
            # Create the section
            section = Section(
                section_id=section_id,
                title=section_title if section_title else f"Section {idx + 1}",
                text=p,
                section_type=section_type,
                is_header=is_header,
                section_number=section_number,
                position=position
            )
            
            sections.append(section)
            current_position += length + 2  # Account for newlines
            
        return sections
    
    def _determine_section_type(self, text: str) -> str:
        """
        Determine the type of a section based on its content.
        
        Args:
            text: Section text
            
        Returns:
            Section type classification
        """
        text_lower = text.lower()
        
        # Check against section type indicators
        for section_type, indicators in self.section_type_indicators.items():
            for indicator in indicators:
                if indicator.lower() in text_lower:
                    return section_type
        
        # Default classifications
        if len(text.split()) < 10 and text.isupper():
            return "header"
        
        return "clause"  # Default