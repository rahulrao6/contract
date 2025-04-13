"""
Pydantic models for Contract Analyzer API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AnalyzeTextRequest(BaseModel):
    """Request for text-based analysis."""
    text: str
    filename: Optional[str] = "document.txt"

class SectionSummary(BaseModel):
    """Summary of a contract section."""
    section_id: str
    title: str
    summary: str

class KeyPointResponse(BaseModel):
    """Key point from contract analysis."""
    point_id: str
    category: str
    description: str
    importance: str = "medium"
    extracted_value: Optional[str] = None

class RiskResponse(BaseModel):
    """Risk identified in contract analysis."""
    risk_id: str
    risk_name: str
    risk_level: str
    risk_category: str
    section_id: str
    description: str
    confidence: float
    suggested_changes: Optional[str] = None

class SectionResponse(BaseModel):
    """Contract section from analysis."""
    section_id: str
    title: str
    text: str
    section_type: str
    is_header: bool
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)

class MetadataResponse(BaseModel):
    """Contract metadata."""
    file_name: str
    file_type: str
    file_size_kb: float
    page_count: int
    word_count: int
    contract_type: str
    type_confidence: float
    has_pii: bool = False
    pii_types: List[str] = Field(default_factory=list)

class SummaryResponse(BaseModel):
    """Contract summary."""
    overall: str
    key_points: List[KeyPointResponse]
    basics: Dict[str, Any]

class AnalysisResponse(BaseModel):
    """API response for analysis results."""
    contract_id: str
    status: str
    processing_time_ms: int
    metadata: MetadataResponse
    summary: SummaryResponse
    risks_count: int
    risks_by_category: Dict[str, int]
    sections_count: int
    has_pii: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class DetailedAnalysisResponse(AnalysisResponse):
    """Detailed API response with full analysis results."""
    sections: List[SectionResponse]
    risks: List[RiskResponse]
    extracted_data: Dict[str, Any]
    cross_references: Dict[str, List[str]]