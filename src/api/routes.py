"""
API routes for Contract Analyzer.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from src.config import Config
from src.contract_analyzer import ContractAnalyzer, AnalysisResult
from src.api.models import AnalyzeTextRequest, AnalysisResponse, DetailedAnalysisResponse
from src.utils import asdict

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1")

# Dependency to get analyzer instance
def get_analyzer():
    """Dependency to get analyzer instance."""
    config = Config()
    return ContractAnalyzer(config)

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Contract Analyzer API", "version": Config().VERSION}

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": Config().VERSION}

@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    analyzer: ContractAnalyzer = Depends(get_analyzer)
):
    """
    Analyze a contract document file.
    
    Args:
        file: Contract document file
        analyzer: Contract analyzer instance
        
    Returns:
        Analysis summary
    """
    try:
        file_content = await file.read()
        
        # Analyze document
        result = await analyzer.analyze_document(
            file_content=file_content, 
            filename=file.filename
        )
        
        # Convert to response format
        return _create_response(result)
        
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    analyzer: ContractAnalyzer = Depends(get_analyzer)
):
    """
    Analyze contract text.
    
    Args:
        request: Contract text and optional filename
        analyzer: Contract analyzer instance
        
    Returns:
        Analysis summary
    """
    try:
        # Analyze document
        result = await analyzer.analyze_document(
            text=request.text,
            filename=request.filename
        )
        
        # Convert to response format
        return _create_response(result)
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/contracts/{contract_id}", response_model=DetailedAnalysisResponse)
async def get_contract_details(
    contract_id: str,
    analyzer: ContractAnalyzer = Depends(get_analyzer)
):
    """
    Get detailed contract analysis.
    
    Args:
        contract_id: Contract ID
        analyzer: Contract analyzer instance
        
    Returns:
        Detailed analysis
    """
    # In a real implementation, this would retrieve stored analysis from a database
    # This is a placeholder implementation
    raise HTTPException(status_code=404, detail="Contract not found")

def _create_response(result: AnalysisResult) -> Dict[str, Any]:
    """
    Create API response from analysis result.
    
    Args:
        result: Analysis result
        
    Returns:
        API response format
    """
    # Count risks by category
    risks_by_category = {}
    for risk in result.risks:
        category = risk.risk_category
        risks_by_category[category] = risks_by_category.get(category, 0) + 1
    
    # Create response
    response = {
        "contract_id": result.contract_id,
        "status": result.status,
        "processing_time_ms": result.processing_time_ms,
        "metadata": {
            "file_name": result.metadata.file_name,
            "file_type": result.metadata.file_type,
            "file_size_kb": result.metadata.file_size_kb,
            "page_count": result.metadata.page_count,
            "word_count": result.metadata.word_count,
            "contract_type": result.metadata.detected_contract_type,
            "type_confidence": result.metadata.contract_type_confidence
        },
        "summary": {
            "overall": result.summary.overall_summary,
            "key_points": [asdict(kp) for kp in result.summary.key_points[:3]],  # Top 3 key points
            "basics": result.summary.contract_basics
        },
        "risks_count": len(result.risks),
        "risks_by_category": risks_by_category,
        "sections_count": len(result.sections),
        "has_pii": result.has_pii,
        "errors": result.errors,
        "warnings": result.warnings
    }
    
    return response