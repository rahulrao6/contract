"""
Information extraction model for contract analysis.
"""

import time
import torch
import logging
import re
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from .base import ContractMLModel, ModelResult

logger = logging.getLogger(__name__)

class ContractExtractor(ContractMLModel):
    """Model for extracting specific information from contracts."""
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        extraction_types: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize contract extraction model.
        
        Args:
            model_name: Pretrained model name
            extraction_types: List of information types to extract
            cache_dir: Directory for model caching
            device: Device to use (cuda or cpu)
        """
        super().__init__(
            model_name=model_name,
            task_type="question_answering",
            cache_dir=cache_dir,
            device=device
        )
        
        # Default extraction types if not provided
        if extraction_types is None:
            extraction_types = [
                "parties", "effective_date", "termination_date", 
                "payment_terms", "renewal_terms", "governing_law", 
                "venue", "notice_period"
            ]
        
        self.extraction_types = extraction_types
        
        # Initialize QA prompts for each extraction type
        self.extraction_prompts = {
            "parties": "Who are the parties to this agreement?",
            "effective_date": "What is the effective date of this agreement?",
            "termination_date": "What is the termination date of this agreement?",
            "payment_terms": "What are the payment terms in this agreement?",
            "renewal_terms": "What are the renewal terms in this agreement?",
            "governing_law": "Which law governs this agreement?",
            "venue": "What is the venue for disputes under this agreement?",
            "notice_period": "What is the notice period for termination under this agreement?"
        }
        
        # Additional extraction patterns using regex
        self._initialize_extraction_patterns()
        
        # Load model
        self.load_model()
    
    def _initialize_extraction_patterns(self):
        """Initialize regex patterns for information extraction."""
        self.extraction_patterns = {
            "parties": [
                r'(?:agreement|contract)\s+(?:is\s+)?(?:made\s+and\s+entered\s+into|between|by\s+and\s+between)\s+([^,]+?)\s+(?:and)\s+([^,\.]+)',
                r'(?:THIS AGREEMENT|THIS CONTRACT).{1,50}?(?:by|between)\s+([^,]+?)\s+(?:and)\s+([^,\.]+)'
            ],
            "effective_date": [
                r'(?:effective|commencement)\s+date\s+(?:is|shall\s+be|of)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:effective|shall\s+commence)\s+(?:as\s+of|on)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            "termination_date": [
                r'(?:terminat|expir|end)(?:ion|e|es|ing)\s+date\s+(?:is|shall\s+be|of)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:shall|will)\s+(?:terminat|expir|end|conclude)(?:e|ion)?\s+(?:on|upon)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            "governing_law": [
                r'(?:governed|construed|interpreted).{1,20}?(?:by|in\s+accordance\s+with).{1,20}?(?:laws|law)\s+of\s+(?:the\s+)?(?:State\s+of|Commonwealth\s+of|Province\s+of)?\s+([A-Za-z\s]+)',
                r'governing\s+law.{1,20}?(?:shall\s+be|is)\s+(?:that\s+of\s+)?(?:the\s+)?(?:State\s+of|Commonwealth\s+of|Province\s+of)?\s+([A-Za-z\s]+)'
            ]
        }
    
    def load_model(self):
        """Load the extraction model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to specified device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Loaded extraction model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading extraction model: {str(e)}")
            raise
    
    def predict(self, texts: List[str], questions: List[str], batch_size: int = 8) -> ModelResult:
        """
        Extract answers from text based on questions.
        
        Args:
            texts: List of contexts to extract from
            questions: List of questions to ask
            batch_size: Processing batch size
            
        Returns:
            ModelResult with extraction results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure texts and questions have the same length
        if len(texts) != len(questions):
            raise ValueError("Number of texts must match number of questions")
        
        start_time = time.time()
        predictions = []
        confidence_scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            
            # Process each text-question pair
            for text, question in zip(batch_texts, batch_questions):
                # Skip empty or very short texts
                if not text or len(text.strip()) < 20:
                    predictions.append("")
                    confidence_scores.append(0.0)
                    continue
                
                try:
                    # Prepare input
                    inputs = self.tokenizer(
                        question,
                        text,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get answer
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    # Get most likely answer span
                    start_idx = torch.argmax(start_logits).item()
                    end_idx = torch.argmax(end_logits).item()
                    
                    # Ensure valid span
                    if end_idx < start_idx:
                        end_idx = start_idx
                    
                    # Calculate confidence
                    start_probs = torch.nn.functional.softmax(start_logits, dim=1)
                    end_probs = torch.nn.functional.softmax(end_logits, dim=1)
                    confidence = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
                    
                    # Extract answer text
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                    answer_tokens = tokens[start_idx:end_idx+1]
                    answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
                    
                    # Clean answer
                    answer = answer.strip()
                    
                    predictions.append(answer)
                    confidence_scores.append(confidence)
                    
                except Exception as e:
                    logger.error(f"Error extracting answer: {str(e)}")
                    predictions.append("")
                    confidence_scores.append(0.0)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ModelResult(
            model_name=self.model_name,
            predictions=predictions,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time_ms,
            metadata={
                'num_samples': len(texts),
                'batch_size': batch_size,
                'task': self.task_type
            }
        )
    
    def extract_information(self, text: str, extraction_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Extract specific information from contract text.
        Uses both ML model and regex patterns for best results.
        
        Args:
            text: Contract text to analyze
            extraction_types: Types of information to extract (defaults to all)
            
        Returns:
            Dictionary of extracted information with metadata
        """
        # Use specified extraction types or default to all
        if extraction_types is None:
            extraction_types = self.extraction_types
        
        results = {}
        
        # First try rule-based extraction using patterns
        pattern_results = self._extract_with_patterns(text, extraction_types)
        
        # Then try ML-based extraction for missing or low-confidence results
        ml_extraction_types = [
            ext_type for ext_type in extraction_types
            if ext_type not in pattern_results or pattern_results[ext_type]["confidence"] < 0.8
        ]
        
        if ml_extraction_types and len(text) >= 20:  # Only use ML if we have text and missing extractions
            ml_results = self._extract_with_ml(text, ml_extraction_types)
            
            # Merge results, preferring high-confidence results
            for ext_type in extraction_types:
                if ext_type in ml_results and (
                    ext_type not in pattern_results or 
                    ml_results[ext_type]["confidence"] > pattern_results[ext_type]["confidence"]
                ):
                    results[ext_type] = ml_results[ext_type]
                elif ext_type in pattern_results:
                    results[ext_type] = pattern_results[ext_type]
                else:
                    # No result from either method
                    results[ext_type] = {
                        "value": "",
                        "confidence": 0.0,
                        "method": "none"
                    }
        else:
            # Just use pattern results
            results = pattern_results
            
            # Add empty entries for missing types
            for ext_type in extraction_types:
                if ext_type not in results:
                    results[ext_type] = {
                        "value": "",
                        "confidence": 0.0,
                        "method": "none"
                    }
        
        return results
    
    def _extract_with_patterns(self, text: str, extraction_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract information using regex patterns.
        
        Args:
            text: Contract text to analyze
            extraction_types: Types of information to extract
            
        Returns:
            Dictionary of extracted information with metadata
        """
        results = {}
        
        # Process each extraction type that has patterns
        for ext_type in extraction_types:
            if ext_type not in self.extraction_patterns:
                continue
                
            # Try each pattern for this type
            for pattern in self.extraction_patterns[ext_type]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Extract the first match
                    if isinstance(matches[0], tuple):
                        # If pattern has groups, join them
                        value = " and ".join(m for m in matches[0] if m.strip())
                    else:
                        value = matches[0]
                    
                    # Store result
                    results[ext_type] = {
                        "value": value.strip(),
                        "confidence": 0.9,  # High confidence for pattern matches
                        "method": "pattern"
                    }
                    break
        
        return results
    
    def _extract_with_ml(self, text: str, extraction_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract information using ML model.
        
        Args:
            text: Contract text to analyze
            extraction_types: Types of information to extract
            
        Returns:
            Dictionary of extracted information with metadata
        """
        results = {}
        
        # Prepare questions and contexts
        questions = [self.extraction_prompts[ext_type] for ext_type in extraction_types]
        contexts = [text] * len(questions)
        
        # Get predictions
        ml_results = self.predict(contexts, questions)
        
        # Process results
        for i, ext_type in enumerate(extraction_types):
            if i < len(ml_results.predictions):
                value = ml_results.predictions[i]
                confidence = ml_results.confidence_scores[i]
                
                # If we got an answer with reasonable confidence
                if value and confidence >= 0.5:
                    results[ext_type] = {
                        "value": value,
                        "confidence": confidence,
                        "method": "ml"
                    }
        
        return results