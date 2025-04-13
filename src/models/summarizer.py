"""
Summarization model for contract analysis.
"""

import time
import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
from datetime import datetime
import json

from .base import ContractMLModel, ModelResult

logger = logging.getLogger(__name__)

class ContractSummarizer(ContractMLModel):
    """Model for generating summaries of contracts and sections."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize contract summarizer.
        
        Args:
            model_name: Pretrained model name
            cache_dir: Directory for model caching
            device: Device to use (cuda or cpu)
        """
        super().__init__(
            model_name=model_name,
            task_type="summarization",
            cache_dir=cache_dir,
            device=device
        )
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the summarizer model and tokenizer."""
        try:
            # Try to use the pipeline API for efficiency
            device_id = 0 if self.device == "cuda" else -1
            self.summarization_pipeline = pipeline(
                "summarization", 
                model=self.model_name, 
                device=device_id,
                cache_dir=self.cache_dir
            )
            
            # Also get the tokenizer for direct access
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            # Get the model for direct access if needed
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info(f"Loaded summarization model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise
    
    def predict(self, texts: List[str], batch_size: int = 4) -> ModelResult:
        """
        Generate summaries for contract texts.
        
        Args:
            texts: List of texts to summarize
            batch_size: Processing batch size
            
        Returns:
            ModelResult with summaries
        """
        if self.summarization_pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        predictions = []
        confidence_scores = []
        
        # Use smaller batch size for summarization due to memory constraints
        effective_batch_size = min(batch_size, 4)
        
        # Process in batches
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i:i+effective_batch_size]
            
            # Process each text
            for text in batch_texts:
                # Skip empty or very short texts
                if not text or len(text.strip()) < 50:
                    predictions.append("")
                    confidence_scores.append(0.0)
                    continue
                
                try:
                    # Truncate text to model's maximum input length
                    max_tokens = self.tokenizer.model_max_length
                    truncated_text = self.tokenizer.decode(
                        self.tokenizer.encode(text, truncation=True, max_length=max_tokens),
                        skip_special_tokens=True
                    )
                    
                    # Generate summary
                    summary = self.summarization_pipeline(
                        truncated_text,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    
                    # Extract summary text
                    summary_text = summary[0]['summary_text'] if summary else ""
                    predictions.append(summary_text)
                    
                    # Use a placeholder confidence since summarization doesn't naturally provide one
                    confidence_scores.append(0.95)
                    
                except Exception as e:
                    logger.error(f"Error summarizing text: {str(e)}")
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
                'batch_size': effective_batch_size,
                'task': self.task_type
            }
        )
    
    def summarize(
        self, 
        texts: List[str], 
        max_length: int = 150, 
        min_length: int = 30,
        batch_size: int = 4
    ) -> List[str]:
        """
        Convenient method to get summaries directly.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            batch_size: Processing batch size
            
        Returns:
            List of summary texts
        """
        # Use existing predict method and extract the predictions
        result = self.predict(texts, batch_size)
        return result.predictions