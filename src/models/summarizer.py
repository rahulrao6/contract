#!/usr/bin/env python3
"""
Summarization model for contract analysis.
"""

import time
import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime

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
        Initialize the contract summarizer.
        
        Args:
            model_name: Pretrained model name.
            cache_dir: Directory for model caching.
            device: Device to use (e.g. "cuda" or "cpu"). Defaults to CPU if not provided.
        """
        super().__init__(
            model_name=model_name,
            task_type="summarization",
            cache_dir=cache_dir,
            device=device
        )
        self.load_model()
    
    def load_model(self):
        """Load the summarization model, tokenizer, and pipeline."""
        try:
            # Set the device id for the pipeline: 0 for cuda or -1 for cpu.
            device_id = 0 if self.device == "cuda" else -1
            
            # Initialize the summarization pipeline.
            # (Do NOT pass cache_dir to the pipeline as it causes unused-keyword errors.)
            self.summarization_pipeline = pipeline(
                "summarization", 
                model=self.model_name, 
                device=device_id
            )
            
            # Load tokenizer and model separately (pass cache_dir here so that caching works)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
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
        Generate summaries for a list of texts.
        
        Args:
            texts: List of texts to summarize.
            batch_size: Processing batch size.
            
        Returns:
            ModelResult containing a list of summary texts and confidence scores.
        """
        if self.summarization_pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        predictions = []
        confidence_scores = []
        
        # Use a maximum effective batch size of 4 for memory efficiency.
        effective_batch_size = min(batch_size, 4)
        
        # Limit input tokens to the model's maximum allowed tokens, or cap it (here 1024)
        max_input_tokens = min(self.tokenizer.model_max_length, 1024)
        
        # Process texts in batches
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i:i+effective_batch_size]
            for text in batch_texts:
                # Skip empty or very short texts
                if not text or len(text.strip()) < 50:
                    predictions.append("")
                    confidence_scores.append(0.0)
                    continue
                
                try:
                    # Encode and truncate text to a maximum of max_input_tokens
                    encoded = self.tokenizer.encode(
                        text, 
                        truncation=True, 
                        max_length=max_input_tokens
                    )
                    truncated_text = self.tokenizer.decode(
                        encoded, 
                        skip_special_tokens=True
                    )
                    
                    # Generate the summary using the pipeline
                    summary = self.summarization_pipeline(
                        truncated_text,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    summary_text = summary[0]['summary_text'] if summary and "summary_text" in summary[0] else ""
                    predictions.append(summary_text)
                    
                    # Since the summarization pipeline does not produce a true confidence score,
                    # we use a placeholder value.
                    confidence_scores.append(0.95)
                except Exception as e:
                    logger.error(f"Error summarizing text: {str(e)}")
                    predictions.append("")
                    confidence_scores.append(0.0)
        
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
        A convenient method to return a list of summaries for the provided texts.
        
        Args:
            texts: List of texts to summarize.
            max_length: Maximum length of each summary.
            min_length: Minimum length of each summary.
            batch_size: Batch size used during processing.
            
        Returns:
            A list of summary strings.
        """
        result = self.predict(texts, batch_size)
        return result.predictions

