"""
Base model class for all contract analysis models.
"""

import os
import json
import time
import torch
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    """Stores results from model predictions."""
    model_name: str
    predictions: List[Any]
    confidence_scores: List[float]
    processing_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContractMLModel:
    """Base class for all contract analysis models."""
    
    def __init__(
        self,
        model_name: str,
        task_type: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        num_labels: int = 2
    ):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the pretrained model
            task_type: Type of task (classification, extraction, etc.)
            cache_dir: Directory to cache model files
            device: Device to use (cuda or cpu)
            num_labels: Number of labels for classification tasks
        """
        self.model_name = model_name
        self.task_type = task_type
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        self.training_metadata = {}
        
        logger.info(f"Initializing {self.__class__.__name__} with model {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load the model and tokenizer.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def predict(self, texts: List[str], batch_size: int = 8) -> ModelResult:
        """
        Make predictions on input texts.
        This method should be implemented by subclasses.
        
        Args:
            texts: List of text inputs
            batch_size: Batch size for processing
            
        Returns:
            ModelResult with predictions and metadata
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[Any],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Any]] = None,
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        output_dir: str = "./model_output"
    ) -> Dict[str, Any]:
        """
        Train or fine-tune the model.
        This method should be implemented by subclasses that support training.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Output directory for trained model
            
        Returns:
            Training metrics
        """
        raise NotImplementedError("Training not implemented for this model")
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer is not loaded")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model files
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save metadata
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            metadata = {
                "model_name": self.model_name,
                "task_type": self.task_type,
                "num_labels": self.num_labels,
                "is_trained": self.is_trained,
                "training_metadata": self.training_metadata,
                "timestamp": datetime.now().isoformat()
            }
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load_from_dir(cls, model_dir: str, device: Optional[str] = None):
        """
        Load a model from a directory.
        
        Args:
            model_dir: Directory containing saved model
            device: Device to use (cuda or cpu)
            
        Returns:
            Loaded model instance
        """
        # Load metadata
        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=model_dir,  # Use directory as model name
            task_type=metadata.get("task_type", "classification"),
            num_labels=metadata.get("num_labels", 2),
            device=device
        )
        
        # Load model
        instance.load_model()
        
        # Set metadata
        instance.is_trained = metadata.get("is_trained", False)
        instance.training_metadata = metadata.get("training_metadata", {})
        
        logger.info(f"Loaded model from {model_dir}")
        return instance

    def _timer(self, func, *args, **kwargs):
        """Helper method to time function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = int((end_time - start_time) * 1000)  # ms
        return result, execution_time