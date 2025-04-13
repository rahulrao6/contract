"""
Contract classifier model for section and document classification.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
from datetime import datetime
import json

from .base import ContractMLModel, ModelResult

logger = logging.getLogger(__name__)

class ContractClassifier(ContractMLModel):
    """Model for classifying contract sections or document types."""
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        num_labels: int = 8,  # Default for common section types
        labels: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize contract classifier.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of classification labels
            labels: List of label names
            cache_dir: Directory for model caching
            device: Device to use (cuda or cpu)
        """
        super().__init__(
            model_name=model_name,
            task_type="classification",
            num_labels=num_labels,
            cache_dir=cache_dir,
            device=device
        )
        
        # Default classification labels if not provided
        default_labels = [
            "covenant", "representation", "condition", "termination", 
            "payment", "liability", "definition", "general"
        ]
        
        # Set label mappings
        self.labels = labels or default_labels[:num_labels]
        self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the classifier model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir
            )
            
            # Move model to specified device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Loaded classifier model: {self.model_name} with {self.num_labels} labels")
            
        except Exception as e:
            logger.error(f"Error loading classifier model: {str(e)}")
            raise
    
    def predict(self, texts: List[str], batch_size: int = 8) -> ModelResult:
        """
        Classify contract sections or documents.
        
        Args:
            texts: List of text sections to classify
            batch_size: Processing batch size
            
        Returns:
            ModelResult with classification outputs
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        predictions = []
        confidence_scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Skip empty or very short texts
            if not batch_texts or all(len(text.strip()) < 10 for text in batch_texts):
                continue
                
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process outputs
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_confs = torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
            
            predictions.extend(batch_preds)
            confidence_scores.extend(batch_confs)
        
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
    
    def predict_with_labels(self, texts: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """
        Classify and return with label names.
        
        Args:
            texts: List of text sections to classify
            batch_size: Processing batch size
            
        Returns:
            Dictionary with labeled predictions
        """
        results = self.predict(texts, batch_size)
        
        # Map numerical predictions to label names
        labeled_predictions = [self.idx_to_label.get(pred, f"unknown_{pred}") for pred in results.predictions]
        
        return {
            'predictions': labeled_predictions,
            'confidence_scores': results.confidence_scores,
            'processing_time_ms': results.processing_time_ms,
            'metadata': results.metadata
        }
    
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
        Train the classifier model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels (can be indices or label names)
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Output directory for trained model
            
        Returns:
            Training metrics
        """
        from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
        from torch.utils.data import Dataset
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert string labels to indices if needed
        if train_labels and isinstance(train_labels[0], str):
            train_labels = [self.label_to_idx.get(label, 0) for label in train_labels]
        
        if val_labels and isinstance(val_labels[0], str):
            val_labels = [self.label_to_idx.get(label, 0) for label in val_labels]
        
        # Split into validation set if not provided
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
        
        # Create tokenizer and model if not loaded
        if self.tokenizer is None or self.model is None:
            self.load_model()
        
        # Define dataset class
        class ClassificationDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                
                return item
        
        # Create datasets
        train_dataset = ClassificationDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ClassificationDataset(val_texts, val_labels, self.tokenizer)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            acc = accuracy_score(labels, predictions)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        train_result = trainer.train()
        
        # Evaluate model
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model(f"{output_dir}/final")
        
        # Update metadata
        self.is_trained = True
        self.training_metadata = {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_time_seconds": train_result.metrics.get("train_runtime", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "eval_metrics": eval_result,
            "labels": self.labels,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(self.training_metadata, f, indent=2)
        
        logger.info(f"Classifier trained successfully. Results: {eval_result}")
        return eval_result