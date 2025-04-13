"""
Risk detection model for contract analysis.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import time
from datetime import datetime
import json

from .base import ContractMLModel, ModelResult

logger = logging.getLogger(__name__)

class RiskDataset(Dataset):
    """Dataset for risk detection training and evaluation."""
    
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

class RiskDetectionModel(ContractMLModel):
    """Specialized model for identifying risks in contracts."""
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        risk_categories: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize risk detection model.
        
        Args:
            model_name: Pretrained model name
            risk_categories: List of risk categories to detect
            cache_dir: Directory for model caching
            device: Device to use (cuda or cpu)
        """
        # Default risk categories if not provided
        if risk_categories is None:
            risk_categories = [
                "auto_renewal", "hidden_fee", "termination_without_notice",
                "unilateral_modification", "unlimited_liability", "waiver_of_rights",
                "jurisdiction_restriction", "data_usage", "arbitration_clause",
                "class_action_waiver", "non_assignment", "minimum_commitment",
                "liquidated_damages", "non_compete", "evergreen_renewal",
                "unilateral_price_change", "early_termination_fee"
            ]
        
        self.risk_categories = risk_categories
        num_labels = len(risk_categories)
        
        super().__init__(
            model_name=model_name,
            task_type="classification",
            num_labels=num_labels,
            cache_dir=cache_dir,
            device=device
        )
        
        # Map risk categories to indices
        self.risk_to_idx = {risk: i for i, risk in enumerate(risk_categories)}
        self.idx_to_risk = {i: risk for i, risk in enumerate(risk_categories)}
        
        # Category and severity mappings
        self._initialize_risk_mappings()
    
    def _initialize_risk_mappings(self):
        """Initialize risk category and severity mappings."""
        self.risk_category_map = {
            "auto_renewal": "renewal_risk",
            "hidden_fee": "payment_risk",
            "termination_without_notice": "termination_risk",
            "unilateral_modification": "modification_risk",
            "unlimited_liability": "liability_risk",
            "waiver_of_rights": "limitation_risk",
            "jurisdiction_restriction": "jurisdiction_risk",
            "data_usage": "privacy_risk",
            "arbitration_clause": "jurisdiction_risk",
            "class_action_waiver": "limitation_risk",
            "non_assignment": "limitation_risk", 
            "minimum_commitment": "payment_risk",
            "liquidated_damages": "liability_risk",
            "non_compete": "limitation_risk",
            "evergreen_renewal": "renewal_risk",
            "unilateral_price_change": "payment_risk",
            "early_termination_fee": "termination_risk"
        }
        
        self.risk_severity_map = {
            "auto_renewal": "medium",
            "hidden_fee": "high",
            "termination_without_notice": "high",
            "unilateral_modification": "medium",
            "unlimited_liability": "high",
            "waiver_of_rights": "medium",
            "jurisdiction_restriction": "medium",
            "data_usage": "medium",
            "arbitration_clause": "medium",
            "class_action_waiver": "high",
            "non_assignment": "low",
            "minimum_commitment": "medium",
            "liquidated_damages": "medium",
            "non_compete": "medium",
            "evergreen_renewal": "high",
            "unilateral_price_change": "high",
            "early_termination_fee": "medium"
        }
    
    def load_model(self):
        """Load the risk detection model and tokenizer."""
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
            
            logger.info(f"Loaded risk detection model: {self.model_name} with {self.num_labels} labels")
            
        except Exception as e:
            logger.error(f"Error loading risk detection model: {str(e)}")
            raise
    
    def predict(self, texts: List[str], batch_size: int = 8) -> ModelResult:
        """
        Make risk predictions on contract texts.
        
        Args:
            texts: List of text sections to analyze
            batch_size: Processing batch size
            
        Returns:
            ModelResult with prediction outputs
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
    
    def predict_risks(self, texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify risks in contract texts with confidence filtering.
        
        Args:
            texts: List of text sections to analyze
            threshold: Minimum confidence threshold for reporting risks
            
        Returns:
            List of detected risks with metadata
        """
        results = self.predict(texts)
        
        risk_findings = []
        
        for i, (pred, conf) in enumerate(zip(results.predictions, results.confidence_scores)):
            if conf >= threshold:
                risk_name = self.idx_to_risk.get(pred, f"unknown_{pred}")
                
                risk_finding = {
                    'text_idx': i,
                    'risk_name': risk_name,
                    'confidence': float(conf),
                    'risk_category': self.get_risk_category(risk_name),
                    'risk_level': self.get_risk_severity(risk_name)
                }
                
                risk_findings.append(risk_finding)
        
        return risk_findings
    
    def get_risk_category(self, risk_name: str) -> str:
        """Get the category for a risk name."""
        return self.risk_category_map.get(risk_name, "other_risk")
    
    def get_risk_severity(self, risk_name: str) -> str:
        """Get the severity level for a risk name."""
        return self.risk_severity_map.get(risk_name, "medium")
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        output_dir: str = "./model_output"
    ) -> Dict[str, Any]:
        """
        Train the risk detection model.
        
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
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into validation set if not provided
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42
            )
        
        # Create tokenizer and model if not loaded
        if self.tokenizer is None or self.model is None:
            self.load_model()
        
        # Create datasets
        train_dataset = RiskDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = RiskDataset(val_texts, val_labels, self.tokenizer)
        
        # Data collator for dynamic padding in batches
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
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(self.training_metadata, f, indent=2)
        
        logger.info(f"Model trained successfully. Results: {eval_result}")
        return eval_result