#!/usr/bin/env python3
"""
Train models for the Contract Analyzer using CUAD dataset.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.risk_detector import RiskDetectionModel
from src.models.classifier import ContractClassifier
from src.models.extractor import ContractExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_cuad_data(csv_path: str) -> pd.DataFrame:
    """
    Load CUAD dataset from CSV file.
    
    Args:
        csv_path: Path to CUAD CSV file
        
    Returns:
        Dataframe with CUAD data
    """
    logger.info(f"Loading CUAD data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CUAD dataset")
        return df
    except Exception as e:
        logger.error(f"Error loading CUAD data: {str(e)}")
        raise

def prepare_risk_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare training data for risk detection model.
    
    Args:
        df: DataFrame with CUAD data
        
    Returns:
        Dictionary with prepared training data
    """
    logger.info("Preparing risk detection training data")
    
    # Map CUAD provisions to risk categories
    provision_risk_map = {
        "Termination For Convenience": "termination_without_notice",
        "Unlimited/All-encompassing Indemnification": "unlimited_liability",
        "Non-Compete": "non_compete",
        "Liquidated Damages": "liquidated_damages",
        "Auto-Renewal": "auto_renewal",
        "Most Favored Nation": "unilateral_price_change",
        "Anti-Assignment": "non_assignment",
        "Minimum Commitment": "minimum_commitment"
    }
    
    texts = []
    labels = []
    
    # Process each provision
    for provision, risk in provision_risk_map.items():
        # Get positive examples with text content
        answer_col = f"{provision}-Answer"
        
        if answer_col in df.columns:
            positive_examples = df[df[answer_col].notna()][answer_col].tolist()
            
            for example in positive_examples:
                if isinstance(example, str) and len(example.strip()) > 20:
                    texts.append(example)
                    labels.append(risk)
    
    logger.info(f"Prepared {len(texts)} training examples for {len(set(labels))} risk categories")
    
    return {
        "texts": texts,
        "labels": labels,
        "risk_categories": list(set(labels))
    }

def prepare_classification_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare training data for section classification model.
    
    Args:
        df: DataFrame with CUAD data
        
    Returns:
        Dictionary with prepared training data
    """
    logger.info("Preparing classification training data")
    
    # Map CUAD provisions to section types
    provision_section_map = {
        "Governing Law": "jurisdiction",
        "Termination For Convenience": "termination",
        "Notice Period To Terminate Renewal": "termination",
        "Renewal Term": "renewal",
        "Payment Terms": "payment",
        "License Grant": "intellectual_property",
        "Non-Compete": "covenant",
        "Non-Disparagement": "covenant",
        "Unlimited/All-encompassing Indemnification": "liability",
        "Indemnification": "liability",
        "Confidentiality": "confidentiality"
    }
    
    texts = []
    labels = []
    
    # Process each provision
    for provision, section_type in provision_section_map.items():
        # Get examples with text content
        answer_col = f"{provision}-Answer"
        
        if answer_col in df.columns:
            examples = df[df[answer_col].notna()][answer_col].tolist()
            
            for example in examples:
                if isinstance(example, str) and len(example.strip()) > 20:
                    texts.append(example)
                    labels.append(section_type)
    
    logger.info(f"Prepared {len(texts)} training examples for {len(set(labels))} section types")
    
    return {
        "texts": texts,
        "labels": labels,
        "section_types": list(set(labels))
    }

def prepare_extraction_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare training data for information extraction.
    
    Args:
        df: DataFrame with CUAD data
        
    Returns:
        Dictionary with prepared training data
    """
    logger.info("Preparing extraction training data")
    
    # Map CUAD provisions to extraction types
    extraction_map = {
        "Agreement Date": "effective_date",
        "Effective Date": "effective_date",
        "Expiration Date": "termination_date",
        "Governing Law": "governing_law",
        "Notice Period To Terminate Renewal": "notice_period",
        "Parties": "parties"
    }
    
    contexts = []
    questions = []
    answers = []
    
    # Process each provision
    for provision, extraction_type in extraction_map.items():
        # Get document text
        doc_col = f"Document Name"
        answer_col = f"{provision}-Answer"
        
        if doc_col in df.columns and answer_col in df.columns:
            # Filter rows with answers
            valid_rows = df[(df[doc_col].notna()) & (df[answer_col].notna())]
            
            for _, row in valid_rows.iterrows():
                doc_text = row[doc_col]
                answer_text = row[answer_col]
                
                if isinstance(doc_text, str) and isinstance(answer_text, str):
                    if len(doc_text.strip()) > 50 and len(answer_text.strip()) > 3:
                        # Create question based on extraction type
                        if extraction_type == "effective_date":
                            question = "What is the effective date of this agreement?"
                        elif extraction_type == "termination_date":
                            question = "What is the termination date of this agreement?"
                        elif extraction_type == "governing_law":
                            question = "Which law governs this agreement?"
                        elif extraction_type == "notice_period":
                            question = "What is the notice period for termination?"
                        elif extraction_type == "parties":
                            question = "Who are the parties to this agreement?"
                        else:
                            question = f"What is the {extraction_type.replace('_', ' ')}?"
                        
                        contexts.append(doc_text)
                        questions.append(question)
                        answers.append(answer_text)
    
    logger.info(f"Prepared {len(contexts)} training examples for information extraction")
    
    return {
        "contexts": contexts,
        "questions": questions,
        "answers": answers
    }

def train_risk_model(config: Config, data: Dict[str, Any], output_dir: str):
    """
    Train the risk detection model.
    
    Args:
        config: Configuration settings
        data: Prepared training data
        output_dir: Output directory for trained model
    """
    logger.info("Training risk detection model")
    
    # Initialize risk detector
    risk_detector = RiskDetectionModel(
        model_name=config.LEGAL_BERT_MODEL,
        risk_categories=data.get("risk_categories"),
        cache_dir=config.CACHE_DIR,
        device=config.DEVICE
    )
    
    # Train model
    result = risk_detector.train(
        train_texts=data["texts"],
        train_labels=[data["risk_categories"].index(label) for label in data["labels"]],
        batch_size=config.BATCH_SIZE,
        epochs=3,
        output_dir=output_dir
    )
    
    logger.info(f"Risk detector training complete. Results: {result}")

def train_classifier(config: Config, data: Dict[str, Any], output_dir: str):
    """
    Train the section classifier model.
    
    Args:
        config: Configuration settings
        data: Prepared training data
        output_dir: Output directory for trained model
    """
    logger.info("Training section classifier model")
    
    # Initialize classifier
    classifier = ContractClassifier(
        model_name=config.LEGAL_BERT_MODEL,
        num_labels=len(data.get("section_types", [])),
        labels=data.get("section_types", []),
        cache_dir=config.CACHE_DIR,
        device=config.DEVICE
    )
    
    # Train model
    result = classifier.train(
        train_texts=data["texts"],
        train_labels=data["labels"],
        batch_size=config.BATCH_SIZE,
        epochs=3,
        output_dir=output_dir
    )
    
    logger.info(f"Classifier training complete. Results: {result}")

def train_extractor(config: Config, data: Dict[str, Any], output_dir: str):
    """
    Train the information extraction model.
    
    Args:
        config: Configuration settings
        data: Prepared training data
        output_dir: Output directory for trained model
    """
    logger.info("Training information extraction model")
    
    # Initialize extractor
    extractor = ContractExtractor(
        model_name=config.LEGAL_BERT_MODEL,
        cache_dir=config.CACHE_DIR,
        device=config.DEVICE
    )
    
    # For extraction models, the training process would be more complex
    # This would involve setting up a QA training pipeline
    logger.info("Note: Extraction model training requires custom QA pipeline setup")
    
    # Save extraction data for future use
    with open(os.path.join(output_dir, "extraction_data.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(data["contexts"]),
            "extraction_types": list(set([q.split()[3] for q in data["questions"]]))
        }, f, indent=2)
    
    logger.info(f"Extraction training data saved to {output_dir}")

def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description="Train models for Contract Analyzer")
    parser.add_argument("--model", choices=["risk", "classifier", "extractor", "all"], default="all",
                       help="Model to train (default: all)")
    parser.add_argument("--cuad-data", default="./data/cuad/master_clauses.csv",
                       help="Path to CUAD dataset CSV")
    parser.add_argument("--output-dir", default="./models",
                       help="Output directory for trained models")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    config.BATCH_SIZE = args.batch_size
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    risk_dir = os.path.join(args.output_dir, "risk_detector")
    classifier_dir = os.path.join(args.output_dir, "section_classifier")
    extractor_dir = os.path.join(args.output_dir, "information_extractor")
    
    for directory in [risk_dir, classifier_dir, extractor_dir]:
        os.makedirs(directory, exist_ok=True)
    
    try:
        # Load CUAD data
        df = load_cuad_data(args.cuad_data)
        
        # Train risk model
        if args.model in ["risk", "all"]:
            risk_data = prepare_risk_training_data(df)
            if risk_data["texts"]:
                train_risk_model(config, risk_data, risk_dir)
            else:
                logger.warning("No risk detection training data found")
        
        # Train classifier model
        if args.model in ["classifier", "all"]:
            classification_data = prepare_classification_data(df)
            if classification_data["texts"]:
                train_classifier(config, classification_data, classifier_dir)
            else:
                logger.warning("No classification training data found")
        
        # Train extractor model
        if args.model in ["extractor", "all"]:
            extraction_data = prepare_extraction_data(df)
            if extraction_data["contexts"]:
                train_extractor(config, extraction_data, extractor_dir)
            else:
                logger.warning("No extraction training data found")
        
        logger.info("Training complete")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()