#!/usr/bin/env python3
"""
Process CUAD dataset for Contract Analyzer training.
"""

import os
import sys
import pandas as pd
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_cuad_dataset(input_csv: str, output_dir: str) -> None:
    """
    Process CUAD dataset and create training files.
    
    Args:
        input_csv: Path to CUAD CSV file
        output_dir: Directory to save processed data
    """
    logger.info(f"Processing CUAD dataset from {input_csv}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # Check columns
    logger.info(f"CSV has {len(df)} rows and {len(df.columns)} columns")
    
    # Create risk training data
    risk_provisions = [
        "Auto-Renewal", "Termination For Convenience", "Liquidated Damages",
        "Non-Compete", "Most Favored Nation", "Anti-Assignment",
        "Change Of Control", "Uncapped Liability"
    ]
    
    risk_data = create_risk_training_data(df, risk_provisions)
    pd.DataFrame(risk_data).to_csv(os.path.join(output_dir, "risk_training.csv"), index=False)
    logger.info(f"Created risk training data with {len(risk_data['texts'])} examples")
    
    # Create section classification data
    section_provisions = [
        "Governing Law", "Termination For Convenience", "Renewal Term",
        "Payment Terms", "Confidentiality", "Indemnification",
        "Warranties", "Insurance", "Liability"
    ]
    
    section_data = create_section_training_data(df, section_provisions)
    pd.DataFrame(section_data).to_csv(os.path.join(output_dir, "section_training.csv"), index=False)
    logger.info(f"Created section training data with {len(section_data['texts'])} examples")
    
    # Create extraction training data
    extraction_provisions = [
        "Effective Date", "Expiration Date", "Renewal Term",
        "Governing Law", "Parties", "Agreement Date",
        "Notice Period To Terminate Renewal"
    ]
    
    extraction_data = create_extraction_training_data(df, extraction_provisions)
    pd.DataFrame(extraction_data).to_csv(os.path.join(output_dir, "extraction_training.csv"), index=False)
    logger.info(f"Created extraction training data with {len(extraction_data['contexts'])} examples")
    
    logger.info("CUAD dataset processing complete")

def create_risk_training_data(df: pd.DataFrame, provision_types: List[str]) -> Dict[str, List[Any]]:
    """
    Create training data for risk detection.
    
    Args:
        df: CUAD DataFrame
        provision_types: List of provision types to include
        
    Returns:
        Dictionary with training data
    """
    texts = []
    labels = []
    
    # Create label mapping
    label_map = {
        "Auto-Renewal": "auto_renewal",
        "Termination For Convenience": "termination_without_notice",
        "Liquidated Damages": "liquidated_damages",
        "Non-Compete": "non_compete",
        "Most Favored Nation": "unilateral_price_change",
        "Anti-Assignment": "non_assignment",
        "Change Of Control": "assignment_restriction",
        "Uncapped Liability": "unlimited_liability"
    }
    
    # Process each provision
    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        if answer_col in df.columns:
            # Get positive examples
            positive = df[df[answer_col].notna()]
            for _, row in positive.iterrows():
                text = row[answer_col]
                if isinstance(text, str) and len(text.strip()) > 20:
                    texts.append(text.strip())
                    labels.append(label_map.get(provision, provision.lower().replace(' ', '_')))
            
            # Get some negative examples
            negative_provisions = [p for p in provision_types if p != provision]
            for neg_provision in negative_provisions[:2]:  # Limit to 2 negative sources per provision
                neg_col = f"{neg_provision}-Answer"
                if neg_col in df.columns:
                    negative = df[df[neg_col].notna()].sample(min(10, df[df[neg_col].notna()].shape[0]))
                    for _, row in negative.iterrows():
                        text = row[neg_col]
                        if isinstance(text, str) and len(text.strip()) > 20:
                            texts.append(text.strip())
                            labels.append("no_risk")  # Negative class
    
    return {
        "texts": texts,
        "labels": labels,
        "provision_types": provision_types
    }

def create_section_training_data(df: pd.DataFrame, provision_types: List[str]) -> Dict[str, List[Any]]:
    """
    Create training data for section classification.
    
    Args:
        df: CUAD DataFrame
        provision_types: List of provision types to include
        
    Returns:
        Dictionary with training data
    """
    texts = []
    labels = []
    
    # Create label mapping
    label_map = {
        "Governing Law": "jurisdiction",
        "Termination For Convenience": "termination",
        "Renewal Term": "renewal",
        "Payment Terms": "payment",
        "Confidentiality": "confidentiality",
        "Indemnification": "indemnification",
        "Warranties": "warranty",
        "Insurance": "insurance",
        "Liability": "liability"
    }
    
    # Process each provision
    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        if answer_col in df.columns:
            # Get examples
            examples = df[df[answer_col].notna()]
            for _, row in examples.iterrows():
                text = row[answer_col]
                if isinstance(text, str) and len(text.strip()) > 20:
                    texts.append(text.strip())
                    labels.append(label_map.get(provision, provision.lower().replace(' ', '_')))
    
    return {
        "texts": texts,
        "labels": labels,
        "section_types": list(set(labels))
    }

def create_extraction_training_data(df: pd.DataFrame, provision_types: List[str]) -> Dict[str, List[Any]]:
    """
    Create training data for information extraction.
    
    Args:
        df: CUAD DataFrame
        provision_types: List of provision types to include
        
    Returns:
        Dictionary with training data
    """
    contexts = []
    questions = []
    answers = []
    
    # Create question mapping
    question_map = {
        "Effective Date": "What is the effective date of this agreement?",
        "Expiration Date": "What is the expiration date of this agreement?",
        "Renewal Term": "What are the renewal terms of this agreement?",
        "Governing Law": "Which law governs this agreement?",
        "Parties": "Who are the parties to this agreement?",
        "Agreement Date": "When was this agreement made or signed?",
        "Notice Period To Terminate Renewal": "What is the notice period to terminate renewal?"
    }
    
    # Process each provision
    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        doc_col = "Document Name"
        
        if answer_col in df.columns and doc_col in df.columns:
            # Get examples with both document and answer
            examples = df[(df[answer_col].notna()) & (df[doc_col].notna())]
            for _, row in examples.iterrows():
                doc_text = row[doc_col]
                answer_text = row[answer_col]
                
                if isinstance(doc_text, str) and isinstance(answer_text, str):
                    if len(doc_text.strip()) > 50 and len(answer_text.strip()) > 3:
                        contexts.append(doc_text.strip())
                        questions.append(question_map.get(
                            provision, 
                            f"What is the {provision.lower().replace(' ', ' ')}?"
                        ))
                        answers.append(answer_text.strip())
    
    return {
        "contexts": contexts,
        "questions": questions,
        "answers": answers,
        "provision_types": provision_types
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <output_dir>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    process_cuad_dataset(input_csv, output_dir)