#!/usr/bin/env python3
"""
Process CUAD dataset for Contract Analyzer training.
"""

import os
import sys
import pandas as pd
import logging
from typing import Dict, List, Any

# -------------------------------------------------------------------
# Ensure the project root is in sys.path so that "src" modules are available.
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------------------------------------------------

# Import the Config from your src directory.
try:
    from src.config import Config
except ModuleNotFoundError:
    print("Error: Could not import 'src.config'. Make sure to run from project root or set PYTHONPATH appropriately.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Create a default configuration instance.
# If your Config requires an 'app' argument, then we pass a dummy value ("cli").
if "app" in Config.__init__.__code__.co_varnames:
    config_instance = Config(app="cli")
else:
    config_instance = Config()

# Set a default MIN_RISK_TEXT_LENGTH if not defined.
if not hasattr(config_instance, "MIN_RISK_TEXT_LENGTH"):
    config_instance.MIN_RISK_TEXT_LENGTH = 20

def process_cuad_dataset(input_csv: str, output_dir: str) -> None:
    """
    Process CUAD dataset and create training files.

    Args:
        input_csv: Path to CUAD CSV file.
        output_dir: Directory to save processed data.
    """
    logger.info(f"Processing CUAD dataset from {input_csv}")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
    # Normalize column names by stripping extra spaces
    df.columns = [col.strip() for col in df.columns]

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
        df: CUAD DataFrame.
        provision_types: List of provision types to include.

    Returns:
        Dictionary with keys 'texts' and 'labels'.
    """
    texts = []
    labels = []
    min_length = config_instance.MIN_RISK_TEXT_LENGTH

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

    # Allow very short answers if they are exactly "Yes" or "No"
    def valid_text(text_val: str) -> bool:
        t = text_val.strip()
        return (len(t) >= min_length) or (t.lower() in ["yes", "no"])

    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        if answer_col in df.columns:
            # Positive examples
            positive = df[df[answer_col].notna()]
            for _, row in positive.iterrows():
                text_val = row[answer_col]
                if isinstance(text_val, str) and valid_text(text_val):
                    texts.append(text_val.strip())
                    labels.append(label_map.get(provision, provision.lower().replace(" ", "_")))
            # Negative examples from other provisions
            negative_provisions = [p for p in provision_types if p != provision]
            for neg_provision in negative_provisions[:2]:
                neg_col = f"{neg_provision}-Answer"
                if neg_col in df.columns:
                    # Safely sample up to 10 available examples
                    sample_n = min(10, df[df[neg_col].notna()].shape[0])
                    negative = df[df[neg_col].notna()].sample(n=sample_n, random_state=42)
                    for _, row in negative.iterrows():
                        text_val = row[neg_col]
                        if isinstance(text_val, str) and valid_text(text_val):
                            texts.append(text_val.strip())
                            labels.append("no_risk")
    return {"texts": texts, "labels": labels}

def create_section_training_data(df: pd.DataFrame, provision_types: List[str]) -> Dict[str, List[Any]]:
    """
    Create training data for section classification.

    Args:
        df: CUAD DataFrame.
        provision_types: List of provision types to include.

    Returns:
        Dictionary with keys 'texts' and 'labels'.
    """
    texts = []
    labels = []
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
    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        if answer_col in df.columns:
            examples = df[df[answer_col].notna()]
            for _, row in examples.iterrows():
                text_val = row[answer_col]
                if isinstance(text_val, str) and len(text_val.strip()) > 20:
                    texts.append(text_val.strip())
                    labels.append(label_map.get(provision, provision.lower().replace(" ", "_")))
    return {"texts": texts, "labels": labels}

def create_extraction_training_data(df: pd.DataFrame, provision_types: List[str]) -> Dict[str, List[Any]]:
    """
    Create training data for information extraction.

    Args:
        df: CUAD DataFrame.
        provision_types: List of provision types to include.

    Returns:
        Dictionary with keys 'contexts', 'questions', and 'answers'.
    """
    contexts = []
    questions = []
    answers = []
    question_map = {
        "Effective Date": "What is the effective date of this agreement?",
        "Expiration Date": "What is the expiration date of this agreement?",
        "Renewal Term": "What are the renewal terms of this agreement?",
        "Governing Law": "Which law governs this agreement?",
        "Parties": "Who are the parties to this agreement?",
        "Agreement Date": "When was this agreement made or signed?",
        "Notice Period To Terminate Renewal": "What is the notice period to terminate renewal?"
    }
    for provision in provision_types:
        answer_col = f"{provision}-Answer"
        doc_col = "Document Name"
        if answer_col in df.columns and doc_col in df.columns:
            valid_rows = df[(df[answer_col].notna()) & (df[doc_col].notna())]
            for _, row in valid_rows.iterrows():
                doc_text = row[doc_col]
                answer_text = row[answer_col]
                if isinstance(doc_text, str) and isinstance(answer_text, str):
                    if len(doc_text.strip()) > 50 and len(answer_text.strip()) > 3:
                        contexts.append(doc_text.strip())
                        questions.append(question_map.get(provision, f"What is the {provision.lower()}?"))
                        answers.append(answer_text.strip())
    return {"contexts": contexts, "questions": questions, "answers": answers}

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
