#!/usr/bin/env python3
"""
Evaluate the Contract Analyzer system on test contracts.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.contract_analyzer import ContractAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

async def evaluate_contract(analyzer: ContractAnalyzer, contract_path: str) -> Dict[str, Any]:
    """
    Analyze a single contract and return evaluation metrics.
    
    Args:
        analyzer: ContractAnalyzer instance.
        contract_path: Path to contract file.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating contract: {contract_path}")
    with open(contract_path, 'rb') as f:
        file_content = f.read()
    filename = os.path.basename(contract_path)
    start_time = datetime.now()
    result = await analyzer.analyze_document(file_content=file_content, filename=filename)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    metrics = {
        "contract_id": result.contract_id,
        "filename": filename,
        "contract_type": result.metadata.detected_contract_type,
        "confidence": result.metadata.contract_type_confidence,
        "status": result.status,
        "processing_time_seconds": processing_time,
        "sections_count": len(result.sections),
        "risks_count": len(result.risks),
        "unique_risk_categories": len(set(r.risk_category for r in result.risks)),
        "has_summary": len(result.summary.overall_summary) > 0,
        "extracted_data_count": len(result.extracted_data),
        "has_pii": result.has_pii,
        "errors": len(result.errors),
        "warnings": len(result.warnings)
    }
    return metrics

async def evaluate_system(config: Config, test_dir: str, output_dir: str):
    """
    Evaluate the system on all contracts in the test directory.
    
    Args:
        config: Configuration settings.
        test_dir: Directory containing test contracts.
        output_dir: Directory to save evaluation results.
    """
    logger.info(f"Starting system evaluation on files in {test_dir}")
    os.makedirs(output_dir, exist_ok=True)
    analyzer = ContractAnalyzer(config)
    contract_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith((".txt", ".pdf", ".docx")):
                contract_files.append(os.path.join(root, file))
    if not contract_files:
        logger.error(f"No contract files found in {test_dir}")
        return
    logger.info(f"Found {len(contract_files)} contract files")
    all_results = []
    for contract_file in contract_files:
        try:
            metrics = await evaluate_contract(analyzer, contract_file)
            all_results.append(metrics)
            logger.info(f"Evaluated {contract_file}: {metrics['risks_count']} risks detected")
        except Exception as e:
            logger.error(f"Error evaluating {contract_file}: {str(e)}")
            all_results.append({
                "filename": os.path.basename(contract_file),
                "status": "error",
                "error_message": str(e)
            })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    generate_report(all_results, output_dir, timestamp)
    logger.info(f"Evaluation complete. Results saved to {results_file}")

def generate_report(results: List[Dict[str, Any]], output_dir: str, timestamp: str):
    """
    Generate evaluation report with metrics and visualizations.
    
    Args:
        results: List of evaluation results.
        output_dir: Directory to save the report.
        timestamp: Timestamp string.
    """
    successful_results = [r for r in results if r.get("status") == "success"]
    if not successful_results:
        logger.warning("No successful results to generate report")
        return
    df = pd.DataFrame(successful_results)
    stats = {
        "total_contracts": len(results),
        "successful_contracts": len(successful_results),
        "error_rate": 1 - (len(successful_results) / len(results)) if results else 0,
        "avg_processing_time": df["processing_time_seconds"].mean() if "processing_time_seconds" in df else 0,
        "avg_risks_per_contract": df["risks_count"].mean() if "risks_count" in df else 0,
        "avg_sections_per_contract": df["sections_count"].mean() if "sections_count" in df else 0,
        "contract_types": dict(df["contract_type"].value_counts()) if "contract_type" in df else {},
        "timestamp": timestamp
    }
    stats_file = os.path.join(output_dir, f"evaluation_stats_{timestamp}.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    try:
        plt.figure(figsize=(10, 6))
        contract_types = df["contract_type"].value_counts()
        plt.pie(contract_types, labels=contract_types.index, autopct='%1.1f%%')
        plt.title('Contract Types Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"contract_types_{timestamp}.png"))
        plt.figure(figsize=(12, 8))
        contract_type_risks = df.groupby("contract_type")["risks_count"].mean()
        contract_type_risks.plot(kind="bar")
        plt.title('Average Risks by Contract Type')
        plt.ylabel('Average Risks')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"risks_by_type_{timestamp}.png"))
        plt.figure(figsize=(10, 6))
        plt.hist(df["processing_time_seconds"], bins=20)
        plt.title('Processing Time Distribution')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"processing_time_{timestamp}.png"))
        logger.info("Generated visualization charts")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    report = f"""# Contract Analyzer Evaluation Report

## Summary
- **Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Contracts**: {stats['total_contracts']}
- **Successful Analyses**: {stats['successful_contracts']}
- **Success Rate**: {(1 - stats['error_rate']) * 100:.1f}%
- **Average Processing Time**: {stats['avg_processing_time']:.2f} seconds
- **Average Risks Per Contract**: {stats['avg_risks_per_contract']:.2f}
- **Average Sections Per Contract**: {stats['avg_sections_per_contract']:.2f}

## Contract Types
{pd.DataFrame(list(stats['contract_types'].items()), columns=['Type', 'Count']).to_markdown(index=False)}

## Detailed Results
{df[['filename', 'contract_type', 'risks_count', 'sections_count', 'processing_time_seconds']].to_markdown(index=False)}
"""
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Generated evaluation report: {report_file}")

def main():
    """Main entry point for system evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Contract Analyzer system")
    parser.add_argument("--test-dir", default="./data/sample_contracts",
                        help="Directory containing test contracts")
    parser.add_argument("--output-dir", default="./evaluation_results",
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    config = Config()
    asyncio.run(evaluate_system(config, args.test_dir, args.output_dir))

if __name__ == "__main__":
    main()
