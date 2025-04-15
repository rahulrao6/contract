#!/usr/bin/env python3
"""
Main entry point for Contract Analyzer.
"""

import os
import logging
import asyncio
import argparse
import json

from src.config import Config
from src.contract_analyzer import ContractAnalyzer
from src.api.server import create_app, run_server  # Ensure these are correct for your API

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

async def analyze_file(file_path: str, config: Config) -> dict:
    """
    Analyze a contract file and return results.
    
    Args:
        file_path: Path to contract file.
        config: Configuration settings.
        
    Returns:
        Dictionary of analysis results.
    """
    analyzer = ContractAnalyzer(config)
    with open(file_path, 'rb') as f:
        file_content = f.read()
    filename = os.path.basename(file_path)
    result = await analyzer.analyze_document(file_content=file_content, filename=filename)
    return {
        "contract_id": result.contract_id,
        "contract_type": result.metadata.detected_contract_type,
        "risks_count": len(result.risks),
        "sections_count": len(result.sections),
        "summary": result.summary.overall_summary,
        "key_risks": [
            {
                "name": risk.risk_name,
                "level": risk.risk_level,
                "category": risk.risk_category,
                "description": risk.risk_description
            }
            for risk in result.risks[:5]
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Contract Analyzer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    api_parser = subparsers.add_parser("api", help="Run API server")
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a contract file")
    analyze_parser.add_argument("file", help="Path to contract file")
    analyze_parser.add_argument("--output", "-o", help="Output file for analysis results")
    args = parser.parse_args()
    
    config = Config()
    
    if args.command == "api":
        run_server()
    elif args.command == "analyze":
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        result = asyncio.run(analyze_file(args.file, config))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

app = create_app(Config())