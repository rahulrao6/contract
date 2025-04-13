"""
Main entry point for Contract Analyzer.
"""

import os
import logging
import asyncio
import argparse
from typing import Dict, Any, Optional

from src.config import Config
from src.contract_analyzer import ContractAnalyzer
from src.api.server import create_app, run_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

async def analyze_file(file_path: str, config: Optional[Config] = None) -> Dict[str, Any]:
    """
    Analyze a contract file from the command line.
    
    Args:
        file_path: Path to contract file
        config: Optional configuration
        
    Returns:
        Analysis result as dictionary
    """
    if config is None:
        config = Config()
        
    analyzer = ContractAnalyzer(config)
    
    # Read file
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    # Analyze
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
            for risk in result.risks[:5]  # Top 5 risks
        ]
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Contract Analyzer")
    
    # Define subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run API server")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a contract file")
    analyze_parser.add_argument("file", help="Path to contract file")
    analyze_parser.add_argument("--output", "-o", help="Output file for analysis results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process command
    if args.command == "api":
        # Run API server
        run_server()
    elif args.command == "analyze":
        # Analyze file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
            
        # Run analysis
        result = asyncio.run(analyze_file(args.file))
        
        # Output results
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            import json
            print(json.dumps(result, indent=2))
    else:
        # Default to show help
        parser.print_help()

if __name__ == "__main__":
    main()