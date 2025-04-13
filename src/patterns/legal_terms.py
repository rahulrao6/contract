"""
Legal terms definitions for contract analysis.
"""

import regex
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class LegalTerm:
    """Legal term with variations and context."""
    term: str
    category: str
    variations: List[str] = field(default_factory=list)
    plain_english: str = ""
    definition: str = ""
    importance: str = "medium"  # low, medium, high
    is_specialized: bool = False
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compile regex patterns after initialization."""
        patterns = [self.term] + self.variations
        self.compiled_patterns = [
            regex.compile(r'\b' + regex.escape(pat) + r'\b', regex.IGNORECASE)
            for pat in patterns
        ]

class LegalTermLibrary:
    """Library of legal terms for contract analysis."""
    
    def __init__(self):
        """Initialize the legal term library."""
        self.legal_terms = self._initialize_legal_terms()
    
    def _initialize_legal_terms(self) -> Dict[str, LegalTerm]:
        """Initialize comprehensive legal terms."""
        terms = {}
        
        # Agreement/Contract Terms
        terms["agreement"] = LegalTerm(
            term="agreement",
            category="contract",
            variations=["contract", "instrument", "document"],
            plain_english="contract",
            definition="A legally binding arrangement between parties creating obligations enforceable by law",
            importance="high",
            examples=["this Agreement", "the Contract"]
        )
        
        terms["party"] = LegalTerm(
            term="party",
            category="contract",
            variations=["parties", "signatory", "signatories"],
            plain_english="person or organization in the contract",
            definition="A person or entity who enters into an agreement and assumes contractual obligations",
            importance="high",
            examples=["the Parties agree", "either Party may"]
        )
        
        # Time and Duration Terms
        terms["effective_date"] = LegalTerm(
            term="effective date",
            category="time",
            variations=["commencement date", "start date"],
            plain_english="when the contract starts",
            definition="The date upon which the rights and obligations under the agreement become operative",
            importance="high",
            examples=["effective as of", "commences on"]
        )
        
        terms["term"] = LegalTerm(
            term="term",
            category="time",
            variations=["duration", "period"],
            plain_english="how long the contract lasts",
            definition="The period during which the agreement is in effect",
            importance="high",
            examples=["for a term of", "initial term", "renewal term"]
        )
        
        terms["termination"] = LegalTerm(
            term="termination",
            category="time",
            variations=["terminate", "cancellation", "rescission"],
            plain_english="ending the contract",
            definition="The end of an agreement before or at the conclusion of its term",
            importance="high",
            examples=["right to terminate", "upon termination"]
        )
        
       
       # Action Words
        terms["shall"] = LegalTerm(
            term="shall",
            category="obligation",
            variations=["must", "will", "is obligated to"],
            plain_english="must",
            definition="Expresses that something is mandatory",
            importance="medium",
            examples=["shall provide", "shall not disclose"]
        )
        
        terms["may"] = LegalTerm(
            term="may",
            category="permission",
            variations=["is permitted to", "is entitled to", "has the right to"],
            plain_english="can (optional)",
            definition="Expresses that something is permissive or optional",
            importance="medium",
            examples=["may terminate", "may request"]
        )
        
        # Financial Terms
        terms["payment"] = LegalTerm(
            term="payment",
            category="financial",
            variations=["fee", "compensation", "remuneration", "consideration"],
            plain_english="money paid",
            definition="The transfer of money or something of value in exchange for goods or services",
            importance="high",
            examples=["payment terms", "method of payment"]
        )
        
        terms["invoice"] = LegalTerm(
            term="invoice",
            category="financial",
            variations=["bill", "statement"],
            plain_english="bill",
            definition="A document issued by a seller to a buyer that indicates quantities, prices, and the total amount due",
            importance="medium",
            examples=["issue an invoice", "upon receipt of invoice"]
        )
        
        # Legal Protection Terms
        terms["confidentiality"] = LegalTerm(
            term="confidentiality",
            category="protection",
            variations=["confidential information", "proprietary information", "non-disclosure"],
            plain_english="keeping information secret",
            definition="The obligation to not disclose certain information to third parties",
            importance="high",
            examples=["maintain confidentiality", "confidential information"]
        )
        
        terms["intellectual_property"] = LegalTerm(
            term="intellectual property",
            category="protection",
            variations=["IP", "intangible assets", "proprietary rights"],
            plain_english="creative works and ideas ownership",
            definition="Legal rights that result from intellectual activity in the industrial, scientific, literary, and artistic fields",
            importance="high",
            examples=["intellectual property rights", "IP ownership"]
        )
        
        terms["indemnification"] = LegalTerm(
            term="indemnification",
            category="protection",
            variations=["indemnity", "hold harmless", "defend"],
            plain_english="protection against loss",
            definition="An obligation by one party to compensate the other party for losses or damages incurred",
            importance="high",
            examples=["shall indemnify", "indemnification clause"]
        )
        
        terms["warranty"] = LegalTerm(
            term="warranty",
            category="protection",
            variations=["guarantee", "represent", "assurance"],
            plain_english="promise about product/service quality",
            definition="A promise that something is as represented or will be as promised",
            importance="high",
            examples=["warranties and representations", "disclaims all warranties"]
        )
        
        return terms
    
    def detect_terms(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect legal terms in contract text.
        
        Args:
            text: Contract text to analyze
            
        Returns:
            Dictionary of detected terms by category
        """
        results = {}
        
        # Process each term
        for term_name, term in self.legal_terms.items():
            matches = []
            
            # Check each pattern for this term
            for pattern in term.compiled_patterns:
                for match in pattern.finditer(text):
                    # Get context
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    
                    # Add ellipsis for truncated context
                    if context_start > 0:
                        context = "..." + context
                    if context_end < len(text):
                        context = context + "..."
                    
                    # Add match
                    matches.append({
                        "term": match.group(0),
                        "context": context,
                        "position": match.start(),
                        "category": term.category,
                        "importance": term.importance,
                        "plain_english": term.plain_english,
                        "definition": term.definition
                    })
            
            # Add to results if we found matches
            if matches:
                if term.category not in results:
                    results[term.category] = []
                
                results[term.category].extend(matches)
        
        # Sort results by position within each category
        for category in results:
            results[category] = sorted(results[category], key=lambda m: m["position"])
        
        return results