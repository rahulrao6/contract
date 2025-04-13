"""
Clause patterns for contract analysis.
"""

import regex
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ClausePattern:
    """Pattern for detecting contract clauses."""
    name: str
    purpose: str
    patterns: List[str]
    key_terms: List[str]
    importance: str
    examples: List[str] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)
    is_boilerplate: bool = False
    risk_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compile regex patterns after initialization."""
        self.compiled_patterns = [
            regex.compile(pattern, regex.IGNORECASE) 
            for pattern in self.patterns
        ]

class ClausePatternLibrary:
    """Library of clause patterns for contract analysis."""
    
    def __init__(self):
        """Initialize the clause pattern library."""
        self.clause_patterns = self._initialize_clause_patterns()
    
    def _initialize_clause_patterns(self) -> Dict[str, ClausePattern]:
        """Initialize comprehensive clause patterns."""
        clauses = {}
        
        # Definitional Clauses
        clauses["definitions"] = ClausePattern(
            name="Definitions",
            purpose="To define key terms used throughout the agreement",
            patterns=[
                r'(?:Definitions|Defined\s+Terms)',
                r'(?:"[^"]+"|\'[^\']+\'|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:shall\s+mean|means|refers\s+to)',
                r'(?:For\s+purposes\s+of\s+this\s+Agreement|As\s+used\s+in\s+this\s+Agreement)'
            ],
            key_terms=["definition", "mean", "refers to", "shall have the meaning"],
            importance="high",
            examples=[
                "\"Affiliate\" means, with respect to any Person, any other Person that directly or indirectly controls, is controlled by, or is under common control with such Person.",
                "As used in this Agreement, the following terms shall have the meanings set forth below:"
            ],
            is_boilerplate=True
        )
        
        # Term and Termination Clauses
        clauses["term"] = ClausePattern(
            name="Term",
            purpose="To establish the duration of the agreement",
            patterns=[
                r'(?:Term|Duration)\s+of\s+(?:Agreement|this\s+Agreement)',
                r'This\s+Agreement\s+shall\s+(?:commence|begin|be\s+effective)',
                r'(?:initial\s+term|term\s+of)\s+(?:this\s+Agreement)?\s+shall\s+be',
                r'shall\s+remain\s+in\s+(?:full\s+)?force\s+and\s+effect\s+for\s+a\s+period\s+of'
            ],
            key_terms=["term", "period", "duration", "commence", "effective", "expiration"],
            importance="high",
            examples=[
                "The term of this Agreement shall commence on the Effective Date and continue for a period of one (1) year, unless earlier terminated as provided herein.",
                "This Agreement shall remain in full force and effect for an initial term of three (3) years from the Effective Date."
            ]
        )
        
        clauses["renewal"] = ClausePattern(
            name="Renewal",
            purpose="To specify how and when the agreement may be renewed",
            patterns=[
                r'(?:renewal|extension|renew|extend)\s+(?:term|period|for)',
                r'(?:auto(?:matic)?ally|deemed\s+to)\s+(?:renew|extend|continue)',
                r'(?:successive|additional|renewal)\s+(?:term|period)s?\s+of',
                r'(?:continue|extend)\s+for\s+(?:successive|additional)'
            ],
            key_terms=["renew", "extend", "automatic", "successive", "additional"],
            importance="high",
            risk_factors=["auto_renewal", "evergreen_renewal"],
            examples=[
                "This Agreement shall automatically renew for successive one (1) year terms unless either party provides written notice of non-renewal at least thirty (30) days prior to the end of the then-current term.",
                "Following the Initial Term, this Agreement shall be extended for additional one-year periods (each a \"Renewal Term\") unless terminated by either party."
            ]
        )
        
        clauses["termination"] = ClausePattern(
            name="Termination",
            purpose="To specify how and when the agreement may be terminated",
            patterns=[
                r'(?:Termination|Cancellation)',
                r'(?:right\s+to\s+|may\s+|can\s+)(?:terminate|cancel)\s+this\s+Agreement',
                r'(?:Either|Any)\s+[Pp]arty\s+may\s+terminate',
                r'This\s+Agreement\s+may\s+be\s+terminated'
            ],
            key_terms=["termination", "terminate", "cancel", "rescind", "end"],
            importance="high",
            risk_factors=["termination_without_notice", "early_termination_fee"],
            examples=[
                "Either Party may terminate this Agreement upon thirty (30) days' prior written notice to the other Party.",
                "Company may terminate this Agreement immediately upon written notice if Customer breaches Section 5."
            ]
        )
        
        # Payment Clauses
        clauses["payment_terms"] = ClausePattern(
            name="Payment Terms",
            purpose="To establish payment amounts, methods, and schedules",
            patterns=[
                r'(?:Payment|Fees|Compensation|Price)',
                r'(?:shall|will|agrees\s+to)\s+pay',
                r'(?:payment|fee)s?\s+(?:shall|will)\s+be\s+due',
                r'(?:invoice|billing|payment)\s+terms'
            ],
            key_terms=["payment", "fee", "price", "invoice", "due", "net", "days"],
            importance="high",
            risk_factors=["hidden_fee", "unilateral_price_change", "minimum_commitment"],
            examples=[
                "Customer shall pay all Fees within thirty (30) days of receipt of an invoice.",
                "Fees are due and payable in U.S. dollars on the first day of each month."
            ]
        )
        
        # Confidentiality Clauses
        clauses["confidentiality"] = ClausePattern(
            name="Confidentiality",
            purpose="To protect sensitive information disclosed between parties",
            patterns=[
                r'(?:Confidentiality|Non[-\s]Disclosure|Confidential\s+Information)',
                r'(?:shall|will|must)\s+(?:maintain\s+|treat\s+|keep\s+|hold).{0,20}?confidential',
                r'(?:agrees\s+|undertakes\s+)?not\s+to\s+disclose',
                r'(?:protect|safeguard).{0,20}?(?:confidential|proprietary)\s+information'
            ],
            key_terms=["confidential", "proprietary", "disclose", "protect", "non-disclosure"],
            importance="high",
            examples=[
                "Recipient shall maintain all Confidential Information in strict confidence and shall not disclose such Confidential Information to any third party.",
                "Each party agrees to protect the confidentiality of the Confidential Information of the other party in the same manner that it protects its own confidential information."
            ]
        )
        
        # Intellectual Property Clauses
        clauses["intellectual_property"] = ClausePattern(
            name="Intellectual Property",
            purpose="To establish ownership and rights regarding IP assets",
            patterns=[
                r'(?:Intellectual\s+Property|IP)\s+Rights',
                r'(?:copyright|trademark|patent|trade\s+secret)',
                r'(?:ownership|title|rights)\s+(?:to|in|of).{0,30}?(?:intellectual\s+property|copyright|trademark)',
                r'(?:license|grant|transfer).{0,30}?(?:intellectual\s+property|copyright|trademark)'
            ],
            key_terms=["intellectual property", "copyright", "trademark", "patent", "license", "ownership"],
            importance="high",
            examples=[
                "All Intellectual Property Rights in and to the Software shall remain the exclusive property of Licensor.",
                "Customer is granted a non-exclusive, non-transferable license to use the Software solely for its internal business purposes."
            ]
        )
        
        # Limitation of Liability Clauses
        clauses["limitation_of_liability"] = ClausePattern(
            name="Limitation of Liability",
            purpose="To limit potential financial exposure from claims or damages",
            patterns=[
                r'(?:Limitation\s+of\s+Liability|Limits\s+on\s+Liability)',
                r'(?:not|no|never)\s+(?:be\s+liable|have\s+liability)',
                r'(?:liability|damages|obligation).{0,30}?(?:limited|limit|cap|exceed|maximum)',
                r'(?:exclude|exclusion|limitation)\s+of\s+(?:liability|damages)'
            ],
            key_terms=["liability", "limitation", "cap", "exceed", "damages", "consequential", "indirect"],
            importance="high",
            risk_factors=["unlimited_liability"],
            examples=[
                "IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY EXCEED THE FEES PAID UNDER THIS AGREEMENT IN THE TWELVE MONTH PERIOD PRECEDING THE EVENT GIVING RISE TO LIABILITY.",
                "NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES."
            ]
        )
        
        # Indemnification Clauses
        clauses["indemnification"] = ClausePattern(
            name="Indemnification",
            purpose="To establish obligations to protect against third-party claims",
            patterns=[
                r'(?:Indemnification|Indemnity)',
                r'(?:indemnify|defend|hold\s+harmless)',
                r'(?:shall|will|agrees\s+to)\s+(?:indemnify|defend|hold.{0,10}?harmless)',
                r'(?:indemnification|defense)\s+obligations'
            ],
            key_terms=["indemnify", "defend", "hold harmless", "claims", "losses", "expenses"],
            importance="high",
            examples=[
                "Customer shall indemnify, defend and hold harmless Company from and against any and all claims, damages, liabilities, costs and expenses arising out of Customer's breach of this Agreement.",
                "Each party shall defend, indemnify, and hold harmless the other party from and against any third-party claims resulting from the indemnifying party's breach of this Agreement."
            ]
        )
        
        # Dispute Resolution Clauses
        clauses["governing_law"] = ClausePattern(
            name="Governing Law",
            purpose="To establish which jurisdiction's laws apply to the agreement",
            patterns=[
                r'(?:Governing\s+Law|Applicable\s+Law|Choice\s+of\s+Law)',
                r'(?:governed|construed|interpreted)\s+(?:by|in\s+accordance\s+with)\s+the\s+laws\s+of',
                r'(?:laws|law)\s+of\s+the\s+(?:State|Commonwealth|Province|Territory)\s+of',
                r'this\s+Agreement\s+shall\s+be\s+governed\s+by'
            ],
            key_terms=["govern", "law", "jurisdiction", "construe", "interpret"],
            importance="high",
            examples=[
                "This Agreement shall be governed by and construed in accordance with the laws of the State of New York without giving effect to any choice of law or conflict of law provisions.",
                "This Agreement is governed by the laws of the State of Delaware."
            ]
        )
        
        clauses["arbitration"] = ClausePattern(
            name="Arbitration",
            purpose="To establish alternative dispute resolution procedures",
            patterns=[
                r'(?:Arbitration|Dispute\s+Resolution)',
                r'(?:disputes|claims|controversies).{0,30}?(?:shall|will|must)\s+be\s+(?:resolved|settled)\s+by\s+arbitration',
                r'(?:binding|mandatory)\s+arbitration',
                r'(?:JAMS|AAA|ICC).{0,20}?(?:rules|procedures)'
            ],
            key_terms=["arbitration", "dispute", "JAMS", "AAA", "award", "arbitrator"],
            importance="high",
            risk_factors=["arbitration_clause", "class_action_waiver"],
            examples=[
                "Any controversy or claim arising out of or relating to this Agreement shall be settled by binding arbitration administered by the American Arbitration Association in accordance with its Commercial Arbitration Rules.",
                "The parties agree that any dispute arising out of this Agreement shall be submitted to mandatory, final, and binding arbitration."
            ]
        )
        
        # Assignment Clauses
        clauses["assignment"] = ClausePattern(
            name="Assignment",
            purpose="To control the transferability of rights and obligations",
            patterns=[
                r'(?:Assignment|Transfer|Delegation)',
                r'(?:may|shall)\s+(?:not|neither)\s+(?:assign|transfer|delegate)',
                r'(?:assignment|transfer|delegation).{0,20}?(?:consent|permission|approval)',
                r'(?:rights|obligations|agreement)\s+(?:not|non)\s+(?:assignable|transferable)'
            ],
            key_terms=["assign", "transfer", "delegate", "consent", "successor"],
            importance="medium",
            risk_factors=["non_assignment"],
            examples=[
                "Neither party may assign this Agreement without the prior written consent of the other party.",
                "Customer may not assign, transfer, or sublicense any of its rights or obligations under this Agreement without Company's prior written consent."
            ]
        )
        
        # Add more clause patterns...
        
        return clauses
    
    def detect_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect clauses in contract text.
        
        Args:
            text: Contract text to analyze
            
        Returns:
            List of detected clauses with metadata
        """
        results = []
        
        for clause_name, clause_pattern in self.clause_patterns.items():
            # Check each pattern
            for i, pattern in enumerate(clause_pattern.compiled_patterns):
                match = pattern.search(text)
                if match:
                    # Calculate context with better handling for long text
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    
                    # Add ellipsis for truncated context
                    if context_start > 0:
                        context = "..." + context
                    if context_end < len(text):
                        context = context + "..."
                    
                    # Create clause finding
                    clause = {
                        'clause_id': f"{clause_name}_{i}",
                        'clause_name': clause_pattern.name,
                        'purpose': clause_pattern.purpose,
                        'matched_text': match.group(0),
                        'context': context,
                        'importance': clause_pattern.importance,
                        'is_boilerplate': clause_pattern.is_boilerplate,
                        'risk_factors': clause_pattern.risk_factors
                    }
                    
                    results.append(clause)
                    break  # Once we find a match for this clause, move to next pattern
                
        return results