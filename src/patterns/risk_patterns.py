"""
Risk patterns for contract analysis.
"""

import regex
import logging
from typing import Dict, List, Any, Optional, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RiskPattern:
    """Pattern for detecting risks in contracts."""
    name: str
    patterns: List[str]
    category: str
    severity: str
    explanation: str
    impact: str
    context_keywords: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    suggested_alternative: str = ""
    
    def __post_init__(self):
        """Compile regex patterns after initialization."""
        self.compiled_patterns = [
            regex.compile(pattern, regex.IGNORECASE) 
            for pattern in self.patterns
        ]
        self.compiled_exceptions = [
            regex.compile(pattern, regex.IGNORECASE) 
            for pattern in self.exceptions
        ] if self.exceptions else []

class RiskPatternLibrary:
    """Library of risk patterns for contract analysis."""
    
    def __init__(self):
        """Initialize the risk pattern library."""
        self.risk_patterns = self._initialize_risk_patterns()
        
    def _initialize_risk_patterns(self) -> Dict[str, RiskPattern]:
        """Initialize comprehensive risk patterns."""
        risk_patterns = {}
        
        # Auto Renewal
        risk_patterns["auto_renewal"] = RiskPattern(
            name="Automatic Renewal",
            patterns=[
                r"auto(?:matic(?:ally)?)?[\-\s]+renew(?:al)?",
                r"(?:shall|will|is)\s+(?:automatically|be\s+deemed\s+to)\s+(?:renew|extend|continue)",
                r"renew(?:s|ed)?\s+(?:automatically|without\s+notice)",
                r"renewal\s+will\s+occur\s+unless\s+(?:notice|written\s+notice)\s+is\s+(?:provided|given|delivered)",
                r"continue\s+(?:in\s+effect|in\s+force)\s+for\s+(?:successive|additional)\s+(?:periods|terms)"
            ],
            category="renewal_risk",
            severity="medium",
            impact="Contract continues automatically without explicit consent",
            explanation="Automatic renewal provisions can extend the contract term without your express approval, potentially creating ongoing obligations you no longer want.",
            context_keywords=["term", "renewal", "notice", "terminate", "period"],
            exceptions=[
                r"option\s+to\s+renew", 
                r"may\s+be\s+renewed", 
                r"renewal\s+at\s+the\s+option",
                r"(?:non-renewal|not\s+to\s+renew).{1,30}?(?:at\s+least|minimum)\s+(?:\d+|thirty|sixty|ninety)"
            ],
            suggested_alternative="This Agreement may be renewed for additional one-year terms upon mutual written agreement of the parties at least 30 days prior to the expiration of the then-current term."
        )
        
        # Evergreen Renewal
        risk_patterns["evergreen_renewal"] = RiskPattern(
            name="Evergreen Renewal",
            patterns=[
                r"(?:renew|extend|continue).{1,30}?indefinitely",
                r"(?:perpetual|evergreen)\s+renewal",
                r"auto(?:matically)?\s+renew.{1,50}?until\s+(?:terminated|cancell?ed)",
                r"continue\s+in\s+(?:full\s+)?(?:force|effect).{1,30}?until\s+(?:terminated|cancell?ed)",
                r"successive\s+(?:renewal|additional)\s+(?:terms|periods).{1,50}?without\s+limitation"
            ],
            category="renewal_risk",
            severity="high",
            impact="Contract can renew indefinitely without a clear end point",
            explanation="Evergreen clauses create perpetual renewal cycles that can continue indefinitely until specific termination steps are taken, potentially trapping you in a long-term commitment.",
            context_keywords=["renew", "terminate", "perpetual", "indefinitely", "continuous"],
            exceptions=[
                r"not\s+to\s+exceed.{1,30}?(?:years|months|renewals)",
                r"maximum\s+of.{1,30}?renewal",
                r"limited\s+to.{1,30}?renewal"
            ],
            suggested_alternative="This Agreement may be renewed for up to two additional one-year terms upon mutual written agreement of the parties, for a maximum total contract duration of three years."
        )
        
        # Hidden Fee
        risk_patterns["hidden_fee"] = RiskPattern(
            name="Hidden Fee",
            patterns=[
                r"(?:additional|extra|other)\s+(?:fee|charge|cost)s?(?:\s+may\s+apply|\s+will\s+be\s+charged)?",
                r"(?:fee|charge)s?\s+(?:subject\s+to\s+change|may\s+be\s+modified)",
                r"reserves\s+(?:the\s+)?right\s+to\s+(?:impose|charge|assess)\s+(?:additional|new)\s+(?:fee|charge)s?",
                r"(?:fee|charge)s\s+(?:set\s+forth|listed|described)\s+(?:in\s+Schedule|in\s+Exhibit|on\s+(?:Company|Provider)'s\s+website)",
                r"(?:fee|charge)s\s+not\s+(?:listed|included|set\s+forth)"
            ],
            category="payment_risk",
            severity="high",
            impact="Unexpected charges not clearly disclosed upfront",
            explanation="Hidden fee provisions allow for additional charges beyond the primary fees, which may not be clearly disclosed or may be buried in complex language.",
            context_keywords=["fee", "charge", "payment", "cost", "expense", "invoice"],
            exceptions=[
                r"no\s+additional\s+fees", 
                r"without\s+additional\s+charge",
                r"full\s+amount\s+of\s+all\s+fees\s+is\s+set\s+forth"
            ],
            suggested_alternative="Any additional fees or charges must be agreed upon in writing by both parties before they are assessed. A complete schedule of all possible fees is attached as Exhibit A."
        )
        
        # Unilateral Price Change
        risk_patterns["unilateral_price_change"] = RiskPattern(
            name="Unilateral Price Change",
            patterns=[
                r"(?:price|fee|rate)s?\s+(?:may|can|subject\s+to)\s+(?:change|increase|adjust)",
                r"(?:reserve|right)\s+to\s+(?:change|increase|adjust|modify)\s+(?:price|fee|rate)",
                r"(?:price|fee|rate)\s+(?:changes|increases|adjustments)\s+(?:effective|will\s+apply)",
                r"may\s+(?:change|increase|adjust)\s+(?:price|fee|rate)s?.{1,30}?(?:at\s+any\s+time|from\s+time\s+to\s+time)",
                r"(?:price|fee|rate)s?\s+in\s+effect\s+at\s+the\s+time\s+of\s+(?:billing|invoicing|renewal)"
            ],
            category="payment_risk",
            severity="high",
            impact="Prices can be increased without your agreement",
            explanation="Unilateral price change provisions allow one party to increase prices or rates without requiring the other party's consent, creating financial uncertainty.",
            context_keywords=["price", "fee", "increase", "change", "adjust", "notice"],
            exceptions=[
                r"will\s+not\s+(?:change|increase|adjust)\s+(?:price|fee|rate)",
                r"(?:price|fee|rate)s?\s+(?:shall|will)\s+remain\s+fixed",
                r"increase.{1,30}?(?:limited\s+to|not\s+to\s+exceed|capped\s+at|maximum\s+of)\s+\d+%",
                r"increase.{1,30}?(?:CPI|Consumer\s+Price\s+Index|inflation)"
            ],
            suggested_alternative="Price increases shall not exceed 3% annually and shall only take effect upon renewal. Provider must provide at least 60 days' written notice of any price change, and Customer shall have the right to terminate without penalty if the price increase exceeds 3%."
        )
        
        # Minimum Commitment
        risk_patterns["minimum_commitment"] = RiskPattern(
            name="Minimum Commitment",
            patterns=[
                r"minimum\s+(?:purchase|commitment|spend|payment|fee)",
                r"(?:will|shall|must|agree\s+to)\s+(?:purchase|pay|buy|order)\s+(?:at\s+least|a\s+minimum\s+of)",
                r"minimum\s+(?:monthly|annual|quarterly|yearly)\s+(?:fee|charge|payment)",
                r"committed\s+(?:to\s+)?(?:paying|purchasing).{1,30}?(?:at\s+least|minimum)",
                r"(?:shall|will|must)\s+maintain\s+a\s+minimum\s+(?:balance|level|amount|quantity)"
            ],
            category="payment_risk",
            severity="medium",
            impact="You must pay a minimum amount regardless of actual usage",
            explanation="Minimum commitment provisions require payment of a minimum amount regardless of actual usage or benefit received, potentially resulting in paying for unused services.",
            context_keywords=["minimum", "commitment", "fee", "payment", "purchase"],
            exceptions=[
                r"no\s+minimum\s+(?:purchase|commitment|spend|payment)",
                r"without\s+any\s+minimum\s+(?:purchase|commitment|spend|payment)"
            ],
            suggested_alternative="Customer will pay based on actual usage with no minimum commitment requirement."
        )
        
        # Termination Without Notice
        risk_patterns["termination_without_notice"] = RiskPattern(
            name="Termination Without Notice",
            patterns=[
                r"terminat(?:e|ion)\s+(?:immediately|at\s+any\s+time)\s+(?:without|with\s+no)\s+(?:prior\s+)?notice",
                r"immediate(?:ly)?\s+terminat(?:e|ion)",
                r"(?:may|can|reserves\s+the\s+right\s+to)\s+terminat(?:e|ion)\s+at\s+(?:any\s+time|its\s+(?:sole\s+)?discretion)",
                r"right\s+to\s+terminate\s+this\s+Agreement\s+at\s+any\s+time\s+(?:with|without)\s+cause",
                r"terminat(?:e|ion)\s+(?:with|without)\s+cause\s+(?:at\s+any\s+time|upon\s+(?:notice|written\s+notice))"
            ],
            category="termination_risk",
            severity="high",
            impact="Contract can be ended immediately without warning",
            explanation="Termination without notice provisions allow one party to immediately end the contract without advance warning, which can create operational disruption and uncertainty.",
            context_keywords=["terminate", "termination", "end", "cancel", "notice"],
            exceptions=[
                r"terminate\s+upon\s+(?:\d+|thirty|sixty|ninety)\s+days[\'\s]+notice",
                r"terminate\s+with\s+(?:at\s+least|minimum)\s+(?:\d+|thirty|sixty|ninety)\s+days[\'\s]+notice",
                r"immediate\s+termination.{1,50}?(?:only|solely)\s+(?:in|for|upon).{1,30}?(?:material breach|fraud|bankruptcy|insolvency)"
            ],
            suggested_alternative="Either party may terminate this Agreement: (a) upon 30 days' written notice to the other party; or (b) immediately upon written notice in the event of a material breach by the other party that remains uncured after 10 days' written notice."
        )
        
        # Early Termination Fee
        risk_patterns["early_termination_fee"] = RiskPattern(
            name="Early Termination Fee",
            patterns=[
                r"early\s+termination\s+fee",
                r"fee\s+for\s+(?:early|premature)\s+termination",
                r"(?:terminat|cancell?|end).{1,20}?before.{1,30}?(?:pay|fee|charge|penalty)",
                r"terminat(?:e|ion).{1,30}?(?:pay|owe|liable).{1,30}?(?:remaining|balance|outstanding)",
                r"terminat(?:e|ion).{1,30}?(?:subject\s+to|requires).{1,30}?(?:fee|payment|charge)"
            ],
            category="termination_risk",
            severity="medium",
            impact="Financial penalty for ending the contract before its end date",
            explanation="Early termination fees impose financial penalties for ending the contract before its full term, which can create significant costs for changing providers.",
            context_keywords=["termination", "early", "fee", "penalty", "pay", "cancel"],
            exceptions=[
                r"no\s+(?:fee|charge|penalty)\s+for\s+(?:early|premature)\s+termination",
                r"may\s+terminate.{1,50}?without\s+(?:fee|charge|penalty)"
            ],
            suggested_alternative="Either party may terminate this Agreement with 30 days' written notice without penalty or additional fees."
        )
        
        # Add more risk patterns...
        
        # Unilateral Modification
        risk_patterns["unilateral_modification"] = RiskPattern(
            name="Unilateral Modification",
            patterns=[
                r"(?:unilateral(?:ly)?|sole\s+discretion)\s+(?:modif|chang|alter|amend|updat)",
                r"(?:reserve|retain)s?\s+the\s+right\s+to\s+(?:modif|chang|alter|amend)",
                r"(?:may|can)\s+(?:modif|chang|alter|amend)\s+(?:at\s+any\s+time|without\s+(?:prior\s+)?notice)",
                r"(?:may|can)\s+(?:modif|chang|alter|amend)[\s\w]+(?:from\s+time\s+to\s+time|at\s+its\s+(?:sole\s+)?discretion)",
                r"(?:modif|chang|alter|amend)s?\s+to\s+this\s+Agreement\s+(?:will\s+be\s+effective|shall\s+take\s+effect|become\s+effective)\s+upon\s+posting"
            ],
            category="modification_risk",
            severity="medium",
            impact="Contract terms can be changed without your agreement",
            explanation="Unilateral modification provisions allow one party to change contract terms without the other's consent, creating uncertainty about future rights and obligations.",
            context_keywords=["modify", "change", "alter", "amend", "update", "revise"],
            exceptions=[
                r"(?:modif|chang|alter|amend)(?:ication)?\s+(?:with|upon|requires|subject\s+to)\s+(?:written\s+consent|mutual\s+agreement)",
                r"(?:modif|chang|alter|amend)(?:ication)?\s+must\s+be\s+(?:in\s+writing|written)\s+and\s+signed\s+by\s+both\s+parties"
            ],
            suggested_alternative="Any modification to this Agreement must be in writing and signed by authorized representatives of both parties."
        )
        
        # Unlimited Liability
        risk_patterns["unlimited_liability"] = RiskPattern(
            name="Unlimited Liability",
            patterns=[
                r"unlimited\s+liability",
                r"(?:all|any)\s+liability\s+(?:regardless|irrespective|without\s+limitation)",
                r"liable\s+for\s+(?:all|any)\s+(?:loss|damage|claim)",
                r"indemnify\s+against\s+(?:all|any)\s+(?:loss|damage|claim|liability)",
                r"indemnify,\s+defend,\s+and\s+hold\s+harmless.{1,50}?(?:any\s+and\s+all|all)\s+(?:losses|damages|claims|liabilities)"
            ],
            category="liability_risk",
            severity="high",
            impact="No cap on potential financial responsibility for damages",
            explanation="Unlimited liability provisions create unbounded financial exposure for damages, potentially threatening your organization's financial stability.",
            context_keywords=["liability", "indemnify", "damage", "claim", "loss", "limitation"],
            exceptions=[
                r"neither\s+party\s+shall\s+be\s+liable",
                r"liability\s+(?:shall|will)\s+not\s+exceed",
                r"aggregate\s+liability.{1,30}?limited\s+to",
                r"limit(?:ed|ation)?\s+of\s+liability"
            ],
            suggested_alternative="Each party's total cumulative liability for all claims arising under or relating to this Agreement shall not exceed the total amounts paid or payable under this Agreement in the twelve (12) months preceding the event giving rise to liability."
        )
        
        return risk_patterns
    
    def detect_risks(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect risks in contract text.
        
        Args:
            text: Contract text to analyze
            
        Returns:
            List of detected risks with metadata
        """
        results = []
        
        for risk_name, risk_pattern in self.risk_patterns.items():
            # Skip if an exception pattern matches first
            if any(exc_pattern.search(text) for exc_pattern in risk_pattern.compiled_exceptions):
                continue
                
            # Check context keywords to improve precision
            has_context = not risk_pattern.context_keywords or any(
                keyword in text.lower() for keyword in risk_pattern.context_keywords
            )
            
            if not has_context:
                continue
                
            # Check each pattern
            for i, pattern in enumerate(risk_pattern.compiled_patterns):
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
                        
                    # Create risk finding
                    risk = {
                        'risk_id': f"{risk_name}_{i}",
                        'risk_name': risk_pattern.name,
                        'risk_category': risk_pattern.category,
                        'risk_level': risk_pattern.severity,
                        'matched_text': match.group(0),
                        'context': context,
                        'explanation': risk_pattern.explanation,
                        'impact': risk_pattern.impact,
                        'suggested_alternative': risk_pattern.suggested_alternative,
                        'confidence': 1.0,  # Rule-based patterns have high confidence
                        'detection_method': 'rule_based'
                    }
                    
                    results.append(risk)
                    break  # Once we find a match for this risk, move to next pattern
                    
        return results