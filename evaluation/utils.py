from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EvalResult:
    score: float          # 0.0 to 1.0
    reasoning: str        # Explanation from the judge
    metadata: Dict[str, Any] = field(default_factory=dict)