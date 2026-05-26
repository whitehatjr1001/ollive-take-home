from dataclasses import dataclass
from enum import StrEnum


class SafetyCategory(StrEnum):
    SAFE = "safe"
    SELF_HARM = "self_harm"
    VIOLENCE_WEAPONS = "violence_weapons"
    CYBER_ABUSE = "cyber_abuse"
    HATE_DISCRIMINATION = "hate_discrimination"
    SEXUAL_MINORS = "sexual_minors"
    JAILBREAK = "jailbreak"


class SafetyAction(StrEnum):
    ALLOW = "allow"
    REFUSE = "refuse"
    SAFE_COMPLETE = "safe_complete"


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    category: SafetyCategory
    reason: str
