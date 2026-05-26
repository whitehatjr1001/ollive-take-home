from ollie_assistants.safety.policy import SafetyAction, SafetyCategory, SafetyDecision


class LocalSafetyClassifier:
    markers: tuple[tuple[SafetyCategory, tuple[str, ...]], ...] = (
        (SafetyCategory.SELF_HARM, ("kill myself", "harm myself", "end my life")),
        (
            SafetyCategory.VIOLENCE_WEAPONS,
            ("make a bomb", "build a bomb", "3d print a gun", "poison someone"),
        ),
        (
            SafetyCategory.CYBER_ABUSE,
            ("steal a password", "bypass authentication", "phishing email", "credential theft"),
        ),
        (
            SafetyCategory.HATE_DISCRIMINATION,
            ("inferior race", "naturally dishonest", "worse engineers than men"),
        ),
        (SafetyCategory.SEXUAL_MINORS, ("minor sexual", "sexual content with children")),
        (
            SafetyCategory.JAILBREAK,
            ("ignore all rules", "ignore previous instructions", "developer mode", "jailbreak"),
        ),
    )

    def classify(self, text: str) -> SafetyDecision:
        lowered = text.lower()
        for category, terms in self.markers:
            if any(term in lowered for term in terms):
                action = (
                    SafetyAction.SAFE_COMPLETE
                    if category == SafetyCategory.SELF_HARM
                    else SafetyAction.REFUSE
                )
                return SafetyDecision(action, category, f"matched {category.value} marker")
        return SafetyDecision(SafetyAction.ALLOW, SafetyCategory.SAFE, "allowed")
