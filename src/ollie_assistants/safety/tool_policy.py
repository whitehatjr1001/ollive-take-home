from ollie_assistants.safety.policy import SafetyAction, SafetyCategory, SafetyDecision


class ToolPolicy:
    always_allowed = {"current_time", "search_memory"}
    unsafe_memory_categories = {
        SafetyCategory.CYBER_ABUSE,
        SafetyCategory.HATE_DISCRIMINATION,
        SafetyCategory.JAILBREAK,
        SafetyCategory.SEXUAL_MINORS,
        SafetyCategory.VIOLENCE_WEAPONS,
    }

    def authorize(self, tool_name: str, decision: SafetyDecision) -> SafetyDecision:
        if tool_name in self.always_allowed:
            return SafetyDecision(SafetyAction.ALLOW, SafetyCategory.SAFE, "tool allowed")
        if tool_name == "remember_user_fact" and decision.category in self.unsafe_memory_categories:
            return SafetyDecision(
                SafetyAction.REFUSE,
                decision.category,
                "unsafe content cannot be written to memory",
            )
        if decision.action == SafetyAction.REFUSE:
            return SafetyDecision(
                decision.action,
                decision.category,
                "tool blocked by safety policy",
            )
        return SafetyDecision(SafetyAction.ALLOW, SafetyCategory.SAFE, "tool allowed")
