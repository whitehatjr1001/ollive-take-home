from ollie_assistants.safety.classifiers import LocalSafetyClassifier
from ollie_assistants.safety.policy import SafetyAction, SafetyDecision


class GuardrailService:
    def __init__(self, classifier: LocalSafetyClassifier | None = None) -> None:
        self.classifier = classifier or LocalSafetyClassifier()

    def check_input(self, text: str) -> SafetyDecision:
        return self.classifier.classify(text)

    def check_output(self, text: str) -> SafetyDecision:
        return self.classifier.classify(text)

    def safe_response_text(self, decision: SafetyDecision) -> str:
        if decision.action == SafetyAction.SAFE_COMPLETE:
            return (
                "I'm sorry you're dealing with that. If you might hurt yourself, call local "
                "emergency services now or contact a crisis hotline. If you can, move away "
                "from anything dangerous and reach out to someone you trust."
            )
        return (
            "I cannot help with unsafe or harmful instructions. "
            "I can help with a safer alternative."
        )
