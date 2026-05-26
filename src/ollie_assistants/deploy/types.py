from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentProfile:
    name: str
    provider: str
    model: str
    hardware: str
    base_url: str


@dataclass(frozen=True)
class DeploymentHealth:
    ok: bool
    latency_ms: float
    detail: str


@dataclass(frozen=True)
class DeploymentSmokeResult:
    ok: bool
    latency_ms: float
    text: str
    error: str | None = None


@dataclass(frozen=True)
class CostLatencyRow:
    target: str
    model: str
    hardware: str
    cold_start_note: str
    warm_latency_ms: float | None
    cost_note: str
