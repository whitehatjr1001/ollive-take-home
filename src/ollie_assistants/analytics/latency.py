from collections.abc import Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def p50(values: Sequence[float]) -> float:
    return percentile(values, 0.50)


def p95(values: Sequence[float]) -> float:
    return percentile(values, 0.95)


def p99(values: Sequence[float]) -> float:
    return percentile(values, 0.99)
