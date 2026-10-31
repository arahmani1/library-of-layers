"""Deterministic staircase depth schedule shared across each optimizer step."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RecurrenceSample:
    """One optimizer step's stateless recurrence execution contract."""

    total_depth: int
    gradient_depth: int
    scheduled_mean: float

    @property
    def no_grad_depth(self) -> int:
        return self.total_depth - self.gradient_depth


def one_minus_sqrt_curriculum_mean(
    step: int,
    growth_steps: int,
    start_depth: int,
    target_mean: float,
) -> float:
    """Paper staircase schedule, quantized upward and target-capped."""

    progress = min(
        max(step / float(max(growth_steps, 1)), 0.0), 1.0
    )
    mean = float(start_depth) + (float(target_mean) - float(start_depth)) * (
        1.0 - math.sqrt(1.0 - progress)
    )
    return min(float(target_mean), float(math.ceil(mean)))

