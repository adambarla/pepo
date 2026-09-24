"""Tests for the mixture-proposal rejection sampler acceptance rule."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch

from pepo.generator.best_of_n import acceptance_log_probs

Mode = Literal["min", "mean_std"]


def _members(num_models: int = 3, num_actions: int = 6, seed: int = 0) -> torch.Tensor:
    """(L, A) member distributions over a small discrete response space."""
    gen = torch.Generator().manual_seed(seed)
    logits = 3.0 * torch.randn(num_models, num_actions, generator=gen)
    return torch.softmax(logits.double(), dim=-1)


def _accepted_distribution(probs: torch.Tensor, mode: Mode, eta: float) -> torch.Tensor:
    """Distribution of accepted samples: q(a) * alpha(a), normalized."""
    alpha = acceptance_log_probs(probs.log(), mode, eta).exp()
    accepted = probs.mean(dim=0) * alpha
    return accepted / accepted.sum()


def test_min_mode_samples_normalized_minimum() -> None:
    probs = _members()
    target = probs.min(dim=0).values
    expected = target / target.sum()
    assert torch.allclose(_accepted_distribution(probs, "min", 0.0), expected)


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5])
def test_mean_std_mode_samples_mean_minus_std(eta: float) -> None:
    probs = _members()
    target = (probs.mean(dim=0) - eta * probs.std(dim=0, unbiased=False)).clamp(min=0.0)
    expected = target / target.sum()
    assert torch.allclose(_accepted_distribution(probs, "mean_std", eta), expected)


@pytest.mark.parametrize("mode,eta", [("min", 0.0), ("mean_std", 0.1)])
def test_acceptance_is_a_probability(mode: Mode, eta: float) -> None:
    alpha = acceptance_log_probs(_members().log(), mode, eta).exp()
    assert torch.all(alpha <= 1.0 + 1e-12)
    assert torch.all(alpha >= 0.0)


def test_mean_std_acceptance_lower_bound() -> None:
    """Samuelson: CV <= sqrt(L - 1), so acceptance >= 1 - eta * sqrt(L - 1)."""
    probs = _members(num_models=4, seed=1)
    eta = 0.1
    alpha = acceptance_log_probs(probs.log(), "mean_std", eta).exp()
    assert torch.all(alpha >= 1.0 - eta * math.sqrt(3) - 1e-12)


def test_sequence_log_probs_do_not_underflow() -> None:
    """Long responses have log-probs far below float32 exp range."""
    log_probs = torch.tensor([[-800.0, -1500.0], [-801.0, -1502.0]])
    modes: tuple[Mode, ...] = ("min", "mean_std")
    for mode in modes:
        log_alpha = acceptance_log_probs(log_probs, mode, 0.1)
        assert torch.all(torch.isfinite(log_alpha))
        assert torch.all(log_alpha <= 0.0)
