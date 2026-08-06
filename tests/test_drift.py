import numpy as np
import pytest
from src.monitoring.drift import psi, ks_test


def test_psi_identical_distributions():
    rng = np.random.RandomState(42)
    data = rng.normal(50, 10, 10000)
    score = psi(data, data)
    assert score < 0.01


def test_psi_shifted_distribution():
    rng = np.random.RandomState(42)
    ref = rng.normal(50, 10, 10000)
    cur = rng.normal(70, 10, 10000)
    score = psi(ref, cur)
    assert score > 0.25


def test_psi_nonnegative():
    rng = np.random.RandomState(42)
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(52, 10, 5000)
    score = psi(ref, cur)
    assert score >= 0


def test_ks_identical():
    rng = np.random.RandomState(42)
    data = rng.normal(50, 10, 5000)
    stat, p = ks_test(data, data)
    assert p > 0.05
    assert stat < 0.05


def test_ks_different():
    rng = np.random.RandomState(42)
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(70, 10, 5000)
    stat, p = ks_test(ref, cur)
    assert p < 0.05


def test_psi_moderate_shift():
    rng = np.random.RandomState(42)
    ref = rng.normal(50, 10, 10000)
    cur = rng.normal(55, 12, 10000)
    score = psi(ref, cur)
    assert 0.0 < score < 0.5
