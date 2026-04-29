from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def f_beta_i(t, beta_a, beta_b, beta_c, beta_i):
    return beta_a * np.exp(-beta_b * t + beta_c) + beta_i


def f_delta_i(t, delta_a, delta_b, delta_c, delta_i):
    return delta_a * np.exp(-delta_b * t + delta_c) + delta_i


def beta_rate(t: float, beta_a, beta_b, beta_c, beta_i) -> np.ndarray:
    return np.asarray(f_beta_i(t, beta_a, beta_b, beta_c, beta_i), dtype=float)


def delta_rate(t: float, delta_a, delta_b, delta_c, delta_i) -> np.ndarray:
    return np.asarray(f_delta_i(t, delta_a, delta_b, delta_c, delta_i), dtype=float)


def integrate_beta_delta(
    t1: float,
    t2: float,
    *,
    beta_a: np.ndarray,
    beta_b: np.ndarray,
    beta_c: np.ndarray,
    beta_i: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    delta_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(beta_i)
    beta_int = np.zeros(n, dtype=float)
    delta_int = np.zeros(n, dtype=float)

    for i in range(n):
        beta_int[i], _ = quad(f_beta_i, t1, t2, args=(beta_a[i], beta_b[i], beta_c[i], beta_i[i]))
        delta_int[i], _ = quad(f_delta_i, t1, t2, args=(delta_a[i], delta_b[i], delta_c[i], delta_i[i]))

    return beta_int, delta_int



