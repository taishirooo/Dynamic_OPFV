# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import numpy as np
from typing import Optional

def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.0,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    base_pol = np.zeros_like(expected_reward)
    if is_optimal:
        a = np.argmax(expected_reward, axis=1)
    else:
        a = np.argmin(expected_reward, axis=1)
    base_pol[
        np.arange(expected_reward.shape[0]),
        a,
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol[:, :]


def gen_eps_greedy_masked(
    expected_reward: np.ndarray,
    eps: float = 0.1,
    is_optimal: bool = True,
    available_actions: Optional[np.ndarray] = None,  # ← Optional[...] を使う
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    ε-greedy を可用アクション集合上で実装。
    expected_reward: (n, A) 期待報酬
    available_actions: (n, A) bool; True のみで方策を構成（None のときは全 True）
    返り値: (n, A) 行ごとに和=1 の分布
    """
    er = np.asarray(expected_reward, dtype=float)
    n, A = er.shape
    if available_actions is None:
        avail = np.ones((n, A), dtype=bool)
    else:
        avail = np.asarray(available_actions, dtype=bool)
        if avail.shape != (n, A):
            raise ValueError(f"`available_actions` must have shape {(n, A)}, got {avail.shape}")

    k = avail.sum(axis=1)
    no_avail = k == 0
    if np.any(no_avail):
        avail[no_avail, :] = True
        k[no_avail] = A

    rng = np.random.RandomState(random_state) if random_state is not None else None

    scores = er.copy()
    if is_optimal:
        scores[~avail] = -np.inf
        if rng is not None:
            scores += rng.gumbel(scale=1e-8, size=scores.shape)
        greedy_idx = np.argmax(scores, axis=1)
    else:
        scores[~avail] = np.inf
        if rng is not None:
            scores += rng.gumbel(scale=1e-8, size=scores.shape)
        greedy_idx = np.argmin(scores, axis=1)

    pi = np.zeros((n, A), dtype=float)
    pi[np.arange(n), greedy_idx] = 1.0
    pi *= (1.0 - eps)
    pi += eps * (avail.astype(float) / k.reshape(-1, 1))

    pi = np.clip(pi, 0.0, 1.0)
    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    pi /= row_sums
    return pi
