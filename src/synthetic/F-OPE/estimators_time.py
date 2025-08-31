# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from typing import Callable, Optional

import numpy as np

est_P_phi_t_for_reward_NEAR_ZERO = 0.0000001


# fourier function (scalar version)
def fourier_scalar(x, n, K_plus_delta):
    array = np.array(1)
    for i in range(1, n + 1):
        array = np.append(array, np.sin(2 * np.pi * i * x / (K_plus_delta)))
        array = np.append(array, np.cos(2 * np.pi * i * x / (K_plus_delta)))
    return array


# fourier function (vector version)
def fourier_vec(x, n, K_plus_delta):
    matrix = np.ones(len(x)).astype(int)
    for i in range(1, n + 1):
        matrix = np.column_stack((matrix, np.sin(2 * np.pi * i * x / (K_plus_delta))))
        matrix = np.column_stack((matrix, np.cos(2 * np.pi * i * x / (K_plus_delta))))
    return matrix


# exponetial function (scalar version)
def exponential_scalar(x, n, K_plus_delta):
    array = np.array(1)
    for i in range(1, n + 1):
        array = np.append(array, np.exp(-i * x / (K_plus_delta)))
        array = np.append(array, -np.exp(-i * x / (K_plus_delta)))
    return array


# exponetial function (vector version)
def exponential_vec(x, n, K_plus_delta):
    matrix = np.ones(len(x)).astype(int)
    for i in range(1, n + 1):
        matrix = np.column_stack((matrix, np.exp(-i * x / (K_plus_delta))))
        matrix = np.column_stack((matrix, -np.exp(i * x / (K_plus_delta))))
    return matrix


def create_array(N, K):
    quotient = N // K
    remainder = N % K

    if remainder == 0:
        A = np.array([quotient] * K)
    else:
        A = np.array([quotient + 1] * remainder + [quotient] * (K - remainder))

    return A


def create_episode_context(horizon_array, N, K):
    episode_context = np.zeros((N, K))
    count = 0
    for i in range(len(horizon_array)):
        for j in range(horizon_array[i]):
            episode_context[j + count, i] = 1
        count += horizon_array[i]
    return episode_context


def Prognosticator(
    num_episodes: int,
    phi_scalar_func,
    phi_vector_func,
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    action_dist: np.ndarray,
    num_episodes_after_logged_data: int = 1,
    num_features_for_Prognosticator: int = 3,
) -> np.ndarray:
    # Obtain the numbe of episodes (K) and the number of rounds (n)
    K = num_episodes
    n_rounds = action.shape[0]
    if num_episodes_after_logged_data == 0:
        num_episodes_after_logged_data = 1

    # Create the vector that contains independent variables when we use linear regression X \in R^K
    X = np.arange(1, K + 1)
    # Map each element of X using given feature function to create Phi \in R^(K \times d)
    Phi = phi_vector_func(
        X, num_features_for_Prognosticator, num_episodes_after_logged_data
    )

    # Create the vector that contains importance weight w(x_i, t_i, a_i) for each round i
    iw = action_dist[np.arange(n_rounds), action] / pscore
    round_rewards = reward * iw

    # How to choose each horizon
    horizon_array = create_array(n_rounds, K)
    episode_context = create_episode_context(horizon_array, n_rounds, K)

    # Calculate the vector that contains independent variables V_k(\pi_e; D) Y \in R^K
    Y = round_rewards @ episode_context / horizon_array

    # Calculate the estimate V_t(\pi_e; D) by using OLS estimator
    episode_at_evaluation = num_episodes_after_logged_data
    if phi_vector_func == exponential_vec:
        episode_at_evaluation = episode_at_evaluation / K
    try:
        # If there exists the inverse matrix, then use it
        V_t = (
            phi_scalar_func(
                episode_at_evaluation,
                num_features_for_Prognosticator,
                num_episodes_after_logged_data,
            )
            @ np.linalg.inv(Phi.T @ Phi)
            @ Phi.T
            @ Y
        )

    except np.linalg.LinAlgError:
        # If the matrix is singular, then we use pseudo inverse matrix instead of inverse matirx
        V_t = (
            phi_scalar_func(
                episode_at_evaluation,
                num_features_for_Prognosticator,
                num_episodes_after_logged_data,
            )
            @ np.linalg.pinv(Phi.T @ Phi)
            @ Phi.T
            @ Y
        )
    return V_t


def OPFV(
    time_at_eval: int,
    estimated_rewards_by_reg_model: np.ndarray,
    estimated_rewards_by_reg_model_at_eval: np.ndarray,
    reward: np.ndarray,
    action: np.ndarray,
    time: np.ndarray,
    pscore: np.ndarray,
    action_dist: np.ndarray,
    action_dist_at_eval: np.ndarray,
    phi_scalar_func: Callable = None,  # \phi_r(t)
    phi_scalar_func_for_context: Callable = None,  # \phi_x(t)
    flag_robust_to_non_stationary_context: bool = False,  # if OPFV is robust to non-stationary context or not
    flag_use_true_P_phi_t_for_reward: bool = False,
    P_phi_t_true_for_reward: float = None,  # P(\phi_r(t))
    flag_use_true_P_phi_t_for_context: bool = False,
    P_phi_t_true_for_context: float = None,  # P(\phi_x(t))
    flag_use_true_P_phi_t_for_context_reward: bool = False,
    P_phi_t_true_for_context_reward: float = None,  # P(\phi_{x, r}(t))
) -> np.ndarray:
    # The number of observation: n_rounds
    n = action.shape[0]

    ############ About \phi_r(t) ############
    # Vectorize the phi function
    phi_vector_func_for_reward = np.vectorize(phi_scalar_func)
    # (\phi_r(t_1), \cdots, \phi_r(t_n)) Time structure of the time we observed
    time_structure_for_reward = phi_vector_func_for_reward(time)
    # print(f"time_structure_for_reward = {time_structure_for_reward}")
    # \phi(t_e) Time structure of the time at the evaluation
    time_structure_at_eval_for_reward = phi_vector_func_for_reward(time_at_eval)
    # 1 if \ind{\phi_r(t_i) = \phi_r(t_e)}, and 0 otherwiese
    indicator_phi_r = (
        time_structure_for_reward == time_structure_at_eval_for_reward
    ).astype(int)

    if flag_use_true_P_phi_t_for_reward == False:
        # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
        est_P_phi_t_for_reward = indicator_phi_r.mean()
    else:
        est_P_phi_t_for_reward = P_phi_t_true_for_reward

    if est_P_phi_t_for_reward == 0:
        est_P_phi_t_for_reward = est_P_phi_t_for_reward_NEAR_ZERO

    indicator_phi_x = np.zeros(n)
    indicator_phi_x_r = np.zeros(n)

    est_P_phi_t_for_context = None
    est_P_phi_t_for_context_reward = None

    if flag_robust_to_non_stationary_context == True:
        ############ About \phi_x(t) ############
        # Vectorize the phi function
        phi_vector_func_for_context = np.vectorize(phi_scalar_func_for_context)
        # (\phi_r(t_1), \cdots, \phi_r(t_n)) Time structure of the time we observed
        time_structure_for_rcontext = phi_vector_func_for_context(time)
        # \phi(t_e) Time structure of the time at the evaluation
        time_structure_at_eval_for_context = phi_vector_func_for_context(time_at_eval)
        # 1 if \ind{\phi_r(t_i) = \phi_r(t_e)}, and 0 otherwiese
        indicator_phi_x = (
            time_structure_for_rcontext == time_structure_at_eval_for_context
        ).astype(int)

        if flag_use_true_P_phi_t_for_context == False:
            # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
            est_P_phi_t_for_context = indicator_phi_x.mean()
        else:
            est_P_phi_t_for_context = P_phi_t_true_for_context

        ############ About \phi_{x, r}(t) ############
        indicator_phi_x_r = indicator_phi_x * indicator_phi_r

        if flag_use_true_P_phi_t_for_context_reward == False:
            # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
            est_P_phi_t_for_context_reward = indicator_phi_x_r.mean()
        else:
            est_P_phi_t_for_context_reward = P_phi_t_true_for_context_reward

    # (w(x_1, t_1, a_1), \cdots, w(x_n, t_n, a_n)) importance wieght for each round
    iw = action_dist_at_eval[np.arange(n), action] / pscore

    # (\hat{f}(x_1, t_1, a_1), \cdots, \hat{f}(x_n, t_n, a_n))
    f_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action]

    # E_{\pi_e(a|x_i, t)}[\hat{f}(x_i, t, a)]
    estimated_rewards = np.average(
        estimated_rewards_by_reg_model_at_eval,
        weights=action_dist_at_eval,
        axis=1,
    )

    ########### OPFV only robust to non-stationary reward ###########
    if flag_robust_to_non_stationary_context == False:
        # OPFV calculate expected reward for each round
        estimate_round_rewards = (indicator_phi_r / est_P_phi_t_for_reward) * iw * (
            reward - f_hat_factual
        ) + estimated_rewards

    else:
        ########### OPFV robust to both non-stationary context and reward ###########
        estimate_round_rewards = (
            indicator_phi_x_r / est_P_phi_t_for_context_reward
        ) * iw * (reward - f_hat_factual) + (
            indicator_phi_x / est_P_phi_t_for_context
        ) * estimated_rewards

    return estimate_round_rewards


# --- NEW: estimators_time.py ---

def opfv_future_masked(
    *,
    reward: np.ndarray,                # (n,)
    action: np.ndarray,                # (n,)
    pscore: np.ndarray,                # (n,)
    time: np.ndarray,                  # (n,)
    time_to_phi: Callable[[int], int], # unix -> φ(t)（整数）
    phi_target: int,                   # 目標の時間構造（φ(t')）
    action_dist_at_eval: np.ndarray,   # (n, A)  将来 t' の評価方策（可用集合上で正規化済み）
    f_hat_factual: np.ndarray,         # (n,)    \hat f(x_i, t_i, a_i)
    f_hat_at_eval: np.ndarray,         # (n, A)  \hat f(x_i, t', a) for all a
    avail_mask_log: Optional[np.ndarray] = None,   # (n, A) bool（未使用でも受け取れるように）
    avail_mask_eval: Optional[np.ndarray] = None,  # (n, A) bool
    clip_min_pscore: float = 1e-12,
) -> float:
    """
    Support-aware OPFV:
      V̂ = 1/n Σ_i [  I{φ(t_i)=φ(t')} / P̂(φ(t'))  *  (π_e,t'(a_i|x_i)/pscore_i) * (r_i - f̂_i) ]
           + 1/n Σ_i [ Σ_a π_e,t'(a|x_i) * f̂(x_i,t',a) ].
    ここで π_e,t' は将来の可用集合で再正規化済みを想定。avail_mask_eval が与えられれば安全にゼロ化。
    返り値はスカラー（平均）。
    """
    reward = np.asarray(reward, dtype=float).ravel()
    action = np.asarray(action, dtype=int).ravel()
    pscore = np.asarray(pscore, dtype=float).ravel()
    time   = np.asarray(time, dtype=int).ravel()
    f_hat_factual = np.asarray(f_hat_factual, dtype=float).ravel()
    action_dist_at_eval = np.asarray(action_dist_at_eval, dtype=float)
    f_hat_at_eval = np.asarray(f_hat_at_eval, dtype=float)

    n = reward.shape[0]
    A = action_dist_at_eval.shape[1]
    assert action_dist_at_eval.shape == f_hat_at_eval.shape == (n, A), "shape mismatch"

    # φ(t) の一致指標と、その経験的確率（ゼロ割防止でクリップ）
    phi_vals = np.vectorize(time_to_phi)(time)
    phi_match = (phi_vals == int(phi_target)).astype(float)
    P_hat = max(phi_match.mean(), 1e-12)
    w_time = phi_match / P_hat  # (n,)

    # 将来可用マスク
    if avail_mask_eval is None:
        avail_mask_eval = np.ones_like(action_dist_at_eval, dtype=bool)
    else:
        avail_mask_eval = np.asarray(avail_mask_eval, dtype=bool)
        if avail_mask_eval.shape != (n, A):
            raise ValueError(f"`avail_mask_eval` must have shape {(n, A)}, got {avail_mask_eval.shape}")

    # 重要度重み（将来 π_e / 過去 pscore）。不可用アクションは 0 に。
    pscore_safe = np.clip(pscore, clip_min_pscore, None)
    row = np.arange(n)
    iw = action_dist_at_eval[row, action] / pscore_safe
    iw *= avail_mask_eval[row, action].astype(float)  # 将来不可用なら 0

    # 補正項（DR の残差部分）
    residual = reward - f_hat_factual
    term_residual = w_time * iw * residual  # (n,)

    # 期待値項（可用集合のみに限定して積を取る：π はすでに可用集合で正規化されている想定だが二重に安全化）
    dm_mat = action_dist_at_eval * f_hat_at_eval * avail_mask_eval.astype(float)
    term_dm = dm_mat.sum(axis=1)  # (n,)

    V_hat_i = term_residual + term_dm
    return float(V_hat_i.mean())