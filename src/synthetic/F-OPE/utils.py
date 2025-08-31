# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime

import conf
import numpy as np
from regression_model_time import RegressionModelTimeStructure
from sklearn.ensemble import RandomForestRegressor

fromtimestamp_vec = np.vectorize(datetime.datetime.fromtimestamp)


def show_hyperparameters(
    time_at_evaluation_start: int = None,
    time_at_evaluation_end: int = None,
    flag_show_time_at_evaluation: bool = True,
    time_at_evaluation_list: callable = None,
):
    print(f"################# START hyperparameters #################")

    print(f"### About Seeds and Number of Samples ###")
    print(f"number of seeds = {conf.n_seeds}")
    print(
        f"number of seeds for time at evaluation = {conf.n_seeds_for_time_eval_sampling}"
    )
    print(f"number of training samples (n) = {conf.num_val}")
    print(f"number of test samples = {conf.num_test}\n")

    print(f"### About Time Structure ###")
    print(
        f"number of true time structures for reward (|C_r|) = {conf.num_time_structure_for_logged_data}"
    )
    print(f"strength of time structure for reward (lambda) = {conf.lambda_ratio}\n")

    print(f"### About Prognosticator ###")
    print(f"list of time features for Prognosticator = {conf.phi_scalar_func_list}")
    print(
        f"optimality of the data driven feature selection for Prognosticator = {conf.flag_Prognosticator_optimality}"
    )
    print(
        f"number of time features for Prognosticator = {conf.num_features_for_Prognosticator}"
    )
    print(
        f"list of the numbers of time features for Prognosticator = {conf.num_features_for_Prognosticator_list}\n"
    )

    print(f"### About Logged Data Collection Period and Evaluation Period ###")
    print(
        f"time when we start collecting the logged data = {datetime.datetime.fromtimestamp(conf.t_oldest)}"
    )
    print(
        f"time when we finish collecting the logged data = {datetime.datetime.fromtimestamp(conf.t_now)}"
    )
    if flag_show_time_at_evaluation == True:
        print(
            f"time when we start evaluating a target policy = {datetime.datetime.fromtimestamp(time_at_evaluation_start)}"
        )
        print(
            f"time when we finish evaluating a target policy = {datetime.datetime.fromtimestamp(time_at_evaluation_end)}"
        )
    print(f"future time = {datetime.datetime.fromtimestamp(conf.t_future)}\n")

    print(f"### About Parameters for Data Generating Process ###")
    print(f"number of actions (|A|) = {conf.n_actions}")
    print(f"dimension of context (d_x) = {conf.dim_context}")
    print(f"number of users = {conf.n_users}")
    print(f"behavior policy optimality (beta) = {conf.beta}")
    print(f"target policy optimality (epsilon) = {conf.eps}\n")

    print(f"### About Varying Parameters ###")
    print(f"list of the numbers of training samples (n) = {conf.n_rounds_list}")
    print(
        f"list of the strengths of time structure for reward (lambda) = {conf.lambda_ratio_list}"
    )
    print(
        f"list of the numbers of candidate time structures for reward = {conf.candidate_num_time_structure_list}"
    )
    if flag_show_time_at_evaluation == False:
        print(
            f"list of the time at evaluation = {fromtimestamp_vec(time_at_evaluation_list)}"
        )

    print(f"################# END hyperparameters #################\n\n")


def calculate_hat_f_train_and_eval(
    phi_scalar_func_for_OPFV, val_bandit_data, dataset, time_at_eval_vec, round
):
    # Vectorize the phi function
    phi_vector_func = np.vectorize(phi_scalar_func_for_OPFV)
    # (\phi(t_1), \cdots, \phi(t_n)) Time structure of the time we observed
    time_structure = phi_vector_func(val_bandit_data["time"])

    # Machine learning model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_time_structure = RegressionModelTimeStructure(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=val_bandit_data["action_context"],
        # A machine learning model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        ),
    )

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  g_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    hat_g_x_phi_t_a = reg_model_time_structure.fit_predict(
        context=val_bandit_data["context"],  # context; x
        time_structure=time_structure,  # time structure: phi(t)
        action=val_bandit_data["action"],  # action; a
        reward=val_bandit_data["reward"],  # reward; r
        n_folds=2,
        random_state=12345 + round,
    )

    hat_g_x_phi_t_a = np.squeeze(hat_g_x_phi_t_a, axis=2)

    hat_f_x_t_a = hat_g_x_phi_t_a

    hat_g_x_phi_t_a_at_eval = reg_model_time_structure.predict(
        context=val_bandit_data["context"],
        time_structure=phi_vector_func(time_at_eval_vec),
    )
    hat_g_x_phi_t_a_at_eval = np.squeeze(hat_g_x_phi_t_a_at_eval, axis=2)
    hat_g_x_phi_t_a_at_eval

    hat_f_x_t_a_at_eval = hat_g_x_phi_t_a_at_eval

    return hat_f_x_t_a, hat_f_x_t_a_at_eval


# ==== utils.py に追記（新規関数だけ） ======================================
from typing import Callable, Optional
import numpy as np
from regression_model_time import (
    RegressionModelTimeWithEmbedding,
    RegressionModelTimeStructureWithEmbedding,
)
from policy import gen_eps_greedy_masked
from sklearn.ensemble import RandomForestRegressor


def time_to_phi_vec(phi_scalar_func: Callable[[int], int], times: np.ndarray) -> np.ndarray:
    """φ: int(unix_time) -> int(time_structure) をベクトル化"""
    return np.vectorize(phi_scalar_func)(times)


def get_availability_mask(dataset, times: np.ndarray, fallback_shape: tuple) -> np.ndarray:
    """データセットが _availability を持っていればそれを使い、無ければ全 True を返す"""
    avail = getattr(dataset, "_availability", None)
    if callable(avail):
        return avail(times).astype(bool)
    return np.ones(fallback_shape, dtype=bool)


def calculate_hat_f_train_and_eval_with_embedding(
    phi_scalar_func_for_OPFV: Callable[[int], int],
    val_bandit_data: dict,
    dataset,
    time_at_eval_vec: np.ndarray,
    round: int,
    action_embedding_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    use_time_structure: bool = True,
    base_model: Optional[RandomForestRegressor] = None,
):
    """
    埋め込み対応の回帰で、学習時点の \hat f(x,t,a) と 将来 \hat f(x,t',a) を返す。
    返り値:
      hat_f_x_t_a:          (n, n_actions)
      hat_f_x_t_a_at_eval:  (n, n_actions)
    """
    if base_model is None:
        base_model = RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        )

    if use_time_structure:
        # φ(t) を使う版（RegressionModelTimeStructure と同じIF）
        time_structure = time_to_phi_vec(phi_scalar_func_for_OPFV, val_bandit_data["time"])
        reg = RegressionModelTimeStructureWithEmbedding(
            base_model=base_model,
            n_actions=dataset.n_actions,
            action_embedding_func=action_embedding_func,
        )
        hat_g = reg.fit_predict(
            context=val_bandit_data["context"],
            time_structure=time_structure,
            action=val_bandit_data["action"],
            reward=val_bandit_data["reward"],
            n_folds=2,
            random_state=12345 + round,
        )
        hat_f_x_t_a = np.squeeze(hat_g, axis=2)

        hat_g_eval = reg.predict(
            context=val_bandit_data["context"],
            time_structure=time_to_phi_vec(phi_scalar_func_for_OPFV, time_at_eval_vec),
        )
        hat_f_x_t_a_at_eval = np.squeeze(hat_g_eval, axis=2)
    else:
        # 生の time（Unix秒など）を使う版（RegressionModelTime と同じIF）
        reg = RegressionModelTimeWithEmbedding(
            base_model=base_model,
            n_actions=dataset.n_actions,
            action_embedding_func=action_embedding_func,
        )
        hat_q = reg.fit_predict(
            context=val_bandit_data["context"],
            time=val_bandit_data["time"],
            action=val_bandit_data["action"],
            reward=val_bandit_data["reward"],
            n_folds=2,
            random_state=12345 + round,
        )
        hat_f_x_t_a = np.squeeze(hat_q, axis=2)

        hat_q_eval = reg.predict(
            context=val_bandit_data["context"],
            time=time_at_eval_vec,
        )
        hat_f_x_t_a_at_eval = np.squeeze(hat_q_eval, axis=2)

    return hat_f_x_t_a, hat_f_x_t_a_at_eval


def build_eval_policy_with_mask(
    *,
    dataset,
    contexts_eval: np.ndarray,   # (n_eval, d)
    times_eval: np.ndarray,      # (n_eval,)
    eps_eval: float = conf.eps,
):
    """
    将来 t' の真の期待報酬 q、可用マスク、可用集合上の ε-greedy 方策 π_e を作って返す。
    返り値:
      q_eval:   (n_eval, A)
      pi_e:     (n_eval, A)
      avail:    (n_eval, A) bool
    """
    # 真の将来期待報酬 q(x,t',a)
    _, _, q_eval = dataset.synthesize_expected_reward(contexts_eval, times_eval)

    # 可用マスク（無ければ全 True）
    avail = get_availability_mask(dataset, times_eval, fallback_shape=q_eval.shape)

    # 可用集合上の ε-greedy
    pi_e = gen_eps_greedy_masked(
        expected_reward=q_eval, eps=eps_eval, is_optimal=True, available_actions=avail
    )
    return q_eval, pi_e, avail
