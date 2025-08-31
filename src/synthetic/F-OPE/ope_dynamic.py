# --- NEW: ope.py に追記する私用ランナー ---
from typing import Callable
import numpy as np
import conf
from sklearn.ensemble import RandomForestRegressor

from obp.ope import OffPolicyEvaluation, RegressionModel
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import DoublyRobust as DR
from obp.ope import DirectMethod as DM

from synthetic_time import unix_time_to_time_structure_n_tree
from utils import calculate_hat_f_train_and_eval

# 既存（ベースライン）も併用
from estimators_time import OPFV, Prognosticator, fourier_scalar, fourier_vec

# ★ ここが新規：可用アクション対応の方策生成・推定器
from policy import gen_eps_greedy_masked
from estimators_time import opfv_future_masked


def run_ope_masked(
    dataset,
    round,
    time_at_evaluation,
    estimated_policy_value_list,
    val_bandit_data,
    action_dist_val,
    num_episodes_for_Prognosticator,
    num_time_structure_from_t_now_to_time_at_evaluation: int,
    num_true_time_structure_for_OPFV_reward: int,
    num_true_time_structure_for_OPFV_for_context: int = None,
    eps=conf.eps,
    flag_calulate_robust_OPFV: bool = False,
    flag_Prognosticator_optimality: bool = True,
    num_features_for_Prognosticator_list: np.ndarray = np.ndarray([1, 2, 3]),
    true_policy_value: float = None,
    flag_include_DM: bool = True,
    flag_calculate_data_driven_OPFV: bool = False,
    candidate_num_time_structure_list: Callable = None,
):
    """
    既存 run_ope をベースに、将来方策と推定を「可用アクション集合」上で行う私用版。
    既存コード・APIは変更しない。
    """

    # ---- φ_r(t), φ_x(t) のスカラー関数（既存と同じ構成） ----
    def phi_scalar_func_for_OPFV(unix_time):
        return unix_time_to_time_structure_n_tree(
            unix_time, num_true_time_structure_for_OPFV_reward
        )

    if flag_calulate_robust_OPFV:

        def phi_scalar_func_for_OPFV_for_context(unix_time):
            return unix_time_to_time_structure_n_tree(
                unix_time, num_true_time_structure_for_OPFV_for_context
            )

    def finest_time_structure(unix_time):
        return unix_time_to_time_structure_n_tree(
            unix_time, candidate_num_time_structure_list[-1]
        )

    # ============================================================
    # 1) ベースライン: DM / IPS / DR（既存そのまま）
    # ============================================================
    reg_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        ),
    )
    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],
        action=val_bandit_data["action"],
        reward=val_bandit_data["reward"],
        n_folds=2,
        random_state=12345 + round,
    )

    if flag_include_DM:
        ope_estimators = [IPS(estimator_name="IPS"), DR(estimator_name="DR"), DM(estimator_name="DM")]
    else:
        ope_estimators = [IPS(estimator_name="IPS"), DR(estimator_name="DR")]

    ope = OffPolicyEvaluation(bandit_feedback=val_bandit_data, ope_estimators=ope_estimators)
    estimated_policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val[:, :, np.newaxis],
        estimated_rewards_by_reg_model=estimated_rewards,
        pi_b=val_bandit_data["pi_b"],
    )

    # ============================================================
    # 2) Prognosticator（既存と同じ。最適化モードも踏襲）
    # ============================================================
    sorted_indices = np.argsort(val_bandit_data["time"])
    action_sorted_Prog = val_bandit_data["action"][sorted_indices]
    reward_sorted_Prog = val_bandit_data["reward"][sorted_indices]
    pscore_sorted_Prog = val_bandit_data["pscore"][sorted_indices]
    action_dist_val_sorted_Prog = action_dist_val[sorted_indices]

    if flag_Prognosticator_optimality:
        estimated_mse_list = []
        candidate_estimated_policy_value_list = []
        for i in range(len(conf.phi_scalar_func_list)):
            for num_features_for_Prognosticator in num_features_for_Prognosticator_list:
                cand = Prognosticator(
                    num_episodes=num_episodes_for_Prognosticator,
                    phi_scalar_func=conf.phi_scalar_func_list[i],
                    phi_vector_func=conf.phi_vector_func_list[i],
                    reward=reward_sorted_Prog,
                    action=action_sorted_Prog,
                    pscore=pscore_sorted_Prog,
                    action_dist=action_dist_val_sorted_Prog,
                    num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
                    num_features_for_Prognosticator=num_features_for_Prognosticator,
                )
                candidate_estimated_policy_value_list.append(cand)
                est_mse = (cand - true_policy_value) ** 2
                estimated_mse_list.append(est_mse)
        min_index = int(np.argmin(estimated_mse_list))
        estimated_policy_values["Prognosticator"] = candidate_estimated_policy_value_list[min_index]
    else:
        estimated_policy_values["Prognosticator"] = Prognosticator(
            num_episodes=num_episodes_for_Prognosticator,
            phi_scalar_func=fourier_scalar,
            phi_vector_func=fourier_vec,
            reward=reward_sorted_Prog,
            action=action_sorted_Prog,
            pscore=pscore_sorted_Prog,
            action_dist=action_dist_val_sorted_Prog,
            num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
            num_features_for_Prognosticator=conf.num_features_for_Prognosticator,
        )

    # ============================================================
    # 3) Future 方策＆OPFV（可用アクション対応）
    # ============================================================
    n = val_bandit_data["action"].shape[0]
    time_at_eval_vec = np.full(n, time_at_evaluation, dtype=int)

    # 3-1) \hat f(x,t,a) と \hat f(x,t',a) を既存の関数で作成
    hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
        phi_scalar_func_for_OPFV, val_bandit_data, dataset, time_at_eval_vec, round
    )

    # 3-2) 将来の真の期待報酬 q(x,t',a) と 可用マスク A_{t'}
    _, _, q_eval_true = dataset.synthesize_expected_reward(
        contexts=val_bandit_data["context"], times=time_at_eval_vec
    )
    if hasattr(dataset, "_availability"):
        avail_eval = dataset._availability(time_at_eval_vec).astype(bool)
    else:
        avail_eval = np.ones_like(q_eval_true, dtype=bool)

    # 3-3) 将来の評価方策 π_e(·|x,t') を“可用集合で正規化”して生成
    action_dist_val_at_eval = gen_eps_greedy_masked(
        expected_reward=q_eval_true,
        eps=eps,
        is_optimal=True,
        available_actions=avail_eval,
    )

    # 3-4) ログ側の可用マスク（DynamicAction… のときだけ存在）
    avail_log = val_bandit_data.get(
        "available_actions",
        np.ones_like(q_eval_true, dtype=bool),
    )

    # 3-5) OPFV（support-aware 版）
    f_hat_factual = hat_f_x_t_a[np.arange(n), val_bandit_data["action"]]
    estimated_policy_values["OPFV-masked"] = opfv_future_masked(
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        pscore=val_bandit_data["pscore"],
        time=val_bandit_data["time"],
        time_to_phi=phi_scalar_func_for_OPFV,
        phi_target=phi_scalar_func_for_OPFV(time_at_evaluation),
        action_dist_at_eval=action_dist_val_at_eval,
        f_hat_factual=f_hat_factual,
        f_hat_at_eval=hat_f_x_t_a_at_eval,
        avail_mask_log=avail_log,
        avail_mask_eval=avail_eval,
        clip_min_pscore=1e-12,
    )

    # 3-6) 参考: 既存 OPFV もベースラインとして残す（将来方策は可用集合で作ったものを使用）
    estimated_policy_values["OPFV"] = OPFV(
        phi_scalar_func=phi_scalar_func_for_OPFV,
        phi_scalar_func_for_context=None,
        time_at_eval=time_at_evaluation,
        estimated_rewards_by_reg_model=hat_f_x_t_a,
        estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        time=val_bandit_data["time"],
        pscore=val_bandit_data["pscore"],
        action_dist=action_dist_val,
        action_dist_at_eval=action_dist_val_at_eval,
        flag_robust_to_non_stationary_context=False,
        flag_use_true_P_phi_t_for_reward=False,
        P_phi_t_true_for_reward=None,
        flag_use_true_P_phi_t_for_context=False,
        P_phi_t_true_for_context=None,
        flag_use_true_P_phi_t_for_context_reward=False,
        P_phi_t_true_for_context_reward=None,
    ).mean()

    # 3-7) （任意）robust OPFV もベースラインとして残す
    if flag_calulate_robust_OPFV:
        estimated_policy_values["robust OPFV"] = OPFV(
            phi_scalar_func=phi_scalar_func_for_OPFV,
            phi_scalar_func_for_context=phi_scalar_func_for_OPFV_for_context,
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=True,
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,
        ).mean()

    # 3-8) （任意）data-driven OPFV（φ候補の中から選択）も、将来方策は可用集合で生成
    if flag_calculate_data_driven_OPFV:
        hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
            finest_time_structure, val_bandit_data, dataset, time_at_eval_vec, round
        )
        estimated_value_with_finest = OPFV(
            phi_scalar_func=finest_time_structure,
            phi_scalar_func_for_context=None,
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=False,
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,
        ).mean()

        estimated_mse_list = []
        candidate_estimated_value_list = []
        for candidate_num_time_structure in candidate_num_time_structure_list:

            def candidate_phi_scalar_func(unix_time):
                return unix_time_to_time_structure_n_tree(
                    unix_time, candidate_num_time_structure
                )

            hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
                candidate_phi_scalar_func, val_bandit_data, dataset, time_at_eval_vec, round
            )

            candidate_value_round_rewards = OPFV(
                phi_scalar_func=candidate_phi_scalar_func,
                phi_scalar_func_for_context=None,
                time_at_eval=time_at_evaluation,
                estimated_rewards_by_reg_model=hat_f_x_t_a,
                estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
                reward=val_bandit_data["reward"],
                action=val_bandit_data["action"],
                time=val_bandit_data["time"],
                pscore=val_bandit_data["pscore"],
                action_dist=action_dist_val,
                action_dist_at_eval=action_dist_val_at_eval,
                flag_robust_to_non_stationary_context=False,
                flag_use_true_P_phi_t_for_reward=False,
                P_phi_t_true_for_reward=None,
                flag_use_true_P_phi_t_for_context=False,
                P_phi_t_true_for_context=None,
                flag_use_true_P_phi_t_for_context_reward=False,
                P_phi_t_true_for_context_reward=None,
            )
            candidate_estimated_value_list.append(candidate_value_round_rewards.mean())

            est_squared_bias = (candidate_value_round_rewards.mean() - estimated_value_with_finest) ** 2
            est_var = np.var(candidate_value_round_rewards, ddof=1) / len(candidate_value_round_rewards)
            estimated_mse_list.append(est_squared_bias + est_var)

        min_index = int(np.argmin(estimated_mse_list))
        estimated_policy_values["data-driven OPFV"] = candidate_estimated_value_list[min_index]

    # ============================================================
    # 4) 真値（与えられていなければ、将来の q と可用マスクで算出）
    # ============================================================
    if true_policy_value is None:
        # 可用集合で再正規化した π_e を使って真値を算出（DynamicAction… はマスク対応の真値関数あり）
        if hasattr(dataset, "calc_ground_truth_policy_value"):
            try:
                true_policy_value = dataset.calc_ground_truth_policy_value(
                    expected_reward=q_eval_true,
                    action_dist=action_dist_val_at_eval,
                    available_actions=avail_eval,
                )
            except TypeError:
                true_policy_value = np.average(q_eval_true, weights=action_dist_val_at_eval, axis=1).mean()
        else:
            true_policy_value = np.average(q_eval_true, weights=action_dist_val_at_eval, axis=1).mean()

    estimated_policy_values["V_t"] = true_policy_value

    # 収集
    estimated_policy_value_list.append(estimated_policy_values)
