# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime

import numpy as np
from estimators_time import fourier_scalar, fourier_vec
from synthetic_time import (
    obtain_num_time_structure,
    unix_time_to_day_of_week,
)

n_seeds_for_time_eval_sampling = 20
n_seeds = 20
num_val = 1000
n_rounds_list = [500, 1000, 2000, 4000]
lambda_ratio_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
num_time_at_evaluation = 8
candidate_num_time_structure_list = range(2, 17, 2)
candidate_num_time_structure_list_for_OPFV = range(2, 17, 2)
reward_std = 1
n_seeds_all = 5
num_test = 10000

################### START time related hyperparameters ###################
num_overlaps = 1
t_now = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2022, month=12, day=31, hour=23, minute=59, second=59)
    )
)
t_oldest = int(
    datetime.datetime.timestamp(
        datetime.datetime(
            year=2023 - num_overlaps, month=1, day=1, hour=0, minute=0, second=0
        )
    )
)
time_at_evaluation = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)
    )
)
t_future = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0)
    )
)
num_cycles_in_evaluation_period: int = 1
################### END time related hyperparameters ###################


################### START time structure, OPFV, Prognosticator related hyperparameters ###################
num_time_structure_for_logged_data = 8
num_true_time_structure_for_OPFV_reward = num_time_structure_for_logged_data
num_episodes_for_Prognosticator = num_time_structure_for_logged_data * num_overlaps
phi_scalar_func_list = [
    fourier_scalar,
]
phi_vector_func_list = [
    fourier_vec,
]
num_features_for_Prognosticator = 3
num_features_for_Prognosticator_list = range(3, 8, 2)
flag_Prognosticator_optimality = True
################### END time structure, OPFV, Prognosticator related hyperparameters ###################


################### START non stationary context hyperparameters ###################
alpha_ratio_list = lambda_ratio_list
alpha_ratio_and_lambda_ratio_list = lambda_ratio_list
sample_non_stationary_context = False
time_structure_func_for_context = unix_time_to_day_of_week
num_time_structure_for_context = obtain_num_time_structure(unix_time_to_day_of_week)
p_1_coef = 3
p_2_coef = 1
################### END non stationary context hyperparameters ###################


################### START data class hyperparameters ###################
n_actions = 10
dim_context = 10
n_users = None
beta = 0.1
alpha_ratio = 0.5
lambda_ratio = 0.5
beta_list = [-0.4, -0.2, 0, 0.2, 0.4]
eps_list = [0.05, 0.1, 0.15, 0.2, 0.25]
flag_simple_reward = True
g_coef = 3
h_coef = 1
random_state = 12345
eps = 0.2
################### END data class hyperparameters ###################


################### START estimator flag hyperparameters ###################
flag_Prognosticator_with_multiple_feature_func = False
flag_include_DM = False
flag_calculate_data_driven_OPFV = True
################### END estimator flag hyperparameters ###################

markersize = 12


# ==================== START dynamic action space / embedding (追加) ====================

# --- データセット選択フラグ（呼び出し側で参照して使い分ける用。既存は無視されます）---
use_dynamic_action_dataset = True        # DynamicActionBanditWithTime を使うなら True
use_time_structure_for_embedding = True  # 埋め込み用の time_like を φ(t) にするなら True、Unix秒など生時刻なら False

# --- 行動可用性（A_t）設定 ---
ONE_DAY  = 24 * 60 * 60
ONE_WEEK = 7 * ONE_DAY

# 例：アクション0から順に1週間ごとに出現（消滅はしない）
action_birth_time = np.array(
    [t_oldest + i * ONE_WEEK for i in range(n_actions)], dtype=int
)
action_death_time = None  # 例：消滅なし。消滅させる場合は np.array([...], dtype=int)

# 例：関数で可用マスクを作る場合（DynamicActionBanditWithTime に渡す）
def availability_func_weekly_stair(times: np.ndarray, A: int = n_actions,
                                   base: int = t_oldest, step: int = ONE_WEEK) -> np.ndarray:
    times = np.asarray(times)
    mask = np.zeros((len(times), A), dtype=bool)
    for a in range(A):
        birth = base + a * step
        mask[:, a] = times >= birth
    return mask

# availability を “時々” 切り替えたい場合のスイッチ（呼び出し側で if で使う）
use_availability_func = False  # True にすると availability_func_weekly_stair を優先し、birth/death は無視

# --- 時間依存アクション埋め込み設定 ---
embed_dim = 16
action_embedding_type = "default"   # 実装切替用のキー（"default", "age", "fourier+id" など）
embedding_random_state = 12345

# 例：埋め込み関数の切替を呼び出し側で行うためのパラメータ（必要なら追加）
# use_time_structure_for_embedding=True のときは φ(t) を渡す想定、False のときは Unix 秒などを渡す想定。

# --- 将来方策・推定の “可用マスク対応” スイッチ（呼び出し側で使う） ---
use_masked_policy = True   # gen_eps_greedy_masked を使う
use_masked_opfv   = True   # opfv_future_masked を使う

# --- ランナーの選択（呼び出し側で if で分岐） ---
use_run_ope_dynamic = True   # run_ope_dynamic.py の私用ランナーを使う

# --- 既存 eps と整合を取るための別名（読みやすさのため） ---
eps_eval = eps

# ==================== END dynamic action space / embedding (追加) ====================
