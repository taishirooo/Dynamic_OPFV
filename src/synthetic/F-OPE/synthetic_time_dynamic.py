# synthetic_time_dynamic.py
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np
import datetime
from scipy.stats import truncnorm
from scipy.special import softmax

from synthetic_time import (
    BaseBanditDataset,  # 既存のベースと同じ想定
    normalize_time,
    unix_time_to_time_structure_n_tree,
)
# ↑ BaseBanditDataset, RewardType が synthetic_time.py にある前提
from synthetic_time import RewardType  # 必要なら import 調整

Array = np.ndarray

@dataclass
class DynamicActionBanditWithTime(BaseBanditDataset):
    # ===== 継承互換の基本パラメタ =====
    n_actions: int
    dim_context: int = 1
    n_users: int = None

    t_oldest: int = int(datetime.datetime.timestamp(datetime.datetime(2022,1,1)))
    t_now:    int = int(datetime.datetime.timestamp(datetime.datetime(2022,6,1)))
    t_future: int = int(datetime.datetime.timestamp(datetime.datetime(2023,1,1)))

    # 報酬側と文脈側の時間構造（φ_r, φ_x）
    num_time_structure: int = 7
    num_time_structure_for_context: int = 7

    # q = λ g + (1-λ) h,   p(x|t) = α p1 + (1-α) p2
    lambda_ratio: float = 0.95
    alpha_ratio:  float = 0.95

    reward_type: str = RewardType.CONTINUOUS.value
    reward_std: float = 1.0
    flag_simple_reward: bool = True
    sample_non_stationary_context: bool = False

    # ====== 行動空間の非定常用 追加パラメタ ======
    # 行動の出現/消滅を与える：birth/death があれば availability_func は不要
    action_birth_time: Optional[Array] = None   # shape (n_actions,)
    action_death_time: Optional[Array] = None   # shape (n_actions,) or None
    # もしくは直接マスクを生成する関数：times -> Bool[n_rounds, n_actions]
    availability_func: Optional[Callable[[Array], Array]] = None

    # 時間依存の行動埋め込み z_a(t)=f_theta(a, φ(t))
    # times: (n_rounds,), action_ids: (n_actions,) -> (n_rounds, n_actions, d_z)
    action_embedding_func: Optional[Callable[[Array, Array], Array]] = None
    embed_dim: int = 16  # z の次元（上の関数があれば自動で上書きされる）

    # ログ方策：未指定なら softmax(beta * q) を可用集合で再正規化
    beta: float = 1.0
    behavior_policy_function: Optional[Callable[[Array, Array], Array]] = None

    # 既知/新規アクションの扱い（評価時）
    eval_mode: str = "ignore_new"  # or "extrapolate"

    random_state: int = 12345
    dataset_name: str = "dynamic_action_bandit_with_time"

    # ====== 内部パラメタ（サンプリング係数）======
    # 埋め込み版の g/h で使う係数（形は最小サンプル）
    def __post_init__(self):
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = np.random.RandomState(self.random_state)

        # 既存に合わせて reward 型設定
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min, self.reward_max = 0.0, 1e10

        # φ_r, φ_x のスカラ関数
        self.time_structure_func = lambda t: unix_time_to_time_structure_n_tree(t, self.num_time_structure)
        self.time_structure_func_for_context = lambda t: unix_time_to_time_structure_n_tree(t, self.num_time_structure_for_context)

        # アクションID配列
        self._action_ids = np.arange(self.n_actions)

        # 埋め込み関数が無ければ簡易な線形埋め込みを自前で用意
        if self.action_embedding_func is None:
            W_a = self.random_.normal(size=(self.n_actions, self.embed_dim))
            W_phi = self.random_.normal(size=(self.num_time_structure, self.embed_dim))
            def default_embed(times: Array, action_ids: Array) -> Array:
                # φ(t) one-hot → 埋め込み, 行ごとに足し合わせ（位置エンコーディング風）
                phi = np.vectorize(self.time_structure_func)(times)  # (n_rounds,)
                Zt = W_a[action_ids][None, :, :] + W_phi[phi][:, None, :]
                return Zt  # (n_rounds, n_actions, d_z)
            self.action_embedding_func = default_embed
        else:
            # 推定しやすいよう embed_dim を関数の返りで上書き
            tmp = self.action_embedding_func(np.array([self.t_oldest]), self._action_ids)
            self.embed_dim = tmp.shape[-1]

        # g/h の係数（最小構成）：文脈・時間・埋め込みの一次・双線形
        self.psi_x = self.random_.uniform(-1, 1, size=self.dim_context)             # g: x
        self.psi_phi = self.random_.uniform(-1, 1, size=self.num_time_structure)    # g: φ(t)
        self.U_phi_z = self.random_.normal(size=(self.num_time_structure, self.embed_dim))  # g: φ × z
        self.B_x_z  = self.random_.normal(size=(self.dim_context, self.embed_dim))          # g: x × z

        self.theta_x = self.random_.uniform(-0.5, 0.5, size=self.dim_context)       # h: x
        self.v_finer = self.random_.uniform(-0.5, 0.5, size=7)                      # h: 曜日
        self.C_finer_z = self.random_.normal(size=(7, self.embed_dim))              # h: 曜日 × z
        self.D_x_z  = self.random_.normal(size=(self.dim_context, self.embed_dim))  # h: x × z

        # 文脈分布の非定常（任意）
        self.gamma = self.random_.uniform(-1, 1, size=self.num_time_structure_for_context)
        self.delta = self.random_.uniform(-1, 1, size=5)

    # === 可用アクションマスク A(t) を作る ===
    def _availability(self, times: Array) -> Array:
        n = times.shape[0]
        if self.availability_func is not None:
            mask = self.availability_func(times)  # (n, A) bool
            return mask.astype(bool)
        # birth/death から作る簡易マスク
        birth = self.action_birth_time if self.action_birth_time is not None else np.full(self.n_actions, self.t_oldest)
        death = self.action_death_time if self.action_death_time is not None else np.full(self.n_actions, np.inf)
        mask = (times[:, None] >= birth[None, :]) & (times[:, None] < death[None, :])
        return mask

    # === 期待報酬 q の合成（埋め込み版の g/h）===
    def synthesize_expected_reward(self, contexts: Array, times: Array) -> Tuple[Array, Array, Array]:
        n = contexts.shape[0]
        # φ_r(t) / 曜日 one-hot
        phi_vec = np.vectorize(self.time_structure_func)(times)                         # (n,)
        Phi = np.eye(self.num_time_structure)[phi_vec]                                  # (n, K_r)
        dow = np.vectorize(lambda ts: datetime.datetime.utcfromtimestamp(int(ts)).weekday())(times)
        Finer = np.eye(7)[dow]                                                          # (n, 7)

        # 行動埋め込み z_a(t)
        Z = self.action_embedding_func(times, self._action_ids)                         # (n, A, d_z)

        # ===== g(x, φ, a) =====
        g1 = contexts @ self.psi_x                                                      # (n,)
        g2 = Phi @ self.psi_phi                                                         # (n,)
        g3 = (Phi @ self.U_phi_z)[:, None, :] * Z                                       # (n, A, d_z)
        g3 = g3.sum(axis=-1)                                                            # (n, A)
        g4 = (contexts @ self.B_x_z)[:, None, :] * Z                                    # (n, A, d_z)
        g4 = g4.sum(axis=-1)                                                            # (n, A)
        g_x_phi_t_a = g1[:, None] + g2[:, None] + g3 + g4                               # (n, A)

        # ===== h(x, t, a) =====（細粒度時間の寄与＋連続t残差の箱）
        h1 = contexts @ self.theta_x                                                    # (n,)
        h2 = Finer @ self.v_finer                                                       # (n,)
        h3 = (Finer @ self.C_finer_z)[:, None, :] * Z                                   # (n, A, d_z)
        h3 = h3.sum(axis=-1)                                                            # (n, A)
        h4 = (contexts @ self.D_x_z)[:, None, :] * Z                                    # (n, A, d_z)
        h4 = h4.sum(axis=-1)                                                            # (n, A)
        h_x_t_a = h1[:, None] + h2[:, None] + h3 + h4                                   # (n, A)

        q = self.lambda_ratio * g_x_phi_t_a + (1 - self.lambda_ratio) * h_x_t_a         # (n, A)

        return g_x_phi_t_a, h_x_t_a, q

    # === ログ生成 ===
    def obtain_batch_bandit_feedback(
        self, n_rounds: int, evaluation_mode: bool=False, time_at_evaluation: int=0, random_state_for_sampling: int=None
    ):
        rng = np.random.RandomState((random_state_for_sampling or 0) + self.random_state)

        # 時刻
        if not evaluation_mode:
            times = rng.uniform(self.t_oldest, self.t_now, size=n_rounds).astype(int)
            times.sort()
        else:
            times = np.full(n_rounds, time_at_evaluation)

        # 文脈
        if not self.sample_non_stationary_context:
            contexts = rng.normal(size=(n_rounds, self.dim_context))
        else:
            norm_t = normalize_time(times, self.t_oldest, self.t_future)
            phi_x = np.vectorize(self.time_structure_func_for_context)(times)
            Phi_x = np.eye(self.num_time_structure_for_context)[phi_x]
            mu1 = Phi_x @ self.gamma
            mu2 = self.delta[0] * norm_t
            contexts = self.alpha_ratio * rng.normal(loc=mu1[:, None], scale=1, size=(n_rounds, self.dim_context)) \
                     + (1 - self.alpha_ratio) * rng.normal(loc=mu2[:, None], scale=1, size=(n_rounds, self.dim_context))

        # 期待報酬
        g, h, expected_reward = self.synthesize_expected_reward(contexts, times)

        # 連続報酬なら切断正規の平均補正（既存と同様の処理）
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            a = (0.0 - expected_reward) / self.reward_std
            b = (1e10 - expected_reward) / self.reward_std
            expected_reward = truncnorm.stats(a=a, b=b, loc=expected_reward, scale=self.reward_std, moments="m")

        # 可用アクションマスク
        avail_mask = self._availability(times)  # (n, A) bool

        # 挙動方策
        if self.behavior_policy_function is None:
            logits = expected_reward
        else:
            # 互換性のため action_context を “固定” の代替としてゼロ行列を渡してもよい
            logits = self.behavior_policy_function(context=contexts, action_context=np.zeros((self.n_actions, 1)), random_state=self.random_state)
        pi_b = softmax(self.beta * logits, axis=1)
        pi_b = pi_b * avail_mask
        pi_b = pi_b / np.clip(pi_b.sum(axis=1, keepdims=True), 1e-12, None)

        # 行動サンプル
        # サンプリング（高速化は任意）
        cum = np.cumsum(pi_b, axis=1)
        u = rng.rand(n_rounds)[:, None]
        actions = (u > cum).sum(axis=1).astype(int)

        # 報酬サンプル
        factual_mean = expected_reward[np.arange(n_rounds), actions]
        if RewardType(self.reward_type) == RewardType.BINARY:
            rewards = rng.binomial(n=1, p=factual_mean)
        else:
            a = (0.0 - factual_mean) / self.reward_std
            b = (1e10 - factual_mean) / self.reward_std
            rewards = truncnorm.rvs(a=a, b=b, loc=factual_mean, scale=self.reward_std, random_state=rng)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            time=times,
            # 互換性のためキーは残す（実体は使われなくてもOK）
            action_context=np.zeros((self.n_actions, 1)),
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=expected_reward,
            g_x_phi_t_a=g,
            h_x_t_a=h,
            pi_b=pi_b[:, :, None],
            pscore=pi_b[np.arange(n_rounds), actions],
            # 新規：時刻ごとの可用アクション
            available_actions=avail_mask.astype(int),
        )

    # 真値の方策価値（可用マスク込みで重み付け）
    def calc_ground_truth_policy_value(self, expected_reward: Array, action_dist: Array, available_actions: Optional[Array]=None) -> float:
        if available_actions is not None:
            masked = action_dist * available_actions
            masked = masked / np.clip(masked.sum(axis=1, keepdims=True), 1e-12, None)
            return np.average(expected_reward, weights=masked, axis=1).mean()
        return np.average(expected_reward, weights=action_dist, axis=1).mean()
