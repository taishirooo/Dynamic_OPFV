# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state, check_scalar


@dataclass
class RegressionModelTime(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                time=time[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray, time: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    time=time,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                time=time,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context, time=time)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                time=time[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(
                context=context[test_idx], time=time[train_idx]
            )
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        return np.c_[context, time, action_context[action]]


@dataclass
class RegressionModelTimeStructure(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                time_structure=time_structure[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray, time_structure: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    time_structure=time_structure,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                time_structure=time_structure,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context, time_structure=time_structure)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                time_structure=time_structure[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(
                context=context[test_idx], time_structure=time_structure[test_idx]
            )
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        return np.c_[context, time_structure, action_context[action]]



# === NEW: regression_model_time.py に追記（既存は変更しない） ===
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state, check_scalar

# ------------------------------------------------------------
# 1) time（連続/整数の時刻）を受け取る埋め込み版
# ------------------------------------------------------------
@dataclass
class RegressionModelTimeWithEmbedding(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    action_embedding_func: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (times, action_ids) -> (n, A, d_z)
    len_list: int = 1
    fitting_method: str = "normal"  # "normal" | "iw" | "mrdr"

    def __post_init__(self) -> None:
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if self.fitting_method not in ["normal", "iw", "mrdr"]:
            raise ValueError("`fitting_method` must be one of 'normal', 'iw', or 'mrdr'")
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError("`base_model` must be a sklearn BaseEstimator")
        self.base_model_list = [clone(self.base_model) for _ in np.arange(self.len_list)]
        self._action_ids = np.arange(self.n_actions)

    # ---- 内部前処理：埋め込み生成 → 指定 action のベクトルを抜き出す ----
    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        # Z: (n, A, d_z)
        Z = self.action_embedding_func(time, self._action_ids)
        # row-wise に選択された action の埋め込みを取得 (n, d_z)
        z_a = Z[np.arange(len(action)), action]
        # 元実装と同様に time もそのまま特徴へ連結
        time_col = time.reshape(-1, 1)
        X = np.c_[context, time_col, z_a]  # (n, d + 1 + d_z)
        return X

    def fit(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]
        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError("`position` elements must be < `len_list`")

        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError("`action_dist` must be (n, A, L)")
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError("shape of `action_dist` must be (n, n_actions, len_list)")
            if pscore is None:
                raise ValueError("`pscore` must be given for 'iw' or 'mrdr'")

        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(context=context[idx], time=time[idx], action=action[idx])
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            elif self.fitting_method == "iw":
                w = action_dist[np.arange(n), action, pos_][idx] / pscore[idx]
                self.base_model_list[pos_].fit(X, reward[idx], sample_weight=w)
            else:  # "mrdr"
                w = action_dist[np.arange(n), action, pos_][idx]
                w *= 1.0 - pscore[idx]
                w /= np.clip(pscore[idx] ** 2, 1e-12, None)
                self.base_model_list[pos_].fit(X, reward[idx], sample_weight=w)

    def predict(self, context: np.ndarray, time: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        # まとめて埋め込みを計算（n, A, d_z）
        Z = self.action_embedding_func(time, self._action_ids)
        time_col = time.reshape(-1, 1)
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for a in range(self.n_actions):
            Xa = np.c_[context, time_col, Z[:, a, :]]
            for pos_ in np.arange(self.len_list):
                model = self.base_model_list[pos_]
                q = model.predict_proba(Xa)[:, 1] if is_classifier(model) else model.predict(Xa)
                q_hat[:, a, pos_] = q
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]
        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(context, time, action, reward, pscore, position, action_dist)
            return self.predict(context, time)

        q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for tr, te in kf.split(context):
            ad_tr = action_dist[tr] if action_dist is not None else None
            self.fit(context[tr], time[tr], action[tr], reward[tr],
                     pscore[tr], position[tr], ad_tr)
            # ★ テストには test の time を渡す（安全版）
            q_hat[te, :, :] = self.predict(context[te], time[te])
        return q_hat


# ------------------------------------------------------------
# 2) time_structure（カテゴリ/インデックス）を受け取る埋め込み版
#    既存 RegressionModelTimeStructure とインターフェースを合わせる
# ------------------------------------------------------------
@dataclass
class RegressionModelTimeStructureWithEmbedding(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    action_embedding_func: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (time_structure, action_ids) -> (n, A, d_z)
    len_list: int = 1
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if self.fitting_method not in ["normal", "iw", "mrdr"]:
            raise ValueError("`fitting_method` must be one of 'normal', 'iw', or 'mrdr'")
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError("`base_model` must be a sklearn BaseEstimator")
        self.base_model_list = [clone(self.base_model) for _ in np.arange(self.len_list)]
        self._action_ids = np.arange(self.n_actions)

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        Z = self.action_embedding_func(time_structure, self._action_ids)  # (n, A, d_z)
        z_a = Z[np.arange(len(action)), action]
        ts_col = time_structure.reshape(-1, 1)
        X = np.c_[context, ts_col, z_a]
        return X

    def fit(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]
        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError("`position` elements must be < `len_list`")

        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError("`action_dist` must be (n, A, L)")
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError("shape of `action_dist` must be (n, n_actions, len_list)")
            if pscore is None:
                raise ValueError("`pscore` must be given for 'iw' or 'mrdr'")

        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(context=context[idx],
                                                time_structure=time_structure[idx],
                                                action=action[idx])
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            elif self.fitting_method == "iw":
                w = action_dist[np.arange(n), action, pos_][idx] / pscore[idx]
                self.base_model_list[pos_].fit(X, reward[idx], sample_weight=w)
            else:
                w = action_dist[np.arange(n), action, pos_][idx]
                w *= 1.0 - pscore[idx]
                w /= np.clip(pscore[idx] ** 2, 1e-12, None)
                self.base_model_list[pos_].fit(X, reward[idx], sample_weight=w)

    def predict(self, context: np.ndarray, time_structure: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        Z = self.action_embedding_func(time_structure, self._action_ids)  # (n, A, d_z)
        ts_col = time_structure.reshape(-1, 1)
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for a in range(self.n_actions):
            Xa = np.c_[context, ts_col, Z[:, a, :]]
            for pos_ in np.arange(self.len_list):
                model = self.base_model_list[pos_]
                q = model.predict_proba(Xa)[:, 1] if is_classifier(model) else model.predict(Xa)
                q_hat[:, a, pos_] = q
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]
        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(context, time_structure, action, reward, pscore, position, action_dist)
            return self.predict(context, time_structure)

        q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for tr, te in kf.split(context):
            ad_tr = action_dist[tr] if action_dist is not None else None
            self.fit(context[tr], time_structure[tr], action[tr], reward[tr],
                     pscore[tr], position[tr], ad_tr)
            q_hat[te, :, :] = self.predict(context[te], time_structure[te])  # ★ test側を使う
        return q_hat
