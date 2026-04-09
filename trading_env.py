"""
Shared trading environment for RL models
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, List


class StockTradingEnv(gym.Env):
    """Stock trading environment for SB3"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int = 5,
        hmax: int = 10,
        initial_amount: float = 100000.0,
        buy_cost_pct: Optional[List[float]] = None,
        sell_cost_pct: Optional[List[float]] = None,
        reward_scaling: float = 1e-4,
        tech_indicator_list: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__()

        self.df = df.copy()
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.buy_cost_pct = buy_cost_pct or [0.001] * stock_dim
        self.sell_cost_pct = sell_cost_pct or [0.001] * stock_dim
        self.tech_indicator_list = tech_indicator_list or ["close", "volume"]

        self._setup_data()
        self._setup_spaces()

        if random_seed is not None:
            np.random.seed(random_seed)

    def _setup_data(self):
        """Pre-process ticker data for O(1) access"""
        self.tickers = sorted(self.df["tic"].unique())[: self.stock_dim]
        self.ticker_data = {
            tic: self.df[self.df["tic"] == tic]
            .sort_values("date")
            .reset_index(drop=True)
            for tic in self.tickers
        }
        self.max_steps = min(len(dd) for dd in self.ticker_data.values())

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        self.state_space = (
            len(self.tech_indicator_list) * self.stock_dim + 1 + self.stock_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_amount
        self.stocks = np.zeros(self.stock_dim)
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        for tic in self.tickers:
            data = self.ticker_data[tic]
            if self.current_step < len(data):
                row = data.iloc[self.current_step]
                for f in self.tech_indicator_list:
                    obs.append(float(row.get(f, 0)))
            else:
                obs.extend([0.0] * len(self.tech_indicator_list))

        obs.append(float(self.balance / self.initial_amount))
        obs.extend([float(s / 100) for s in self.stocks])
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        total_reward = 0.0

        for i, tic in enumerate(self.tickers):
            data = self.ticker_data[tic]
            if self.current_step >= len(data):
                return self._get_obs(), 0.0, True, False, {}

            curr = float(data.iloc[self.current_step]["close"])
            next_pr = float(
                data.iloc[min(self.current_step + 1, len(data) - 1)]["close"]
            )
            trade = action[i] * self.hmax

            if trade > 0:
                shares = int(min(trade, self.balance / curr))
                cost = shares * curr * (1 + self.buy_cost_pct[i])
                if cost <= self.balance:
                    self.balance -= cost
                    self.stocks[i] += shares
            elif trade < 0:
                shares = int(min(-trade, self.stocks[i]))
                self.balance += shares * curr * (1 - self.sell_cost_pct[i])
                self.stocks[i] -= shares

            total_reward += (next_pr - curr) / curr * self.stocks[i]

        self.current_step += 1

        portfolio_value = self.balance + sum(
            self.stocks[i]
            * float(
                self.ticker_data[t].iloc[
                    min(self.current_step, len(self.ticker_data[t]) - 1)
                ]["close"]
            )
            for i, t in enumerate(self.tickers)
        )
        self.history.append(
            {"step": self.current_step, "portfolio_value": portfolio_value}
        )

        done = self.current_step >= self.max_steps - 1
        info = {"total_value": portfolio_value}
        return self._get_obs(), total_reward * self.reward_scaling, done, False, info

    def render(self, mode: str = "human"):
        pass


class StockTradingEnvWithEnsemble(StockTradingEnv):
    """Enhanced environment with ensemble predictions"""

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int = 5,
        hmax: int = 10,
        initial_amount: float = 100000.0,
        buy_cost_pct: Optional[List[float]] = None,
        sell_cost_pct: Optional[List[float]] = None,
        reward_scaling: float = 1e-4,
        tech_indicator_list: Optional[List[str]] = None,
        ensemble_models: Optional[Dict[str, Any]] = None,
        ensemble_feature_list: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ):
        self.ensemble_models = ensemble_models
        self.ensemble_feature_list = ensemble_feature_list or ["close"]

        super().__init__(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            tech_indicator_list=tech_indicator_list,
            random_seed=random_seed,
        )

        self.state_dim = self.state_space + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def _get_ensemble_prob(self) -> float:
        """Get ensemble prediction probability"""
        if self.ensemble_models is None:
            return 0.5

        rf_model = self.ensemble_models.get("rf")
        gb_model = self.ensemble_models.get("gb")
        scaler = self.ensemble_models.get("scaler")

        if not all([rf_model, gb_model, scaler]):
            return 0.5

        first_ticker = self.tickers[0]
        data = self.ticker_data[first_ticker]

        if self.current_step >= len(data):
            return 0.5

        row = data.iloc[self.current_step]
        feat_vals = [
            row.get(f, 0) for f in self.ensemble_feature_list if f in row.index
        ]

        if len(feat_vals) == 0:
            return 0.5

        features_scaled = scaler.transform([feat_vals])
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        gb_prob = gb_model.predict_proba(features_scaled)[0][1]

        return (rf_prob + gb_prob) / 2

    def _get_obs(self) -> np.ndarray:
        """Get observation with ensemble prediction"""
        obs = super()._get_obs()
        ensemble_prob = self._get_ensemble_prob()
        return np.append(obs, ensemble_prob).astype(np.float32)
