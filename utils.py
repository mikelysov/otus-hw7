"""
Shared utilities for all scripts
"""

import os
import logging
import warnings
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class LearningRateScheduler(BaseCallback):
    """Reduce learning rate on plateau"""

    def __init__(
        self,
        initial_lr: float,
        min_lr: float = 1e-10,
        factor: float = 0.5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.min_lr = min_lr
        self.factor = factor

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            try:
                if hasattr(self.model, "learning_rate"):
                    current_lr = self.model.learning_rate
                    if current_lr > self.min_lr:
                        self.model.learning_rate = max(
                            current_lr * self.factor, self.min_lr
                        )
                        if self.verbose:
                            logger.info(f"Reduced LR to {self.model.learning_rate:.2e}")
            except Exception:
                pass
        return True


def load_data(output_dir: str):
    """Load train and trade data"""
    train = pd.read_csv(os.path.join(output_dir, "train_data.csv"))
    trade = pd.read_csv(os.path.join(output_dir, "trade_data.csv"))
    return train, trade


def get_env_kwargs(
    df: pd.DataFrame,
    stock_dim: int,
    initial_amount: float,
    commission: float,
    hmax: int = 10,
    reward_scaling: float = 1e-4,
    tech_indicator_list: Optional[List[str]] = None,
    **kwargs,
):
    """Create environment kwargs dict"""
    if tech_indicator_list is None:
        tech_indicator_list = ["close", "volume"]
    return {
        "df": df,
        "stock_dim": stock_dim,
        "hmax": hmax,
        "initial_amount": initial_amount,
        "buy_cost_pct": [commission] * stock_dim,
        "sell_cost_pct": [commission] * stock_dim,
        "reward_scaling": reward_scaling,
        "tech_indicator_list": tech_indicator_list,
        "random_seed": RANDOM_SEED,
        **kwargs,
    }


def calculate_metrics(history: list, initial_balance: float) -> dict:
    """Calculate portfolio metrics"""
    df = pd.DataFrame(history)
    df["returns"] = df["portfolio_value"].pct_change()
    df["cum_returns"] = (1 + df["returns"]).cumprod() - 1
    total_return = (df["portfolio_value"].iloc[-1] - initial_balance) / initial_balance
    avg_return = df["returns"].mean() * 252
    std_return = df["returns"].std() * np.sqrt(252)
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    df["drawdown"] = df["portfolio_value"] / df["portfolio_value"].cummax() - 1
    max_drawdown = df["drawdown"].min()
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_value": df["portfolio_value"].iloc[-1],
        "history": df,
    }
