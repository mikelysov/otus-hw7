"""
Конфигурация проекта ДЗ №7: Финальный ансамбль с RL
Загружается из conf.json
"""

import os
import json
from typing import Dict, Any, List

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf.json")

with open(CONFIG_PATH, "r") as f:
    _cfg: Dict[str, Any] = json.load(f)

OUTPUT_DIR: str = _cfg["dirs"]["output"]
MODEL_DIR: str = _cfg["dirs"]["model"]
ENSEMBLE_DIR: str = _cfg["dirs"]["ensemble"]
RL_ENSEMBLE_DIR: str = _cfg["dirs"]["rl_ensemble"]
RESULTS_DIR: str = _cfg["dirs"]["results"]

TRAIN_START: str = _cfg["data"]["train_start"]
TRAIN_END: str = _cfg["data"]["train_end"]
TRADE_START: str = _cfg["data"]["trade_start"]
TRADE_END: str = _cfg["data"]["trade_end"]

DOW_30_TICKERS: List[str] = _cfg["tickers"]

STOCK_DIM: int = _cfg["trading"]["stock_dim"]
INITIAL_BALANCE: float = _cfg["trading"]["initial_balance"]
COMMISSION: float = _cfg["trading"]["commission"]

TOTAL_TIMESTEPS: int = _cfg["rl"]["total_timesteps"]
N_STEPS: int = _cfg["rl"]["n_steps"]
BATCH_SIZE: int = _cfg["rl"]["batch_size"]
LEARNING_RATE: float = _cfg["rl"]["learning_rate"]
NUM_CPU: int = _cfg["rl"]["num_cpu"]
USE_GPU: bool = _cfg["rl"].get("use_gpu", False)
RL_POLICY: str = _cfg["rl"].get("policy", "MlpPolicy")
RL_ALGORITHM: str = _cfg["rl"].get("algorithm", "PPO")

RF_N_ESTIMATORS: int = _cfg["ml"]["rf_n_estimators"]
RF_MAX_DEPTH: int = _cfg["ml"]["rf_max_depth"]
GB_N_ESTIMATORS: int = _cfg["ml"]["gb_n_estimators"]
GB_MAX_DEPTH: int = _cfg["ml"]["gb_max_depth"]
LABEL_THRESHOLD: float = _cfg["ml"]["label_threshold"]
ENSEMBLE_WEIGHTS: Dict[str, float] = _cfg["ml"]["ensemble_weights"]

FEATURES: Dict[str, List[str]] = _cfg["features"]
INDICATORS: Dict[str, Any] = _cfg["indicators"]
SMA_WINDOWS: List[int] = _cfg["indicators"]["sma_windows"]
TURBULENCE_LOOKBACK: int = _cfg["indicators"]["turbulence_lookback"]
