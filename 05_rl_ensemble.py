"""
ДЗ 7: Финальный ансамбль - RL на вершине ML ансамбля
"""

import os
import json
from pathlib import Path
from config import (
    OUTPUT_DIR,
    ENSEMBLE_DIR,
    RL_ENSEMBLE_DIR,
    STOCK_DIM,
    INITIAL_BALANCE,
    COMMISSION,
    TOTAL_TIMESTEPS,
    N_STEPS,
    BATCH_SIZE,
    LEARNING_RATE,
    FEATURES,
    NUM_CPU,
    USE_GPU,
    RL_POLICY,
    RL_ALGORITHM,
)
from utils import logger, load_data, get_env_kwargs, LearningRateScheduler

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import joblib
from trading_env import StockTradingEnvWithEnsemble


def main():
    logger.info("=" * 60)
    logger.info("ДЗ 7: Финальный ансамбль - RL на вершине ML ансамбля")
    logger.info(f"Используем {NUM_CPU} процессов")

    train_df, trade_df = load_data(OUTPUT_DIR)

    ensemble_models = {
        "rf": joblib.load(os.path.join(ENSEMBLE_DIR, "rf_model.pkl")),
        "gb": joblib.load(os.path.join(ENSEMBLE_DIR, "gb_model.pkl")),
        "scaler": joblib.load(os.path.join(ENSEMBLE_DIR, "scaler.pkl")),
    }

    device = "cuda" if USE_GPU else "cpu"

    env_kwargs = get_env_kwargs(
        df=trade_df,
        stock_dim=STOCK_DIM,
        initial_amount=INITIAL_BALANCE,
        commission=COMMISSION,
        tech_indicator_list=FEATURES["trading"],
        ensemble_models=ensemble_models,
        ensemble_feature_list=FEATURES["ml"],
    )

    logger.info(f"Создание окружения ({NUM_CPU} процессов)")
    train_env = make_vec_env(
        StockTradingEnvWithEnsemble,
        n_envs=NUM_CPU,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={**env_kwargs, "df": train_df},
    )

    eval_env = make_vec_env(
        StockTradingEnvWithEnsemble, n_envs=1, env_kwargs={**env_kwargs, "df": trade_df}
    )

    MODEL_PATH = os.path.join(RL_ENSEMBLE_DIR, "a2c_with_ensemble")

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=RL_ENSEMBLE_DIR,
            n_eval_episodes=3,
            deterministic=True,
        ),
        LearningRateScheduler(initial_lr=LEARNING_RATE, min_lr=1e-10),
    ]

    logger.info(
        f"Обучение {RL_ALGORITHM} ({TOTAL_TIMESTEPS} timesteps, device={device})..."
    )

    if RL_ALGORITHM == "A2C":
        model = A2C(
            RL_POLICY,
            train_env,
            verbose=0,
            n_steps=N_STEPS,
            learning_rate=LEARNING_RATE,
            device=device,
        )
    else:
        model = PPO(
            RL_POLICY,
            train_env,
            verbose=0,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(MODEL_PATH)
    model_size = Path(MODEL_PATH + ".zip").stat().st_size / 1024 / 1024

    logger.info("Бектест RL+Ensemble")
    eval_env_single = StockTradingEnvWithEnsemble(
        df=trade_df,
        stock_dim=STOCK_DIM,
        initial_amount=INITIAL_BALANCE,
        buy_cost_pct=[COMMISSION] * STOCK_DIM,
        sell_cost_pct=[COMMISSION] * STOCK_DIM,
        reward_scaling=1e-4,
        tech_indicator_list=FEATURES["trading"],
        ensemble_models=ensemble_models,
        ensemble_feature_list=FEATURES["ml"],
        random_seed=42,
    )

    obs = eval_env_single.reset()[0]
    done = False
    history = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env_single.step(action)
        if "total_value" in info:
            history.append(info["total_value"])

    final_value = history[-1] if history else INITIAL_BALANCE
    return_pct = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    logger.info(f"Начальный: ${INITIAL_BALANCE:,.2f}")
    logger.info(f"Финальный: ${final_value:,.2f}")
    logger.info(f"Доходность: {return_pct:.2f}%")

    with open(os.path.join(RL_ENSEMBLE_DIR, "results.json"), "w") as f:
        json.dump({"final_value": final_value, "return_pct": return_pct}, f, indent=2)

    logger.info(f"Модель: {MODEL_PATH}.zip ({model_size:.2f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
