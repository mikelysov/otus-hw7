"""
ДЗ 7: Финальный ансамбль - Обучение RL модели
Многопроцессорное обучение с SubprocVecEnv
"""

import os
from pathlib import Path
from config import (
    OUTPUT_DIR,
    MODEL_DIR,
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
from utils import (
    logger,
    load_data,
    get_env_kwargs,
    LearningRateScheduler,
)

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from trading_env import StockTradingEnv


def main():
    logger.info("=" * 60)
    logger.info("ДЗ 7: Финальный ансамбль - Обучение RL модели")
    logger.info(f"Используем {NUM_CPU} процессов")

    np.random.seed(42)

    train_df, trade_df = load_data(OUTPUT_DIR)

    device = "cuda" if USE_GPU else "cpu"
    env_kwargs = get_env_kwargs(
        df=train_df,  # Train on train data
        stock_dim=STOCK_DIM,
        initial_amount=INITIAL_BALANCE,
        commission=COMMISSION,
        tech_indicator_list=FEATURES["trading"],
    )

    eval_env_kwargs = get_env_kwargs(
        df=trade_df,  # Evaluate on trade data
        stock_dim=STOCK_DIM,
        initial_amount=INITIAL_BALANCE,
        commission=COMMISSION,
        tech_indicator_list=FEATURES["trading"],
    )

    logger.info(f"Создание векторизованного окружения ({NUM_CPU} процессов)")
    vec_env = make_vec_env(
        StockTradingEnv,
        n_envs=NUM_CPU,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
    )

    logger.info(
        f"Observation: {vec_env.observation_space}, Action: {vec_env.action_space}"
    )

    eval_env = make_vec_env(StockTradingEnv, n_envs=1, env_kwargs=eval_env_kwargs)

    model_path = os.path.join(MODEL_DIR, f"{RL_ALGORITHM.lower()}_stock_trading")

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=model_path.rsplit("/", 1)[0],
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
            vec_env,
            verbose=0,
            n_steps=N_STEPS,
            learning_rate=LEARNING_RATE,
            device=device,
        )
    else:
        model = PPO(
            RL_POLICY,
            vec_env,
            verbose=0,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(model_path)
    model_size = Path(model_path + ".zip").stat().st_size / 1024 / 1024

    logger.info(f"Готово! Модель: {model_path}.zip ({model_size:.2f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
