"""
ДЗ 7: Финальный ансамбль - Бектест и оценка качества RL модели
"""

import os
import json
import matplotlib.pyplot as plt
from config import (
    OUTPUT_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    STOCK_DIM,
    INITIAL_BALANCE,
    COMMISSION,
    FEATURES,
    RL_ALGORITHM,
)
from utils import logger, load_data, get_env_kwargs, calculate_metrics

import numpy as np
from stable_baselines3 import PPO, A2C
from trading_env import StockTradingEnv


def run_backtest(env, model):
    """Run backtest and return history"""
    obs = env.reset(seed=42)[0]
    done = False
    history = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        history.append(
            {
                "step": len(history),
                "portfolio_value": info.get("total_value", INITIAL_BALANCE),
            }
        )
    return history


def run_benchmark(env, initial_balance):
    """Buy and hold benchmark"""
    env.reset(seed=42)
    per_ticker = initial_balance / len(env.tickers)
    for i, ticker in enumerate(env.tickers):
        data = env.ticker_data[ticker]
        if len(data) > 0:
            env.stocks[i] = per_ticker / data.iloc[0]["close"]
            env.balance -= env.stocks[i] * data.iloc[0]["close"] * 1.001
    final_value = env.balance + sum(
        env.stocks[i] * env.ticker_data[t].iloc[-1]["close"]
        for i, t in enumerate(env.tickers)
    )
    return final_value


def main():
    logger.info("=" * 60)
    logger.info("ДЗ 7: Финальный ансамбль - Бектест")

    train_df, trade_df = load_data(OUTPUT_DIR)
    logger.info(f"Train: {len(train_df)} rows, Trade: {len(trade_df)} rows")

    model_path = os.path.join(MODEL_DIR, f"{RL_ALGORITHM.lower()}_stock_trading")
    try:
        if RL_ALGORITHM == "A2C":
            model = A2C.load(model_path)
        else:
            model = PPO.load(model_path)
        logger.info(f"Модель загружена: {model_path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return

    env_kwargs = get_env_kwargs(
        df=trade_df,
        stock_dim=STOCK_DIM,
        initial_amount=INITIAL_BALANCE,
        commission=COMMISSION,
        tech_indicator_list=FEATURES["trading"],
    )

    env = StockTradingEnv(**env_kwargs)

    logger.info("Запуск бектеста...")
    history = run_backtest(env, model)
    metrics = calculate_metrics(history, INITIAL_BALANCE)

    benchmark_value = run_benchmark(env, INITIAL_BALANCE)
    benchmark_return = (benchmark_value - INITIAL_BALANCE) / INITIAL_BALANCE

    logger.info(f"Результаты RL:")
    logger.info(f"    Начальный: ${INITIAL_BALANCE:,.2f}")
    logger.info(f"    Финальный: ${metrics['final_value']:,.2f}")
    logger.info(f"    Доходность: {metrics['total_return'] * 100:.2f}%")
    logger.info(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"    Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    logger.info(f"Бенчмарк (Buy & Hold): {benchmark_return * 100:.2f}%")

    df = metrics["history"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(df["portfolio_value"], label="RL Model")
    axes[0].axhline(y=INITIAL_BALANCE, color="r", linestyle="--", label="Initial")
    axes[0].set_title("Portfolio Value Over Time")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df["cum_returns"] * 100, label="Cumulative Returns")
    axes[1].set_title("Cumulative Returns (%)")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "backtest_result.png"), dpi=100)

    df.to_csv(os.path.join(RESULTS_DIR, "backtest_history.csv"), index=False)

    metrics_json = {
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_value": metrics["final_value"],
        "benchmark_return": benchmark_return,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("Бектест завершен!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
