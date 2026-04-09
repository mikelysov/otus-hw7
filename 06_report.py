"""
ДЗ №7: Финальный ансамбль - Dashboard
Usage: python 06_report.py
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from config import (
    OUTPUT_DIR,
    MODEL_DIR,
    ENSEMBLE_DIR,
    RL_ENSEMBLE_DIR,
    RESULTS_DIR,
    TRAIN_START,
    TRAIN_END,
    TRADE_START,
    TRADE_END,
    STOCK_DIM,
    INITIAL_BALANCE,
    TOTAL_TIMESTEPS,
    RL_ALGORITHM,
)


def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return default


def main():
    print("=" * 60)
    print("ДЗ №7: Финальный ансамбль - Dashboard")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ДЗ №7: Финальный ансамбль с RL", fontsize=14, fontweight="bold")

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    print("\n[1/3] RL Бектест...")
    metrics = load_json(os.path.join(RESULTS_DIR, "metrics.json"))
    if metrics and os.path.exists(os.path.join(RESULTS_DIR, "backtest_history.csv")):
        df = pd.read_csv(os.path.join(RESULTS_DIR, "backtest_history.csv"))
        ax1.plot(df["portfolio_value"], "b-", label="RL Model", linewidth=1.5)
        ax1.axhline(y=INITIAL_BALANCE, color="r", linestyle="--", label="Initial")
        ax1.set_title(f"RL Бектест: {metrics.get('total_return', 0) * 100:.2f}%")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "Нет данных\nЗапустите backtest",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("RL Бектест")

    print("[2/3] ML Ансамбль...")
    ens_results = load_json(os.path.join(ENSEMBLE_DIR, "ensemble_results.json"))
    if ens_results:
        models = ["RF", "GB", "Voting", "Weighted"]
        accuracies = [
            ens_results.get("rf_accuracy", 0),
            ens_results.get("gb_accuracy", 0),
            ens_results.get("ensemble_accuracy", 0),
            ens_results.get("weighted_accuracy", 0),
        ]
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
        bars = ax2.bar(models, accuracies, color=colors, edgecolor="black")
        ax2.axhline(y=0.5, color="gray", linestyle="--", label="Random (50%)")
        ax2.set_ylim(0.45, 0.65)
        ax2.set_title("ML Ансамбль Accuracy")
        ax2.set_ylabel("Accuracy")
        for bar, acc in zip(bars, accuracies):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "Нет данных\nЗапустите ensemble",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("ML Ансамбль")

    print("[3/3] RL + Ensemble...")
    rl_ens = load_json(os.path.join(RL_ENSEMBLE_DIR, "results.json"))
    if rl_ens and metrics:
        returns = [metrics.get("total_return", 0) * 100, rl_ens.get("return_pct", 0)]
        labels = ["RL Only", "RL + Ensemble"]
        colors = ["#3498db", "#e74c3c"]
        bars = ax3.bar(labels, returns, color=colors, edgecolor="black")
        ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax3.set_title("Сравнение доходности")
        ax3.set_ylabel("Return (%)")
        for bar, ret in zip(bars, returns):
            y = bar.get_height() + 1 if ret >= 0 else bar.get_height() - 3
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{ret:.1f}%",
                ha="center",
                va="bottom" if ret >= 0 else "top",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "Нет данных\nЗапустите rl_ensemble",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("RL + Ensemble")

    print("Генерация итогов...")
    rl_return = f"{metrics.get('total_return', 0) * 100:.2f}%" if metrics else "N/A"
    sharpe = f"{metrics.get('sharpe_ratio', 0):.2f}" if metrics else "N/A"
    max_dd = f"{metrics.get('max_drawdown', 0) * 100:.2f}%" if metrics else "N/A"
    ens_acc = (
        f"{ens_results.get('weighted_accuracy', 0) * 100:.1f}%"
        if ens_results
        else "N/A"
    )

    info_text = f"""Конфигурация:
• Период: {TRAIN_START} - {TRADE_END}
• Stock Dim: {STOCK_DIM}
• Initial: ${INITIAL_BALANCE:,}
• RL: {RL_ALGORITHM} ({TOTAL_TIMESTEPS:,} steps)

Метрики:
• RL Return: {rl_return}
• Sharpe: {sharpe}
• Max DD: {max_dd}
• Ensemble Acc: {ens_acc}"""
    ax4.text(
        0.05,
        0.95,
        info_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )
    ax4.axis("off")
    ax4.set_title("Итоги")

    plt.tight_layout()

    dashboard_path = os.path.join(RESULTS_DIR, "dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    print(f"\nDashboard сохранен: {dashboard_path}")

    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()
