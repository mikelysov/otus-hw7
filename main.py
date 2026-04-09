"""
ДЗ 7: Финальный ансамбль - Единая точка входа
Usage: python main.py [command]

Commands:
  all         - Запустить весь пайплайн (01-06)
  data        - Этап 1: Загрузка и обработка данных
  train       - Этап 2: Обучение RL модели
  backtest    - Этап 3: Бектест
  ensemble    - Этап 4: ML ансамбль
  rl_ensemble - Этап 5: RL на вершине ML ансамбля
  report      - Этап 6: Отчет
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODULES = {
    "data": "01_data.py",
    "train": "02_train.py",
    "backtest": "03_backtest.py",
    "ensemble": "04_ensemble.py",
    "rl_ensemble": "05_rl_ensemble.py",
    "report": "06_report.py",
}

SEQUENCE = ["data", "train", "backtest", "ensemble", "rl_ensemble", "report"]


def run_module(name: str) -> int:
    """Run a single module"""
    script = MODULES[name]
    print(f"\n{'=' * 60}")
    print(f"Running: {script}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script])
    return result.returncode


def run_all():
    """Run all modules in sequence"""
    for name in SEQUENCE:
        code = run_module(name)
        if code != 0:
            print(f"[ERROR] {MODULES[name]} failed with code {code}")
            return code
    print("\n" + "=" * 60)
    print("All modules completed successfully!")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="ДЗ 7: Финальный ансамбль")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", *MODULES.keys()],
        help="Команда для запуска",
    )
    args = parser.parse_args()

    print(f"ДЗ 7: Финальный ансамбль - Команда: {args.command}")

    if args.command == "all":
        return run_all()
    else:
        return run_module(args.command)


if __name__ == "__main__":
    sys.exit(main())
