# ДЗ №7: Финальный ансамбль (FinRL)

Реализация домашнего задания по работе с моделями обучения с подкреплением (RL) и фреймворком FinRL.

## Цель работы

1. Добавить RL модель в торговую систему
2. Оценить качество решения
3. Сформировать ансамбль моделей
4. Добавить RL модель на "вершину" ансамбля

## Структура проекта

```
.
├── main.py                    # Единая точка входа
├── config.py                  # Конфигурация
├── conf.json                  # Параметры
├── utils.py                   # Утилиты
├── trading_env.py             # RL окружение
├── 01_data.py                 # Загрузка и обработка данных
├── 02_train.py                # Обучение RL модели
├── 03_backtest.py             # Бектест и оценка качества
├── 04_ensemble.py             # Создание ML ансамбля
├── 05_rl_ensemble.py          # RL на вершине ансамбля
├── 06_report.py               # Dashboard
├── train_data.csv              # Обучающие данные
├── trade_data.csv              # Торговые данные
├── trained_models/             # Обученные RL модели
├── ensemble_models/            # ML модели ансамбля
├── rl_ensemble_models/         # RL модель с ансамблем
└── results/                    # Результаты
```

## Установка

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка TA-Lib (требуется отдельно)
# Linux: https://ta-lib.github.io/ta-lib-python/
# Mac: brew install ta-lib
# Windows: скачать .dll с https://ta-lib.github.io/ta-lib-python/
pip install TA-Lib
```

## Использование

### Единая точка входа

```bash
# Запустить весь пайплайн
python main.py all

# Запустить конкретный модуль
python main.py data        # Этап 1: Данные
python main.py train       # Этап 2: RL обучение
python main.py backtest    # Этап 3: Бектест
python main.py ensemble    # Этап 4: ML ансамбль
python main.py rl_ensemble # Этап 5: RL+Ensemble
python main.py report      # Этап 6: Dashboard
```

### Отдельные скрипты

```bash
python 01_data.py
python 02_train.py
python 03_backtest.py
python 04_ensemble.py
python 05_rl_ensemble.py
python 06_report.py
```

## Конфигурация

Параметры в `conf.json`:

- **data**: периоды train/test
- **trading**: stock_dim, initial_balance, commission
- **rl**: algorithm, timesteps, learning_rate, num_cpu
- **ml**: n_estimators, max_depth, ensemble_weights
- **features**: список признаков для RL и ML

## Технические детали

- **Данные**: DOW 30 (27 тикеров), период 2010-2025
- **Индикаторы**: SMA, RSI, MACD, Bollinger Bands, ADX, CCI, MOM, ROC, OBV, VIX, Turbulence
- **RL алгоритм**: A2C (по умолчанию) или PPO
- **ML модели**: Random Forest, Gradient Boosting
- **Многопроцессорность**: 16 процессов для обучения

## Требования

- Python 3.14+
- stable-baselines3
- gymnasium
- scikit-learn
- yfinance
- pandas, numpy, matplotlib
- TA-Lib
- FinRL
