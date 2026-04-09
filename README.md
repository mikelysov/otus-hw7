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
├── 01_data.py              # Загрузка и обработка данных
├── 02_train.py             # Обучение RL модели PPO
├── 03_backtest.py          # Бектест и оценка качества
├── 04_ensemble.py          # Создание ML ансамбля
├── 05_rl_ensemble.py       # RL на вершине ансамбля
├── 06_report.py            # Итоговый отчет
├── train_data.csv          # Обучающие данные (2015-2024)
├── trade_data.csv          # Торговые данные (2025)
├── trained_models/         # Обученные RL модели
├── ensemble_models/       # ML модели ансамбля
├── rl_ensemble_models/    # RL модель с ансамблем
└── results/               # Результаты бектеста
```

## Установка

```bash
# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
pip install yfinance pandas numpy scikit-learn matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3
pip install finrl alpaca-trade-api exchange-calendars stockstats
```

## Использование

### Этап 1: Загрузка данных
```bash
python 01_data.py
```
Загружает данные DOW 30 (27 тикеров), добавляет технические индикаторы и сохраняет в CSV.

### Этап 2: Обучение RL модели
```bash
python 02_train.py
```
Обучает PPO модель для торговли акциями.

### Этап 3: Бектест
```bash
python 03_backtest.py
```
Запускает бектест и сравнивает с бенчмарком Buy & Hold.

### Этап 4: Создание ансамбля
```bash
python 04_ensemble.py
```
Обучает Random Forest и Gradient Boosting классификаторы.

### Этап 5: RL + Ансамбль
```bash
python 05_rl_ensemble.py
```
Обучает RL модель с предсказаниями ансамбля в качестве признака.

### Этап 6: Отчет
```bash
python 06_report.py
```
Выводит итоговый отчет о проделанной работе.

## Результаты

| Модель | Доходность | Заметки |
|--------|------------|---------|
| RL (PPO) | -76.12% | Требует больше timesteps |
| Buy & Hold | +31.88% | Бенчмарк |
| ML Ансамбль | 57.39% | Точность классификации |

## Технические детали

- **Данные**: DOW 30 (27 тикеров), период 2015-2025
- **Индикаторы**: SMA, RSI, MACD, Bollinger Bands, VIX
- **RL алгоритм**: PPO (Proximal Policy Optimization)
- **ML модели**: Random Forest, Gradient Boosting
- **Библиотеки**: FinRL, Stable Baselines3, scikit-learn

## Требования

- Python 3.14+
- FinRL
- Stable Baselines3
- scikit-learn
- yfinance
- pandas, numpy, matplotlib
