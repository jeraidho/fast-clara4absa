# Fast CLaRa for ABSA 
**Continuous Latent Reasoning for Multi-Aspect Sentiment Analysis**

## Overview
Этот проект представляет собой реализацию и адаптацию отдельных идей архитектуры **CLaRa (Continuous Latent Reasoning)** (He et al., 2025) для задачи **Aspect-Based Sentiment Analysis (ABSA)**. 

Традиционные RAG и LLM системы страдают от ограниченного окна контекста и раздельной оптимизации поиска и генерации. **Fast CLaRa** решает эти проблемы путем сжатия текста в компактные **Memory Tokens** (непрерывные латентные представления) и совместного обучения ретривера (энкодера) и генератора (декодера) в едином дифференцируемом пространстве.

### Key Innovations:
*   **Shared Continuous Representations:** Сжатие предложений в 8-16 латентных векторов с сохранением многоаспектной семантики (CR ~4x).
*   **Bottleneck Reasoning:** Генератор принимает решения, опираясь исключительно на сжатую память, что заставляет модель проводить глубокую абстракцию смыслов.
*   **Multi-task Curriculum Learning:** Динамическое балансирование лоссов реконструкции текста (Reconstruction), извлечения структуры (Extraction) и классификации (Reasoning).
*   **Semantic Alignment ($L_{MSE}$):** Принудительное выравнивание латентной памяти с семантическим ядром оригинального текста для предотвращения галлюцинаций.

---

## Repository Structure

```text
fast-clara-absa/
├── configs/
│   ├── base_config.py      # Гиперпараметры, конфигурация LoRA и расписание весов лосса
│   └── schema.py           # Определение структур данных и типов задач
├── src/
│   ├── data_utils.py       # Парсинг XML (MAMS), токенизация и сборка батчей
│   ├── model.py            # Архитектура FastClaraModel и логика проброса градиентов
│   ├── logger.py           # Кастомный LocalLogger для CSV-мониторинга в offline-среде
│   └── __init__.py
├── notebooks/
│   ├── model_train.ipynb   # Основной Pipeline обучения: от инициализации до валидации
│   └── test_data_utils.ipynb # Юнит-тесты для проверки корректности сжатия и маскирования
├── research_logs/          # CSV-логи обучения (Losses, Learning Rate, Hyperparams.)
├── data/                   # (Local only) Исходные датасеты MAMS (train/val/test)
└── README.md
```

---

## Methodology

Архитектура базируется на одном **LoRA-адаптере** для Phi-3.5-mini, который обучается решать три задачи одновременно:

1.  **Encoder Phase (Compression):** `Text + [M] tokens` -> `Memory States`.
2.  **Decoder Phase (Reasoning):** `Memory States + Prompt` -> `Sentiment Prediction`.
3.  **Optimization:** Сквозной градиент течет от ответа декодера через латентные векторы обратно в веса компрессора.

### Loss Function
$$L_{total} = \alpha L_{rec} + \beta L_{absa} + \gamma L_{mse}$$
Где $\alpha$ и $\beta$ динамически изменяются в процессе обучения (Curriculum Learning), перенося фокус с точности сжатия на точность логического вывода.

---

## Quick Start


### 1. Data Preparation
Поместите файлы датасета MAMS (`train.xml`, `val.xml`) в папку `data/raw/`.

### 2. Training
Откройте и запустите `notebooks/model_train.ipynb`. 
*Скрипт автоматически настроен на эффективное использование памяти (BF16, Gradient Accumulation).*

### 3. Evaluation
Для проверки обученной модели используйте функцию `evaluate_metrics_pure_python` из ноутбука, которая вычисляет Accuracy и Macro-F1 без использования сторонних библиотек.

---

## Results (MAMS Dataset)
*Backbone: Phi-3.5-mini-instruct | Memory Tokens: 8 | Compression Ratio: 4.16x*

| Metric | Train | Test |
| :--- | :--- | :--- |
| **Accuracy** | ~76.1% | **~67.7%** |
| **Macro F1** | ~76.0% | **~67.3%** |

---

## Acknowledgments
В работе используются идеи статьи *"CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning"* (He et al., 2025).

---
*Developed as part of an experimental research project in NLP.*
