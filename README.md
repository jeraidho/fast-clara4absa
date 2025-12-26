# Fast CLaRa for ABSA task

## Структура репозитория:

```
fast-clara-absa/
├── data/               # (only local)
│   ├── raw/            # original XML of MAMS dataset
│   └── processed/      # preprocessed data or cached tensors (if needed)
├── configs/            # configs and hyperparams (.py)
│   └── base_config.py
├── src/                # main code
│   ├── init.py
│   ├── data_utils.py   # utils for data processing
│   └── model.py        # script with model class
├── notebooks/          # tests notebook etc.
├── train.py            # main training loop script
├── requirements.txt
└── README.md
```
