# LiquidFusion

LiquidFusion is an attention mechanism that blends the ideas of H2O (Heavy Hitter Oracle for Efficient Generative Inference of Large Language Models) and 
StreamingLLM. 

# Notice
The current state of this repo is in progress, therefore some of the evaluation and utility code is partially complete or not correctly referenced. 
Base model code is mostly correct but also still in progress.

## Installation

```bash
git clone https://github.com/thomasjhuang/liquid-fusion.git
cd LiquidFusion
pip install -r requirements.txt
```

## Directory Structure

```bash
liquid-fusion/
├── configs/
│   ├── experiment.yaml
│   ├── model_configs.py
│   └── custom/
│       └── fast_test.yaml
├── data/
│   ├── datasets.py
│   └── preprocessing.py
├── experiments/
│   └── evaluate.py
├── models/
│   ├── attention/
│   │   ├── liquid_fusion.py
│   │   ├── h2o.py
│   │   └── streaming.py
│   └── base_models.py
├── notebooks/
├── scripts/
│   └── run_experiment.py
└── utils/
    └── visualization.py
```

## Installation

```bash
git clone https://github.com/thomasjhuang/liquid-fusion.git
cd liquid-fusion
pip install -r requirements.txt
```

## Usage
Run experiments:
```bash
python -m liquid-fusion.scripts.run_experiment
```

With custom config:
```bash
python -m liquid-fusion.scripts.run_experiment experiment=custom/fast_test
```
