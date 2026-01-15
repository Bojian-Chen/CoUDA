# CoUDA

A PyTorch experimental framework for cross-domain incremental learning in fault diagnosis. The main training entry point is [`trainer.trainer.train`](trainer/trainer.py), which supports incremental adaptation to new domains after single-domain pretraining.

## Repository Structure

- `main.py`: Parses arguments, sets random seeds, calls `train`, and summarizes incremental metrics.
- `trainer/`: Hosts base training [`trainer.base_train.base_train`](trainer/base_train.py) and incremental training [`trainer.incremental_train.incremental_train`](trainer/incremental_train.py).
- `models/`: Provides the ResNet backbone and cosine/Euclidean classifiers, see [`models.resnet32`](models/resnet32.py) and [`models.modified_linear`](models/modified_linear.py).
- `utils.py`: Tooling for data loading, optimizer scheduling, evaluation, and feature processing.

## Dataset Layout

Data are stored as `.mat` files in `data/`, consumed by [`dataloader_domain.dataloader`](dataloader_domain.py). Each file must include:

- `data`: A float matrix shaped $(N, 1024)$, reshaped to $(N,1,32,32)$ or $(N,1,1024)$ according to `--data_dimension`.
- `label`: An integer vector of length $N$.
- Samples are organized as “domain → class → sample”: each domain contains `nb_cl` classes with exactly 100 samples per class. Indices follow
  $$
  \text{index} \in [d \times \text{nb\_cl} \times 100,\ (d+1) \times \text{nb\_cl} \times 100)
  $$
  where $d$ is the domain ID.
- `--Domain_Seq` defines the training session order. Session 0 uses domain `session=0`, subsequent sessions load new domains sequentially with optional few-shot or replay samples controlled by `--nb_shot` and `--nb_exemplar`.

## How to Run

1. Create the environment and install dependencies:
   ```sh
   conda create -n couda python=3.9
   conda activate couda
   pip install -r requirements.txt  # curate the list as needed
   ```
2. Place `.mat` files under `data/` and ensure fields and ordering follow the layout above.
3. Launch an example run:
   ```sh
   python main.py \
   ```
   - `--incremental_mode fine_tuning|single|ours` selects the incremental strategy.
   - `--classifer fc|cos|eu` switches the classification head.
   - `--nb_exemplar` and `--random_exemplar` manage replay data.

## Training Overview

- Session 0: `train` calls [`set_dataset`](utils.py) and [`set_optimizer`](utils.py) before running base training [`base_train`](trainer/base_train.py).
- Later sessions: choose standard fine-tuning or incremental training [`incremental_train`](trainer/incremental_train.py), optionally invoking `set_exemplar` for sample selection.
- Evaluation: `evaluate` reports per-domain accuracy, outputs confusion matrices, and saves features or model checkpoints when `--save_model` is enabled.
- Metrics: Upon completion the script prints $AF$, $AMF$, $AG$, $AA$, $BWT$, $ACC$, and other incremental learning metrics.

## Logs and Results

Models and features are saved under:
```
log/<dataset>/<incremental_mode>/<preprocess>/<train_list>_<backbone>_<...>/<filename>.pth
```
All metrics and per-session accuracies are recorded in `Result.csv` when `--save_model` is enabled.