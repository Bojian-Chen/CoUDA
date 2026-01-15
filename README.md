# CoUDA
This is a PyTorch implementation of CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis Under Dynamic Working Conditions https://ieeexplore.ieee.org/document/10896871.
## Repository Structure

- `main.py`: argument parsing, seeding, `train`, and metric reporting.
- `trainer/`: base (`base_train`) and incremental (`incremental_train`) pipelines.
- `models/`: ResNet backbone plus cosine/Euclidean heads.
- `utils.py`: helpers for datasets, optimizers, evaluation, and features.

## Dataset Layout

- `.mat` files under `data/`, loaded by [`dataloader_domain.dataloader`](dataloader_domain.py).
- `data`: shape $(N,1024)$, reshaped via `--data_dimension`.
- `label`: length $N$ integers.
- Samples follow domain → class → 100 samples; indices obey $d \times \text{nb\_cl} \times 100$ offsets.
- `--Domain_Seq` sets session order; `--nb_shot` and `--nb_exemplar` toggle few-shot and replay data.

## Quick Start

1. Environment:
   ```sh
   conda create -n couda python=3.9
   conda activate couda
   pip install -r requirements.txt
   ```
2. Place `.mat` files in `data/` and ensure their fields follow the layout above.
3. Run:
   ```sh
   python main.py \
   ```
   - `--incremental_mode fine_tuning|single|ours`
   - `--classifer fc|cos|eu`
   - `--nb_exemplar`, `--random_exemplar`

## Training Overview

- Session 0: call [`set_dataset`](utils.py), [`set_optimizer`](utils.py), then [`base_train`](trainer/base_train.py).
- Later sessions: choose fine-tuning or [`incremental_train`](trainer/incremental_train.py) and optionally use `set_exemplar`.
- Evaluation: `evaluate` reports per-domain accuracy, outputs confusion matrices, and can save models/features when `--save_model` is enabled.
- Metrics: the script prints $AF$, $AMF$, $AG$, $AA$, $BWT$, $ACC$, and more after training.

## Logs and Results

Models and features are stored under:
```
log/<dataset>/<incremental_mode>/<preprocess>/<train_list>_<backbone>_<...>/<filename>.pth
```
Enable `--save_model` to generate `Result.csv` with per-session accuracy.

## References

@ARTICLE{10896871,
  author={Chen, Bojian and Zhang, Xinmin and Shen, Changqing and Li, Qi and Song, Zhihuan},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis Under Dynamic Working Conditions}, 
  year={2025},
  volume={21},
  number={5},
  pages={4072-4082},
  keywords={Adaptation models;Fault diagnosis;Employee welfare;Data privacy;Prototypes;Data models;Representation learning;Measurement;Contrastive learning;Training;Catastrophic forgetting;continual learning;fault diagnosis;unsupervised domain adaptation (UDA)},
  doi={10.1109/TII.2025.3538135}}
