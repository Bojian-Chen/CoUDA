# CoUDA

CoUDA 是一个针对跨域故障诊断增量学习的 PyTorch 框架，核心入口为 [`trainer.trainer.train`](trainer/trainer.py)，可在单域预训练后适配新域。

## 仓库结构

- `main.py`：参数解析、随机种子、调用 `train`、汇总指标。
- `trainer/`：基座训练 `base_train` 与增量训练 `incremental_train`。
- `models/`：ResNet 主干及余弦/欧式分类头。
- `utils.py`：数据、优化器、评估、特征相关工具。

## 数据集结构

- `.mat` 数据置于 `data/`，由 [`dataloader_domain.dataloader`](dataloader_domain.py) 加载。
- `data`：形状 $(N,1024)$，依 `--data_dimension` 重塑。
- `label`：长度 $N$ 的整型标签。
- 样本按 域→类→100 样本 排列，索引满足 $d \times \text{nb\_cl} \times 100$ 偏移。
- `--Domain_Seq` 控制会话顺序，`--nb_shot` 与 `--nb_exemplar` 决定少样本与回放数量。

## 快速开始

1. 环境：
   ```sh
   conda create -n couda python=3.9
   conda activate couda
   pip install -r requirements.txt
   ```
2. 将 `.mat` 文件放入 `data/` 并确保字段与结构符合要求。
3. 运行：
   ```sh
   python main.py \
   ```
   - `--incremental_mode fine_tuning|single|ours`
   - `--classifer fc|cos|eu`
   - `--nb_exemplar`、`--random_exemplar`

## 训练流程

- 会话 0：依次调用 [`set_dataset`](utils.py)、[`set_optimizer`](utils.py)，然后执行 [`base_train`](trainer/base_train.py)。
- 增量会话：按需选择微调或 [`incremental_train`](trainer/incremental_train.py)，可借助 `set_exemplar` 进行样本筛选。
- 评估：`evaluate` 输出逐域准确率与混淆矩阵，并在启用 `--save_model` 时保存模型与特征。
- 指标：训练结束后打印 $AF$、$AMF$、$AG$、$AA$、$BWT$、$ACC$ 等指标。

## 日志与结果

模型与特征保存于：
```
log/<dataset>/<incremental_mode>/<preprocess>/<train_list>_<backbone>_<...>/<文件名>.pth
```
启用 `--save_model` 将生成 `Result.csv` 记录各会话准确率。

## 参考资料

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
