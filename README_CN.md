# CoUDA

一个用于故障诊断的跨域增量学习的 PyTorch 实验框架。主训练入口为 [`trainer.trainer.train`](trainer/trainer.py)，可在单域预训练后对新域进行增量适配。

## 仓库结构

- `main.py`：参数解析、随机种子设置、调用 `train` 并统计增量指标。
- `trainer/`：包含基座训练 [`trainer.base_train.base_train`](trainer/base_train.py) 与增量训练 [`trainer.incremental_train.incremental_train`](trainer/incremental_train.py)。
- `models/`：提供 ResNet 主干及余弦/欧式分类头，详见 [`models.resnet32`](models/resnet32.py) 与 [`models.modified_linear`](models/modified_linear.py)。
- `utils.py`：数据加载、优化器调度、评估与特征处理工具。

## 数据集结构

数据以 `.mat` 文件存放于 `data/`，由 [`dataloader_domain.dataloader`](dataloader_domain.py) 读取。文件需包含：

- `data`：形状 $(N,1024)$ 的浮点矩阵；根据 `--data_dimension` 选项重构为 $(N,1,32,32)$ 或 $(N,1,1024)$。
- `label`：长度 $N$ 的整型向量。
- 样本按“域→类别→样本”顺序排布：每个域有 `nb_cl` 个类别，每类准确含 100 个样本。索引计算方式：
  $$
  \text{index} \in [d \times \text{nb\_cl} \times 100,\ (d+1) \times \text{nb\_cl} \times 100)
  $$
  其中 $d$ 为域编号。
- `--Domain_Seq` 指定训练会话顺序。基础会话使用 `session=0` 域，其余会话按序取新域样本，可选少样本或回放样本（`--nb_shot`、`--nb_exemplar` 控制）。

## 运行方式

1. 创建环境并安装依赖：
   ```sh
   conda create -n couda python=3.9
   conda activate couda
   pip install -r requirements.txt  # 请根据实际需求整理依赖
   ```
2. 将 `.mat` 数据放置于 `data/`，确保字段与排列符合上述结构。
3. 运行示例：
   ```sh
   python main.py \
   ```
   - `--incremental_mode fine_tuning|single|ours` 控制增量策略。
   - `--classifer fc|cos|eu` 切换分类头。
   - `--nb_exemplar` 与 `--random_exemplar` 管理样本回放。

## 训练流程概要

- 会话 0：`train` 调用 [`set_dataset`](utils.py) 与 [`set_optimizer`](utils.py)，随后执行基座训练 [`base_train`](trainer/base_train.py)。
- 后续会话：依据模式选择普通微调或增量训练 [`incremental_train`](trainer/incremental_train.py)，并可通过 `set_exemplar` 进行样本精选。
- 评估：`evaluate` 逐域计算准确率并输出混淆矩阵，同时保存特征与模型快照（若开启 `--save_model`）。
- 指标：脚本结束后自动给出 $AF$、$AMF$、$AG$、$AA$、$BWT$、$ACC$ 等增量学习指标。

## 日志与结果

模型与特征保存在：
```
log/<dataset>/<incremental_mode>/<preprocess>/<train_list>_<backbone>_<...>/<文件名>.pth
```
所有指标及分会话准确率可在开启 `--save_model` 时生成的 `Result.csv` 中查看。