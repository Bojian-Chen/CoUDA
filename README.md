# CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis

[![Paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/document/10896871)

This is a PyTorch implementation of the paper: [**CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis Under Dynamic Working Conditions**](https://ieeexplore.ieee.org/document/10896871).

> **⚠️ Important Note:** The link to the [supplementary material](./pdf/SI.pdf) in the [main paper PDF](./pdf/CoUDA.pdf) is incorrect on IEEE Xplore. Please refer to the files in this repository for the correct supplementary material.

---

## 📁 Dataset Layout

**⚠️ SDUST Dataset Notice:** The SDUST dataset used in this study is **not an open-source dataset**. It is an older version of the open-source edition that employed different bearing models and fewer fault types, making it incompatible with the publicly available version.

### Data Structure Requirements

- All `.mat` files should be placed under the `data/` directory
- Files are loaded by [`dataloader_domain.dataloader`](dataloader_domain.py)
- The `.mat` files must follow this structure:

**Sample Organization:**  
Samples are organized as: **domain → class → samples**

**Example:**  
If there are 6 domains and 10 classes (with 100 samples per class as default):
- First 100 samples: domain 0, class 0
- Next 100 samples: domain 0, class 1
- And so on...

### Configuration

- Use the `--Domain_Seq` parameter to set the session order

---

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a new conda environment:

```sh
conda create -n couda
conda activate couda
pip install -r requirements.txt
```

### 2. Data Preparation

- Place your `.mat` files in the `data/` directory
- Ensure the file structure follows the layout described above

### 3. Configuration Settings

Set domain settings and other parameters in `main.py`. Modify the domain settings following this template:

```python
if args.dataset_name == 'SK':
   args.train_list = './SK_all_10classes_train.mat' 
   args.test_list = './SK_all_10classes_test.mat'
   args.Domain_Seq = np.array([0,1,2,3,4,5])  # domain order
   args.nb_session = len(args.Domain_Seq)
   args.nb_cl = 10
```

### 4. Run the Code

```sh
python main.py
```

---

## 📖 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@ARTICLE{10896871,
  author={Chen, Bojian and Zhang, Xinmin and Shen, Changqing and Li, Qi and Song, Zhihuan},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis Under Dynamic Working Conditions}, 
  year={2025},
  volume={21},
  number={5},
  pages={4072-4082},
  keywords={Adaptation models;Fault diagnosis;Employee welfare;Data privacy;Prototypes;Data models;Representation learning;Measurement;Contrastive learning;Training;Catastrophic forgetting;continual learning;contrastive learning;domain adaptation (DA);fault diagnosis;industrial big data},
  doi={10.1109/TII.2025.3538135}}
```

---

## 📄 License & Contact

For questions or issues, please open an issue in this repository.