   # CoUDA
This is a PyTorch implementation of [CoUDA: Continual Unsupervised Domain Adaptation for Industrial Fault Diagnosis Under Dynamic Working Conditions](https://ieeexplore.ieee.org/document/10896871).
Note that the link of supplementary material in the main paper PDF is incorrect, please check it in `/pdf` for more details.


## Dataset Layout
**Please note that the SDUST dataset used in this study is not an open-source dataset.** It is merely an older version of an open-source dataset that employed different bearing models and fewer failures. Due to copyright restrictions, the original data cannot be provided. 

- `.mat` files under `data/`, loaded by [`dataloader_domain.dataloader`](dataloader_domain.py).
- You should ensure that the fields in the `.mat` files follow the layout below:
Samples follow domain → class → samples. 
For example, if there are 6 domains and 10 classes, the first 100 samples (We set 100 samples per class as default) belong to domain 0 class 0, the next 100 samples belong to domain 0 class 1, and so on. Each sample is indexed by the offset rule:  $d$ × `args.nb_cl` × 100 + $c$ × 100 + $s$, where $d$ is the domain index, $c$ is the class index, and $s$ is the sample index within that class. The dimension of `data` is (1,1024) by default, and will be reshaped to (1,32,32).
- `--Domain_Seq` sets session order; 


## Quick Start

1. Environment:
   ```sh
   conda create -n couda
   conda activate couda
   pip install -r requirements.txt
   ```
2. Place `.mat` files in `data/` and ensure their fields follow the layout above.
3. Set domain settings and other parameters in `main.py`.  Please modify the domain settings imitating the following code:
   ```
   if args.dataset_name == 'SK':
      args.train_list = './SK_all_10classes_train.mat' 
      args.test_list = './SK_all_10classes_test.mat'
      args.Domain_Seq = np.array([0,1,2,3,4,5])  # domain order
      args.nb_session = len(args.Domain_Seq)
      args.nb_cl = 10
   ```

4. Run:
   ```
   python main.py 
   ```


## References
If you find this code useful in your research, please consider citing the following paper:
```
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
```