# Worst-group-main

This repository provides the code for my research [Worst-group Error Bounding by Boosting](https://ken.ieice.org/ken/paper/20241130dc5A/) in Pytorch.

We built on the implementation of [group DRO](https://github.com/kohpangwei/group_DRO), to which we add the
implementation of  ```Worst-group Error Bounding by Boosting```. Group DRO was featured on the paper:

> Shiori Sagawa*, Pang Wei Koh*, Tatsunori Hashimoto, and Percy Liang
>
> [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)


[Paper] | [Bibtex] | [Slides](./images/PRMU2024.png)

## Overveiw of Our Method

![Illustration](./images/overview.png)
> In machine learning-based classification, accuracy can vary across different groups. For example, in the classification of normal and abnormal medical images, the accuracy may differ significantly between Hospital A and Hospital B, where each hospital's data represents a distinct group. It is crucial to avoid situations where the accuracy for a particular group is significantly lower. In other words, it is necessary to mitigate the worst-group error. In practical applications, a decline in predictive performance for specific groups, such as those defined by gender or nationality, can pose serious issues. Therefore, this study aims to suppress the worst-group error to prevent the deterioration of predictive performance for certain groups.

## Requirements 
<!-- All codes are written by Python 3.7, and 'requirements.txt' contains required Python packages. -->
- python >= 3.8
- cuda & cudnn

### prerequisitions
- python 3.8.19
- matplotlib 3.7.5
- seaborn  0.13.2
- scikit-learn  1.3.2
- pandas 2.0.3
- Pillow 10.3.0
- torch  2.0.1
- torchvision 0.18.1py
- pytorch_transformers 1.2.0
- tqdm  4.66.4
- numpy 1.24.3


To install fast-setup of this code:

```setup
# pytorch install 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```



<!-- ## Dataset -->


## Training & Test

We provide several examples:

### Artificial dataset
---

- Ours (train)

```bash
python3 train.py --arch 'mlp' --dataset_type 'moon' --eps 0.0005 --gamma 0.1 --loss_type 'CE' --lr 0.01 --max_epoch 10000 --min_size 50 --num_classes 2 --root_log 'log' --root_model 'checkpoint' --seed 1 --store_name 'moon_1' --train_rule 'None'
```