# Learn an Effective Lip Reading Model without Pains

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learn-an-effective-lip-reading-model-without/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=learn-an-effective-lip-reading-model-without)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learn-an-effective-lip-reading-model-without/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=learn-an-effective-lip-reading-model-without)


## Content

- [Introduction](#Introduction)
- [Benchmark](#Benchmark)
- [Dataset Preparation](#Dataset-Preparation)
- [How to test](#How-to-test)
- [How to train](#How-to-train)
- [Dependencies](#Dependencies)
- [Citation](#Citation)
- [License](#License)


## Introduction

This is the repository of [Learn an Effective Lip Reading Model without Pains](https://arxiv.org/abs/2011.07557). In this repository, we provide pre-trained models and training settings for deep lip reading. We evaluate our pipeline on [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14). We obtain **88.4%** and **55.7%** on LRW and LRW-1000, respectively. The results are comparable and even surpass current state-of-the-art results. **Especially, we reach the current state-of-the-art result (55.7%) on LRW-1000 Dataset.** Please note that this repository achieves a slightly higher performance than the initial arxiv version by further adjusting two training details, including increasing the training epochs and adjusting the horizontal flip.

## Benchmark

| Year |      Method          |   LRW  |     LRW-1000    |
|:----:|:--------------------:|:------:|:---------------:|
|2017|[Chung et al.](https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17a/chung17a.pdf)   | 61.1%  | - |
|2017|[Stafylakis et al.](https://arxiv.org/abs/1703.04105)   |83.5% | - |
|2018|[Stafylakis et al.](https://arxiv.org/abs/1811.01194)   |**88.8%** | - |
|2019|[Wang et al.](https://bmvc2019.org/wp-content/uploads/papers/1211-paper.pdf)   |83.3% | 36.9% |
|2019|[Weng et al.](https://arxiv.org/abs/1905.02540)   |84.1% | - |
|2020|[Luo et al.](https://arxiv.org/abs/2003.03983)   | 83.5% | 38.7% |
|2020|[Zhao et al.](https://arxiv.org/abs/2003.06439)   |84.4% | 38.7% |
|2020|[Zhang et al.](https://arxiv.org/abs/2003.03206)   |85.0% | 45.2% |
|2020|[Martinez et al.](https://arxiv.org/abs/2001.08702)   |85.3% | 41.4% |
|2020|[Ma et al.](https://arxiv.org/abs/2007.06504)   |87.7% |43.2%|
|**2020**|    **ResNet18 + BiGRU (Our Baseline)**  |   83.7%   |     46.5%    |
|**2020**|    **Our Method**  |   85.5%   |     48.3%    |
|**2020**|    **Our Method (with word boundary)**  |  88.4%    |     **55.7%**    |



## Dataset Preparation

1. Download  [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.htm) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14) and place `lrw_mp4` and `LRW1000_Public` in the root of this repository. 

2. You can run the  `scripts/prepare_lrw.py` and `scripts/prepare_lrw1000.py` to generate training samples of LRW and LRW-1000 Dataset respectively:

```
python scripts/prepare_lrw.py
python scripts/prepare_lrw1000.py 
```

The mouth videos will be saved in the `.pkl` format

## How to test

Link of pretrained weights:
[Baidu Yun](https://pan.baidu.com/s/1mkKPmW2ezuctyoHwPNUkog)(code: 1hpa), [Google Drive](https://drive.google.com/drive/folders/115bAZXOZqsBJWvX28HjXRwEhNJzJlq88?usp=sharing)

If you can not access to Baidu Yun, please email dalu.feng@vipl.ict.ac.cn or fengdalu@gmail.com

To test our provided weights, you should download weights and place them in the root of this repository. 

For example, to test baseline on LRW Dataset: 

```
python main_visual.py \
    --gpus='0'  \
    --lr=0.0 \
    --batch_size=128 \
    --num_workers=8 \
    --max_epoch=120 \
    --test_interval=1.0 \
    --test=True \
    --save_prefix='checkpoints/lrw-baseline/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=False \
    --mixup=False \
    --label_smooth=False \
    --se=False \
    --weights='checkpoints/lrw-baseline-acc-0.83772.pt'
```

To test our model in LRW-1000 Dataset: 

```
python main_visual.py \
    --gpus='0'  \
    --lr=0.0 \
    --batch_size=128 \
    --num_workers=8 \
    --max_epoch=120 \
    --test_interval=1.0 \
    --test=True \
    --save_prefix='checkpoints/lrw-1000-final/' \
    --n_class=1000 \
    --dataset='lrw1000' \
    --border=True \
    --mixup=False \
    --label_smooth=False \
    --se=True \
    --weights='checkpoints/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.5578.pt'
```

## How to train

For example, to train lrw baseline:

```
python main_visual.py \
    --gpus='0,1,2,3'  \
    --lr=1e-3 \
    --batch_size=400 \
    --num_workers=8 \
    --max_epoch=120 \
    --test_interval=1.0 \
    --test=False \
    --save_prefix='checkpoints/lrw-baseline/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=False \
    --mixup=False \
    --label_smooth=False \
    --se=False  
```

Optional arguments:

- `gpus`: the GPU id used for training
- `lr`: learning rate
- `batch_size`: batch size
- `num_workers`: the number of processes used for data loading
- `max_epoch`: the maximum epochs in training
- `test_interval`: the number of epochs between testing, notice this number can be set lower than 1.0. (e.g. 0.5, 0.25, etc.)
- `test`: test mode
- `save_prefix`: the save prefix of pre-trained weights
- `n_class`: the number of total word classes
- `dataset`: the dataset used for training and testing, only `lrw` and `lrw1000` are supported.
- `border`:  use word boundary indicated variable for training and testing
- `mixup`: use mixup in training
- `label_smooth`: use label_smooth in training
- `se`: use se module in ResNet-18

More training details and setting can be found in [our paper](https://arxiv.org/abs/2011.07557). We plan to include more pretrained models in the future.

# Dependencies

- PyTorch 1.6
- opencv-python

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```
@article{ma2020towards,
  author       = "Feng, Dalu and Yang, Shuang and Shan, Shiguang and Chen, Xilin",
  title        = "Learn an Effective Lip Reading Model without Pains",
  journal      = "arXiv preprint arXiv:2011.07557",
  year         = "2020",
}
```


## License

The MIT License
