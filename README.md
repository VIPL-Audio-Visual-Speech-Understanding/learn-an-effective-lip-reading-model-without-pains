# Learn an Effective Lip Reading Model without Pains

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learn-an-effective-lip-reading-model-without/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=learn-an-effective-lip-reading-model-without)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learn-an-effective-lip-reading-model-without/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=learn-an-effective-lip-reading-model-without)


## Content

- [Introduction](#Introduction)
- [Benchmark](#Benchmark)
- [Dataset Preparation](#Dataset-Preparation)
- [Pretrain Weights](#Pretrain-Weights)
- [How to test](#How-to-test)
- [How to train](#How-to-train)
- [Dependencies](#Dependencies)
- [Citation](#Citation)
- [License](#License)


## Introduction

This is the repository of [An Efficient Software for Building Lip Reading Models Without Pains](https://vipl.ict.ac.cn/uploadfile/upload/2021070711474245.pdf). In this repository, we provide a deep lip reading pipeline as well as pre-trained models and training settings. We evaluate our pipeline on [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14). We obtain **88.4%** and **56.0%** on LRW and LRW-1000, respectively. The results are comparable and even surpass current state-of-the-art results. **Especially, we reach the current state-of-the-art result (56.0%) on LRW-1000 Dataset.**

## Benchmark

| Year |      Method          |   LRW  |     LRW-1000    |
|:----:|:--------------------:|:------:|:---------------:|
|2017|[Chung et al.](https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17a/chung17a.pdf)   | 61.1%  | 25.7% |
|2017|[Stafylakis et al.](https://arxiv.org/abs/1703.04105)   |83.5% | 38.2% |
|2018|[Stafylakis et al.](https://arxiv.org/abs/1811.01194)   |**88.8%** | - |
|2019|[Yang et at.](https://arxiv.org/abs/1810.06990) | - | 38.19% |
|2019|[Wang et al.](https://bmvc2019.org/wp-content/uploads/papers/1211-paper.pdf)   |83.3% | 36.9% |
|2019|[Weng et al.](https://arxiv.org/abs/1905.02540)   |84.1% | - |
|2020|[Luo et al.](https://arxiv.org/abs/2003.03983)   | 83.5% | 38.7% |
|2020|[Zhao et al.](https://arxiv.org/abs/2003.06439)   |84.4% | 38.7% |
|2020|[Zhang et al.](https://arxiv.org/abs/2003.03206)   |85.0% | 45.2% |
|2020|[Martinez et al.](https://arxiv.org/abs/2001.08702)   |85.3% | 41.4% |
|2020|[Ma et al.](https://arxiv.org/abs/2007.06504)   |87.7% |43.2%|
|**2020**|    **ResNet18 + BiGRU (Baseline + Cosine LR)**  |   85.0%   |     47.1%    |
|**2020**|    **ResNet18 + BiGRU (Baseline with word boundary + Cosine LR)**  |   87.5%   |     55.0%    |
|**2020**|    **Our Method**  |   86.2%   |     48.3%    |
|**2020**|    **Our Method (with word boundary)**  |  88.4%    |     **56.0%**    |


## Dataset Preparation

1. Download [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.htm) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14) and link `lrw_mp4` and `LRW1000_Public` in the root of this repository:

```
ln -s PATH_TO_DATA/lrw_mp4 .
ln -s PATH_TO_DATA/LRW1000_Public .
```

2. Run `scripts/prepare_lrw.py` and `scripts/prepare_lrw1000.py` to generate training samples of LRW and LRW-1000 Dataset respectively:

```
python scripts/prepare_lrw.py
python scripts/prepare_lrw1000.py 
```

The mouth videos, labels, and word boundary information will be saved in the `.pkl` format. We pack image sequence as `jpeg` format into our `.pkl` files and decoding via [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG). If you want to use your own dataset, you may need to modify the `utils/dataset.py` file.

## Pretrain Weights

We provide pretrained weight on LRW/LRW-1000 dataset for evaluation. For smaller datasets, the pretrained weights can be provide a good start point for feature extraction, finetuning, and so on.

Link of pretrained weights: [Baidu Yun](https://pan.baidu.com/s/1FuOKEX_ZI7QgobQmZdmq8A) (code: ivgl)

If you can not access to provided links, please email dalu.feng@vipl.ict.ac.cn or fengdalu@gmail.com.

## How to test

To test our provided weights, you should download weights and place them in the root of this repository. 

For example, to test baseline on LRW Dataset: 

```
python main_visual.py \
    --gpus='0'  \
    --lr=0.0 \
    --batch_size=128 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=True \
    --save_prefix='checkpoints/lrw-baseline/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=False \
    --mixup=False \
    --label_smooth=False \
    --se=False \
    --weights='checkpoints/lrw-cosine-lr-acc-0.85080.pt'
```

To test our model in LRW-1000 Dataset: 

```
python main_visual.py \
    --gpus='0'  \
    --lr=0.0 \
    --batch_size=128 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=True \
    --save_prefix='checkpoints/lrw-1000-final/' \
    --n_class=1000 \
    --dataset='lrw1000' \
    --border=True \
    --mixup=False \
    --label_smooth=False \
    --se=True \
    --weights='checkpoints/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt'
```

## How to train

For example, to train lrw baseline:

```
python main_visual.py \
    --gpus='0,1,2,3'  \
    --lr=3e-4 \
    --batch_size=400 \
    --num_workers=8 \
    --max_epoch=120 \
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
- `lr`: learning rate. By default, we automatically applied the [Linear Scale Rule](https://arxiv.org/abs/1706.02677) in code (e.g., lr=3e-4 for 4 GPUs x 32 video/gpu and lr=1.2e-3 for 8 GPUs x 128 video/gpu). We recommend lr=3e-4 for 32 video/gpu when training from scratch. You need to modify the learning rate based on your setting.
- `batch_size`: batch size
- `num_workers`: the number of processes used for data loading
- `max_epoch`: the maximum epochs in training
- `test`: The test mode. When using this mode, the program will only test once and exit.
- `weights`(optional): The path of pre-trained weight. If this option is specified, the model will load the pre-trained weights by the given location.
- `save_prefix`: the save prefix of model parameters
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
- TurboJPEG and [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG)

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```
@inproceedings{feng2021efficient,
  title={An Efficient Software for Building LIP Reading Models Without Pains},
  author={Feng, Dalu and Yang, Shuang and Shan, Shiguang},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--2},
  year={2021},
  organization={IEEE}
}
@article{feng2020learn,
  author       = "Feng, Dalu and Yang, Shuang and Shan, Shiguang and Chen, Xilin",
  title        = "Learn an Effective Lip Reading Model without Pains",
  journal      = "arXiv preprint arXiv:2011.07557",
  year         = "2020",
}
```


## License

The MIT License
