# Learn an Effective Lip Reading Model without Pains

## Introduction

This is the repository of [Learn an Effective Lip Reading Model without Pains](). In this repository, we provide pre-trained models and training settings for deep lip reading. We train our model on [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14). We obtain **88.4%** and **55.7%** on LRW and LRW-1000, respectively. The results are comparable and even surpass current state-of-the-art results.

## Results

|      Method          |   LRW  |     LRW-1000    |
|:--------------------:|:------:|:---------------:|
|    ResNet18 + BiGRU  |   83.7%   |     46.5%    |
|    Ours              |  88.4%    |     55.7%    |



## Dataset Preparation

1. Download  [LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.htm) and [LRW1000 Dataset](https://vipl.ict.ac.cn/view_database.php?id=14) and place them in the root of this repository. 

2. We use pickle format to store our data for effecient training, you can run the  `scripts/` and `2020-06-06` to generate training samples of LRW and LRW-1000 Dataset respectively:

```
python scripts/prepare_lrw.py
python scripts/prepare_lrw1000.py 
```

## Testing

1. 