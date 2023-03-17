# SRFormer: Permuted Self-Attention for Single Image Super-Resolution
Yupeng Zhou <sup>1</sup>, Zhen Li <sup>1</sup>, Chun-Le Guo <sup>1</sup>, Song Bai <sup>2</sup>, Ming-Ming Cheng <sup>1</sup>, Qibin Hou <sup>1</sup>

<sup>1</sup>TMCC, School of Computer Science, Nankai University

<sup>2</sup>ByteDance, Singapore

---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

<p align="center"> <img width="1000" src="figs/simple_compare.png"> </p>

The official PyTorch implementation of SRFormer: Permuted Self-Attention for Single Image Super-Resolution
([arxiv]()). SRFormer achieves **state-of-the-art performance** in
- classical image SR
- lightweight image SR
- real-world image SR

---

> <b>Abstract</b>: In this paper, we introduce SRFormer, a simple yet effective Transformer-based model for single image super-resolution. We rethink the design of the popular shifted window self-attention, expose and analyze several characteristic issues of it, and present permuted self-attention
(PSA). PSA strikes an appropriate balance between the channel and spatial information for self-attention, allowing each Transformer block to build pairwise correlations within large windows with even less computational burden.
Our permuted self-attention is simple and can be easily applied to existing super-resolution networks based on Transformers. Without any bells and whistles, we show that our SRFormer achieves a 33.86dB PSNR score on the Urban100 dataset, which is 0.46dB higher than that of SwinIR but uses
fewer parameters and computations. We hope our simple and effective approach can serve as a useful tool for future research in super-resolution model design. Our code is publicly available at https://github.com/HVision-NKU/SRFormer.




## Contents
1. [Installation & Dataset](#installation--dataset)
2. [Training](#Training)
3. [Testing](#Testing)
4. [Results](#results)
5. [Pretrain Models](#pretrain-models)
5. [Citations](#citations)
6. [License and Acknowledgement](#License-and-Acknowledgement)

## Installation & Dataset
- python 3.8
- pyTorch >= 1.7.0

```bash
cd SRFormer
pip install -r requirements.txt
python setup.py develop
```
### Dataset
We use the same training and testing sets as SwinIR, the following datasets need to be downloaded for training.

| Task                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Training Set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Testing Set |
|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|     :---:         |
| classical image SR                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                               [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) or DIV2K +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images)                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Set5 + Set14 + BSD100 + Urban100 + Manga109  |
| lightweight image SR                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                               [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images)                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Set5 + Set14 + BSD100 + Urban100 + Manga109   |
| real-world image SR                                 | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [OST](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip) (10324 images for sky,water,grass,mountain,building,plant,animal) | RealSRSet+5images|


## Training





1. Please download the dataset corresponding to the task and place them in the folder specified by the training option in folder `/options/train/SRFormer`
2. Follow the instructions below to  train our SRFormer.
```bash
# train SRFormer for classical SR task
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_SRx2_scratch.yml
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_SRx3_scratch.yml
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_SRx4_scratch.yml
# train SRFormer for lightweight SR task
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_light_SRx2_scratch.yml
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_light_SRx3_scratch.yml
./scripts/dist_train.sh 4 options/train/SRFormer/train_SRFormer_light_SRx4_scratch.yml
```


## Testing



```bash
# test SRFormer for classical SR task
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_DF2Ksrx2.yml
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_DF2Ksrx3.yml
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_DF2Ksrx4.yml
# test SRFormer for lightweight SR task
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_light_DIV2Ksrx2.yml
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_light_DIV2Ksrx3.yml
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_light_DIV2Ksrx4.yml
```
## Results

We provide the results on classical image SR, lightweight image SR, realworld image SR. More results can be found in the [paper](). The visual results of SRFormer will upload to google drive soon.

<details>
<summary>classical image SR (click to expan)</summary>

- Results of Table 4 in the  paper

<p align="center">
  <img width="900" src="figs/classicalSR_1.png">
</p>

- Results of Figure 4 in the paper

<p align="center">
  <img width="900" src="figs/classicalSR_2.png">
</p>


</details>

<details>
<summary>lightweight image SR (click to expan)</summary>

- Results of Table 5 in the  paper

<p align="center">
  <img width="900" src="figs/lightweightSR_2.png">
</p>

- Results of Figure 5 in the  paper

<p align="center">
  <img width="900" src="figs/lightweightSR_1.png">
</p>

</details>

<details>
<summary>realworld image SR (click to expan)</summary>

- Results of Figure 8 in the  paper

<p align="center">
  <img width="900" src="figs/realworld.png">
</p>

</details>


## Pretrain Models

Pretrain Models can be download from [google drive](https://drive.google.com/drive/folders/1D5ER_HwYJoyZCcrKVstwE-iEl0hXulwd?usp=sharing).
To reproduce the results in the article, you can download them and put them in the `/PretrainModel` folder.
## Citations
You may want to cite:
```

```
---

## License and Acknowledgement
This project is released under the Apache 2.0 license. The codes are based on  [BasicSR](https://github.com/XPixelGroup/BasicSR), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), and [SwinIR](https://github.com/JingyunLiang/SwinIR). Please also follow their licenses. Thanks for their awesome works.
