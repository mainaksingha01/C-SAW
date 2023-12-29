# C-SAW: Self-Supervised Prompt Learning for Image Generalization in Remote Sensing

Official repository of C-SAW, a vision-language models (VLM) for unknown class and domain generalization using *self-supervised learning* in Remote Sensing.

## **ICVGIP 2023 (Best Paper Award)**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2311.15812)

## Abstract

![teaser](https://github.com/mainaksingha01/C-SAW/blob/master/images/teaser.png)

We focus on domain and class generalization problems in analyzing optical remote sensing images, using the large-scale pre-trained vision-language model (VLM), CLIP. While contrastively trained
VLMs show impressive zero-shot generalization performance, their effectiveness is limited when dealing with diverse domains during training and testing. Existing prompt learning techniques overlook
the importance of incorporating domain and content information into the prompts, which results in a drop in performance while dealing with such multi-domain data. To address these challenges, we
propose a solution that ensures domain-invariant prompt learning while enhancing the expressiveness of visual features. We observe that CLIP’s vision encoder struggles to identify contextual image
information, particularly when image patches are jumbled up. This issue is especially severe in optical remote sensing images, where land-cover classes exhibit well-defined contextual appearances. 
To this end, we introduce C-SAW, a method that complements CLIP with a self-supervised loss in the visual space and a novel prompt learning technique that emphasizes both visual domain and contentspecific 
features. We keep the CLIP backbone frozen and introduce a small set of projectors for both the CLIP encoders to train C-SAW contrastively. Experimental results demonstrate the superiority of C-SAW across 
multiple remote sensing benchmarks and different generalization tasks.

## Architecture

![architecture](https://github.com/mainaksingha01/C-SAW/blob/master/images/csaw.png)

C-SAW utilizes CLIP’s frozen visual and text encoder backbones. The visual attentive token generator (GVAT) generates M visual attentive tokens using intermediate layers (IL) of the source domains S. 
These visual attentive tokens, along with context and class tokens, create text embeddings, forming the visual attentive text prompting (VATP) approach.

## Datasets
- For Base-to-New Class and Cross-Dataset Generalization:
  - [PatternNet](https://sites.google.com/view/zhouwx/dataset)
  - [RSICD](https://github.com/201528014227051/RSICD_optimal)
  - [RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45)
  - [MLRSNet](https://data.mendeley.com/datasets/7j9bv9vwsx/3)

## Released Datasets (Version-2):
- For Domain Generalization:
  - Download the datasets from [APPLeNet](https://github.com/mainaksingha01/APPLeNet)
 
## Code

 - `files` folder contains the dataloader files of each datasets.
 - `models` folder contains the code of our model.
 - Clone this repository [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) inside this repo for the metrices.
 - `scripts` folder holds the scripts of each of the generalization tasks both for training and testing.

```shell
$ cd scripts
$ bash base2new_train.sh patternnet 1
$ bash base2new_test.sh patternnet 1
$ bash crossdataset_train.sh patternnet 1
$ bash crossdataset_test.sh rsicd 1
$ bash domaingen_train.sh patternnetv2 1
$ bash domaingen_test.sh rsicdv2 1
```

## Results

### Base-to-New Class Generalization

![base2new](https://github.com/mainaksingha01/C-SAW/blob/master/images/base2new.png)

### Cross Dataset Generalization

![crossdataset](https://github.com/mainaksingha01/C-SAW/blob/master/images/crossdataset.png)

### Domain Generalization

![domaingen](https://github.com/mainaksingha01/C-SAW/blob/master/images/domaingen.png)

## Bibtex

Please cite the paper if you use our work . Thanks.

```
@article{bhattacharya2023c,
  title={C-SAW: Self-Supervised Prompt Learning for Image Generalization in Remote Sensing},
  author={Bhattacharya, Avigyan and Singha, Mainak and Jha, Ankit and Banerjee, Biplab},
  journal={arXiv preprint arXiv:2311.15812},
  year={2023}
}

@inproceedings{singha2023applenet,
  title={Applenet: Visual attention parameterized prompt learning for few-shot remote sensing image generalization using clip},
  author={Singha, Mainak and Jha, Ankit and Solanki, Bhupendra and Bose, Shirsha and Banerjee, Biplab},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgements

Our code is mainly based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [APPLeNet](https://github.com/mainaksingha01/APPLeNet).
