# Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincaré Ball, CVPR 2024

Welcome to the official page of the paper [Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincaré Ball (CVPR 2024)](https://arxiv.org/pdf/2404.03778.pdf). You can find a video presentation [here](https://www.youtube.com/watch?v=HDrPU6LCs1w&t=2s).


## Abstract

Hierarchy is a natural representation of semantic taxonomies, including the ones routinely used in image segmentation. Indeed, recent work on semantic segmentation reports improved accuracy from supervised training leveraging hierarchical label structures. Encouraged by these results, we revisit the fundamental assumptions behind that work. We postulate and then empirically verify that the reasons for the observed improvement in segmentation accuracy may be entirely unrelated to the use of the semantic hierarchy. To demonstrate this, we design a range of cross-domain experiments with a representative hierarchical approach. We find that on the new testing domains, a flat (non-hierarchical) segmentation network, in which the parents are inferred from the children, has superior segmentation accuracy to the hierarchical approach across the board. Complementing these findings and inspired by the intrinsic properties of hyperbolic spaces, we study a more principled approach to hierarchical segmentation using the Poincaré ball model. The hyperbolic representation largely outperforms the previous (Euclidean) hierarchical approach as well and is on par with our flat Euclidean baseline in terms of segmentation accuracy. However, it additionally exhibits surprisingly strong calibration quality of the parent nodes in the semantic hierarchy, especially on the more challenging domains. Our combined analysis suggests that the established practice of hierarchical segmentation may be limited to in-domain settings, whereas flat classifiers generalize substantially better, especially if they are modeled in the hyperbolic space.

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{weber2024flattening,
  title={Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincar{\'e} Ball},
  author={Weber, Simon and Z{\"o}ng{\"u}r, Bar and Araslanov, Nikita and Cremers, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28223--28232},
  year={2024}
}
```

## Open-Source Implementation

Our implementation is partially based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). 

### Installing libraries

To avoid conflicts between libraries versions, we recommend the following installation:

```
conda create --name hierahyp python=3.8 -y
conda activate hierahyp
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
pip install -U openmim
mim install mmcv-full==1.7.0
pip install mmsegmentation terminaltables psutil geoopt
```

### OOD datasets


The OOD datasets have to be organized as follows:
```bash
├── dataset
    ├── img
    │   ├── dataset_0.jpg
    │   ├── dataset_1.jpg
    │   └── ...
    └── label
        ├── dataset_0.png
        ├── dataset_1.png
        └── ...
```

The script to prepare the datasets is ```preprocess_ood_datasets/prep_dataset.py```. You can set the following options:
* ```--dataset```: dataset name, including 'mapillary', 'idd', 'bdd', 'wilddash', 'acdc'.
* ```--root```: root folder for the dataset.
* ```--savedir```: directory to save.



### Training
We train on Cityscapes, organized following [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html). 

```
# with 2 GPUs

# DeepLabV3+ with Euclidean networks

tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_euclidean.py 2

# DeepLabV3+ with hyperbolic networks

tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hyperbolic.py 2

# OCRNet with Euclidean networks

tools/dist_train.sh configs/ocrnet/ocrnet_hr48_4xb2-80k_cityscapes-512x1024_euclidean.py 2

#OCRNet with hyperbolic networks

tools/dist_train.sh configs/ocrnet/ocrnet_hr48_4xb2-80k_cityscapes-512x1024_hyperbolic.py 2

```


### Models

We release the models, trained on Cityscapes, and used as pretrained weights for inferring OOD detection. 

| | Euclidean | Hyperbolic |
| :---: | :---: | :---: |
| DeepLabV3+ | [weights](https://drive.google.com/drive/folders/1E5mBO79Xqlo1K0JsVAAKf4zFalHxTeSA?usp=sharing) | [weights](https://drive.google.com/drive/folders/1XF7WmoREpJmlwXxdZejPpuK6CTKUlovC?usp=sharing)|
| OCRNet | [weights](https://drive.google.com/drive/folders/1YPheTlwh41RxUdIQGYQ5T6XkLhG4IQd9?usp=sharing) | [weights](https://drive.google.com/drive/folders/1qiz-nK-ONv3nuoZDY7x4psRhC8uGjtEF?usp=sharing) |

### Inference 

You can set the following options when calling ``` dist_test.sh ```.
* ``` --eval ```: accuracy metric. By default, ```--eval mIoU```.
* ``` --T ```: temperature scaling for ECE. By default,  ``` --T 1.0 ```.
* ``` --numclasses ```: number of classes. ```--numclasses 7``` for parent-level predictions, ```--numclasses 19``` for child-level predictions.
* ``` --numbins ```: number of bins for ECE. By default, ```--numbins 20```.
* ``` --infermode ```: by default, ```--infermode softmax```
* ``` --ecemode ```: calibration metric, ece or class-wise ece. By default, ```--ecemode cwece```.

For instance, with configuration configs/deeplabv3plus/ood_mapillary_eucli.py and pretrained weights pretrained/deeplab_eucli/iter_80000.pth:
```
tools/dist_test.sh configs/deeplabv3plus/ood_mapillary_eucli.py pretrained/deeplab_eucli/iter_80000.pth  1 --eval mIoU --T 1.0 --numclasses 7 --numbins 20 --infermode softmax --ecemode cwece
```






