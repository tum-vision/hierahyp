"""
Embedding space. Largely copied from the 
Hyperbolic Image Segmentation (Atigh et al., 2022) author implementation:
https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation

Please consider citing this work:
@article{ghadimiatigh2022hyperbolic,
  title={Hyperbolic Image Segmentation},
  author={GhadimiAtigh, Mina and Schoep, Julian and Acar, Erman and van Noord, Nanne and Mettes, Pascal},
  journal={arXiv preprint arXiv:2203.05898},
  year={2022}
}
"""

import os

import numpy as np

from .abstract_embedding_space import AbstractEmbeddingSpace
from ..layers import hyp_mlr_torch
import torch
from torch import nn

import time

class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
        super().__init__(offsets, normals, curvature, train=train, prototype_path=prototype_path)
             

    def logits_torch(self, embeddings, offsets, normals, curvature):
        return hyp_mlr_torch(
            embeddings,
            c=curvature,
            P_mlr=offsets,
            A_mlr=normals
        )        

