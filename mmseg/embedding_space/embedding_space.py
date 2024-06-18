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

import logging

from ..embedding_space.hyperbolic_embedding_space import HyperbolicEmbeddingSpace
from torch import nn


def EmbeddingSpace(offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
    """Returns correct embedding space obj based on config."""
    base = HyperbolicEmbeddingSpace


    class EmbeddingSpace(base):
        def __init__(self,offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
            super().__init__( offsets, normals, curvature, train=train, prototype_path=prototype_path)

    return EmbeddingSpace(offsets, normals, curvature, train=train, prototype_path=prototype_path)
