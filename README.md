# Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincaré Ball, CVPR 2024

Welcome to the official page of the paper [Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincaré Ball (CVPR 2024)](https://arxiv.org/pdf/2404.03778.pdf). You can find a video presentation [here](https://www.youtube.com/watch?v=HDrPU6LCs1w&t=2s).

## Open-Source Implementation

The official implementation will be available soon.

## Abstract

Hierarchy is a natural representation of semantic taxonomies, including the ones routinely used in image segmentation. Indeed, recent work on semantic segmentation reports improved accuracy from supervised training leveraging hierarchical label structures. Encouraged by these results, we revisit the fundamental assumptions behind that work. We postulate and then empirically verify that the reasons for the observed improvement in segmentation accuracy may be entirely unrelated to the use of the semantic hierarchy. To demonstrate this, we design a range of cross-domain experiments with a representative hierarchical approach. We find that on the new testing domains, a flat (non-hierarchical) segmentation network, in which the parents are inferred from the children, has superior segmentation accuracy to the hierarchical approach across the board. Complementing these findings and inspired by the intrinsic properties of hyperbolic spaces, we study a more principled approach to hierarchical segmentation using the Poincaré ball model. The hyperbolic representation largely outperforms the previous (Euclidean) hierarchical approach as well and is on par with our flat Euclidean baseline in terms of segmentation accuracy. However, it additionally exhibits surprisingly strong calibration quality of the parent nodes in the semantic hierarchy, especially on the more challenging domains. Our combined analysis suggests that the established practice of hierarchical segmentation may be limited to in-domain settings, whereas flat classifiers generalize substantially better, especially if they are modeled in the hyperbolic space.

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{weber2024flattening,
  title={Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincar{\'e} Ball},
  author={Weber, Simon and Z{\"o}ng{\"u}r, Bar{\i}{\c{s}} and Araslanov, Nikita and Cremers, Daniel},
  journal={arXiv e-prints},
  pages={arXiv--2404},
  year={2024}
}
```
