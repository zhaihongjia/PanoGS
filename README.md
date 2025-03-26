<p align="center">

  <h1 align="center"> PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding
  </h1>
  <p align="center">
    <a href="https://zhaihongjia.github.io/"><strong>Hongjia Zhai</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=vn89ztQAAAAJ&view_op=list_works&sortby=pubdate"><strong>Hai Li</strong></a>
    ·
    <a href="https://zju3dv.github.io/panogs/"><strong>Zhenzhe Li</strong></a>
    ·
    <a href="https://zju3dv.github.io/panogs/"><strong>Xiaokun Pan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=_0lKGnkAAAAJ&view_op=list_works&sortby=pubdate"><strong>Yijia He</strong></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/gfzhang/"><strong>Guofeng Zhang</strong></a>
  </p>

[comment]: <> (<h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2503.18107">Paper</a> | <a href="https://zju3dv.github.io/panogs/">Project Page</a></h3>
  <div align="center"></div>

  <a href="">
    <img src="https://raw.githubusercontent.com/zhaihongjia/open_access_assets/main/PanoGS/images/teaser.png" alt="gui" width="100%">
  </a>
</p>
<p align="center">
We present <a href="https://arxiv.org/abs/2503.18107">PanoGS</a>, a novel and effective 3D panoptic open vocabulary scene understanding approach. Technically, to learn accurate 3D language features that can scale to large indoor scenarios, we adopt the pyramid tri-plane to model the latent continuous parametric feature space and use a 3D feature decoder to regress the multi-view fused 2D feature cloud. Besides, we propose language-guided graph cuts that synergistically leverage reconstructed geometry and learned language cues to group 3D Gaussian primitives into a set of super-primitives. To obtain 3D consistent instance, we perform graph clustering based segmentation with SAM-guided edge affinity computation between different super-primitives.
</p>
<br>
 
## NOTES

The code is currently being reorganized and will be open sourced within two weeks.


## Acknowledgement
We sincerely thank the following excellent projects, from which our work has greatly benefited.
- [MonoGS](https://github.com/muskie82/MonoGS)
- [SAI3D](https://github.com/yd-yin/SAI3D)
- [MaskClustering](https://github.com/PKU-EPIC/MaskClustering)
- [OpenScene](https://github.com/pengsongyou/openscene)
- [SAM](https://github.com/facebookresearch/segment-anything)

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{panogs,
      title={{PanoGS}: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding},
      author={Zhai, Hongjia and Li, Hai and Li, Zhenzhe and Pan, Xiaokun and He, Yijia and Zhang, Guofeng},
      booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```