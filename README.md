# Cycle-W-GAN

An experimental TensorFlow implementation of **CycleGAN trained with the Wasserstein objective**: the cross-entropy discriminators of [CycleGAN (Zhu et al., ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) are replaced by critics that estimate the Wasserstein distance ([WGAN, Arjovsky et al., ICML 2017](https://proceedings.mlr.press/v70/arjovsky17a.html)), combined with the usual L1 cycle-consistency loss.

> **Status: research prototype.** The training scaffold (a few hundred steps, no checkpointing, no sample dumping) was set up for experimentation and never finished; notably, the critics have **no Lipschitz constraint** (no weight clipping or gradient penalty), which standard WGAN training would require. Treat this as a starting point, not a working model.

## Design

- `GAN_g.py` — the two generators `gA: A→B` and `gB: B→A`, fully **ResNet-style**: a strided stem, four bottleneck **down-sample blocks**, then five bottleneck **up-sample blocks** whose residual *and* shortcut branches both use transposed convolutions (`res_utils.up_sample_block`) — the "ResNet upsampling block" this repo was built to try out.
- `GAN_f.py` — the two critics `fA` / `fB`: ResNet bottleneck downsampling to a global-average-pooled scalar score (no sigmoid — Wasserstein critics).
- `main.py` — the losses and loop:
  - critics maximize `E[f(real)] − E[f(g(fake))]` for both domains;
  - generators minimize the two Wasserstein estimates plus `0.5 ×` L1 cycle loss (`A→B→A` and `B→A→B`);
  - RMSProp, 5 critic steps per generator step, batch 16.
- `generator.py` — loads the [CycleGAN `apple2orange` dataset](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/) into memory and yields random 128×128 pairs; set `DATASET_DIR` to your local copy.

## Run

Requires TensorFlow 1.x (`tf.contrib.slim`), OpenCV and NumPy.

1. Download an unpaired-translation dataset (e.g. `apple2orange`) and point `DATASET_DIR` in `generator.py` at it.
2. `python main.py`

## References

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}

@inproceedings{arjovsky2017wasserstein,
  title={Wasserstein Generative Adversarial Networks},
  author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2017}
}
```
