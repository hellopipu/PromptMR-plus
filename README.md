# PromptMR+

[[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf)] [[Supplementary](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565-supp.pdf)] [[Slides](https://eccv.ecva.net/media/eccv-2024/Slides/1057_OoLoVIz.pdf)] [[Video](https://youtu.be/8h0LNcypqYk?si=jgxaDXqoKGm35P9g)]

This repository contains the official implementation of the paper *Rethinking Deep Unrolled Models for Accelerated MRI Reconstruction.* In this work, we propose effective gradient-based learning and memory-efficient sensitivity map estimation to enhance deep unrolled models for multicoil MRI reconstruction. PromptMR+ is an example of applying these simple yet efficient techniques to the deep unrolled model PromptMR.

<p align="center">
  <img src="assets/proposed.png" alt="Training GPU Memory on cc-brain dataset" width="600" />
</p>

## News

- **Jan 23, 2025:** **Fully** Reproducible code, model weights and reconstruction results were released.
- **Oct 10, 2024:** PromptMR+ secured 1st place in both tasks of the MICCAI CMRxRecon2024 challenge.
- **Aug 12, 2024:** Paper accepted as an Oral presentation at ECCV 2024 (2.3%).


## Model Weights

We provide the **model weights** and **reconstruction results** for PromptMR/PromptMR+ trained on the FastMRI-knee, FastMRI-brain, Calgary-Campinas-brain, CMRxRecon2023-cardiac and CMRxRecon2024-cardiac datasets, which can be downloaded [here](https://huggingface.co/hellopipu/PromptMR/tree/main).

<details>
  <summary>CC-brain Results</summary>

| Model         | Cas. | Trained on | Training Acc | PSNR/SSIM 5x | PSNR/SSIM 10x |
| ------------- | ---- | ---------- | ------------ | ------------ | ------------- |
| `PromptMR`  | 12   | train      | 5x and 10x   | 36.98/0.9496 | 34.32/0.9302  |
| `PromptMR+` | 12   | train      | 5x and 10x   | 37.32/0.9516 | 34.87/0.9350  |

> Note: test result is on a subset of official val set (10/20), results are better than those reported in the paper.

</details>

<details>
  <summary>CMRxRecon2023-cardiac Results</summary>

| Model         | Cas. | Trained on | Training Acc   | Cine LAX<br />PSNR/SSIM 10x | Cine SAX<br />PSNR/SSIM 10x | Mapping T1w<br />PSNR/SSIM 10x | Mapping T2w<br />PSNR/SSIM 10x |
| ------------- | ---- | ---------- | -------------- | --------------------------- | --------------------------- | ------------------------------ | ------------------------------ |
| `PromptMR`  | 12   | train      | 4x, 8x and 10x | 38.28/0.9560                | 39.18/0.9615                | 38.99/0.9661                   | 37.21/0.9622                   |
| `PromptMR+` | 12   | train      | 4x, 8x and 10x | 39.13/0.9605                | 39.99/0.9658                | 40.37/0.9719                   | 38.22/0.9670                   |

> Note: test result is on the official validation set.

</details>

<details>
  <summary>FastMRI-knee Results</summary>

| Model         | Cas. | Trained on | Training Acc | NMSE/PSNR/SSIM 4x   | NMSE/PSNR/SSIM 8x   |
| ------------- | ---- | ---------- | ------------ | ------------------- | ------------------- |
| `PromptMR`  | 12   | train      | 4x and 8x    | 0.0051/39.71/0.9264 | 0.0080/37.78/0.8984 |
| `PromptMR+` | 12   | train      | 4x and 8x    | 0.0050/39.92/0.9276 | 0.0078/38.09/0.9012 |

> Note: test result is on a subset of official val set (100/199).

</details>

<details>
  <summary>CMRxRecon2024-cardiac Results</summary>

| Model         | Cas. | Trained on | Training Acc   | Task1 Avg<br />PSNR/SSIM | Task2 Avg<br />PSNR/SSIM |
| ------------- | ---- | ---------- | -------------- | ---------------------------- | ---------------------------- |
| `PromptMR`  | 12   | 83% train      | 4x~24x | 38.28/0.9560                 | 39.18/0.9615                 |
| `PromptMR+` | 12   | 83% train      | 4x~24x | 39.13/0.9605                 | 39.99/0.9658                 |

> Note: test result is on the split subset from the official training set (17%). (reported in the STACOM24 paper)

</details>

<details>
  <summary>FastMRI-Brain Results</summary>

| Model         | Cas. | Trained on | Training Acc | NMSE/PSNR/SSIM 4x   | NMSE/PSNR/SSIM 8x   |
| ------------- | ---- | ---------- | ------------ | ------------------- | ------------------- |
| `PromptMR`  | 12   | train+val  | 4x and 8x    | 0.0033/41.59/0.9609 | 0.0063/38.82/0.9465 |
| `PromptMR+` | 12   | train      | 4x and 8x    | 0.0031/41.84/0.9615 | 0.0055/39.46/0.9494 |

> Note: test result is on the official test set. (Not reported in the paper)

</details>

<details>
  <summary>Memory Performance</summary>
<table>
  <tr>
    <td align="center">
      <img src="assets/gpu_mem_train_cc_brain.png" alt="Training GPU Memory on cc-brain dataset" width="400" />
      <br><em>Training GPU Memory on cc-brain dataset.</em>
    </td>
    <td align="center">
      <img src="assets/gpu_mem_test_cmr23.png" alt="Test GPU Memory on CMRxRecon2023 dataset" width="400" />
      <br><em>Test GPU Memory on CMRxRecon2023 dataset.</em>
    </td>
  </tr>
</table>
</details>

## Install

python>=3.8.16, torch>=2.3, lightning>=2.2.4, wandb>=0.17.0

## Data Preparation

Refer to [DATASET.md](DATASET.md) for details.

## Train

For example, train PromptMR+ on fastmri-knee dataset

```python
python main.py fit \
    --config configs/base.yaml \
    --config configs/model/pmr-plus.yaml \
    --config configs/train/pmr-plus/fm-knee.yaml 
```

Training on fastmri-knee requires at least 17G GPU memory if set `use_checkpoint` and `compute_sens_per_coil` in config file.

## Test

For exmaple, run inference of PromptMR+ on fastmri-knee dataset

```python
python main.py predict --config configs/inference/pmr-plus/fm-knee.yaml
```

## License

Non-Commercial Research [License](LICENSE.md).

## Citation

```bibtex
@inproceedings{xin2024rethinking,
  title={Rethinking Deep Unrolled Model for Accelerated MRI Reconstruction},
  author={Xin, Bingyu and Ye, Meng and Axel, Leon and Metaxas, Dimitris N},
  booktitle={European Conference on Computer Vision},
  pages={164--181},
  year={2024},
  organization={Springer}
}
```
