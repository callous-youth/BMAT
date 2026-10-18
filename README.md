# BMAT

Official code for the ECCV 2026 paper *Learning with Bilevel-Minimax
Optimization for Efficient and Reliable Transfer Attacks*.

[Official Paper](https://link.springer.com/chapter/10.1007/978-3-032-37196-6_37) | [arXiv](https://arxiv.org/abs/2608.11815)

This release contains the ImageNet classification implementation of **Fast
BMAT**. It implements Phase-I SWM/IGA trajectory optimization followed by the
standard Phase-II transfer attack. The exploratory Full-BMAT variant and
implementations of external baselines are intentionally not included.

**Fast BMAT has been integrated into [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack), a framework that brings together a wide range of transfer-based adversarial attacks.**

## Environment

The experiments were run with Python 3.10, PyTorch 2.0.0, and CUDA 11.8.

```bash
conda create -n bmat python=3.10 -y
conda activate bmat
pip install -r requirements.txt
```

## Data and weights

Prepare the ILSVRC 2012 validation set using the class-subdirectory layout:

```text
imagenet-val/
  n07880968/ILSVRC2012_val_00001822.JPEG
  ...
```

The 5,000-image ResNet-50 split used in the paper is versioned at
`splits/selected_imagenet_resnet50.csv`.

Do not commit data or weights. Create a local `checkpoints/` directory and
place the following files there:

- `inception_v3_google-1a9a5a14.pth` for the auxiliary-surrogate setting.
- `resnet50_morebayesian_attack.pt` for the single-surrogate setting and MBA
  base attacker.
- The remaining victim-model weights required by `--attack_exist` evaluation.

Torchvision and timm victims download their public pretrained weights on first
use. The custom checkpoints are not redistributed here; obtain them from their
original sources under their respective licenses.

## Reproducing Fast BMAT

The bundled entrypoint keeps the Fast BMAT update order unchanged. By default,
the outer pseudo-surrogate is pretrained Inc-v3, reproducing the
auxiliary-surrogate setting. Pass `--use_single_surrogate` to instead use a
sampled Bayesian ResNet-50 subnet for the strict single-surrogate setting.

The following command uses Fast BMAT with the default Inc-v3 pseudo-surrogate.

```bash
python bmat/bmat_attack.py \
  --method di_mifgsm_fgsm \
  --epsilon 0.03 --alpha 0.006 --niters 10 \
  --meta_steps 3 --inner_loop 10 --attack_lr 2.0 \
  --first_order \
  --batch_size 25 --device_id 0 \
  --imagenet_val_dir /path/to/imagenet-val \
  --state_dict_dic checkpoints \
  --save_dir outputs/di_mi_inception_v3
```

For the strict single-surrogate setting, add `--use_single_surrogate` and use
the MBA inner surrogate:

```bash
python bmat/bmat_attack.py \
  --method di_mba_mifgsm_fgsm \
  --use_single_surrogate \
  --epsilon 0.03 --alpha 0.006 --niters 10 \
  --meta_steps 3 --inner_loop 10 --attack_lr 2.0 \
  --first_order \
  --batch_size 25 --device_id 0 \
  --imagenet_val_dir /path/to/imagenet-val \
  --state_dict_dic checkpoints \
  --mba_checkpoint_path checkpoints \
  --save_dir outputs/di_mi_mba_single
```

`--method` composes the base attackers and transformations evaluated in the
main classification table, including `pgd`, `mifgsm`, `vmifgsm`, `di`, `ti`,
`si`, `sgm`, `ghost`, and `mba`. The published setting is untargeted.

`--mba_task_use_ensemble` is only relevant with `--use_single_surrogate`; by
default this mode uses the final subnet from six MBA samples.

The default split is `splits/selected_imagenet_resnet50.csv`. Pass
`--selected_images_csv /path/to/split.csv` to use a different selected subset.

To evaluate generated examples on the 10 ImageNet victim models, rerun the
same command with `--attack_exist`. Generation writes `batch_*.npy` and
`labels.npy` under `--save_dir`; these artifacts are ignored by git.

For a quick smoke test, append `--max_batches 2`. Omit it for the full
5,000-image split.

## Citation

```bibtex
@inproceedings{liu2026learning,
  title={Learning with Bilevel-Minimax Optimization for Efficient and Reliable Transfer Attacks},
  author={Liu, Yaohua and Guo, Yifan and Gao, Jiaxin},
  booktitle={European Conference on Computer Vision},
  pages={680--701},
  year={2026},
  organization={Springer},
  doi={10.1007/978-3-032-37196-6_37}
}
```
