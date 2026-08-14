import os
import gc
import copy
import torch
import torchvision.transforms as T

import torch.nn as nn
import argparse
import torchvision
import models as MODEL
from torch.backends import cudnn
import numpy as np
import  time
import csv
from utils import *
import higher
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import distributed
import pretrainedmodels
from tqdm import tqdm
import itertools
import timm
import random
import warnings
from ipdb import set_trace as st
from transferattack.utils import *
from transferattack.ensemble.mba import MBA_Tri_Tack
from transferattack.model_related.dra import DRA
from transferattack.model_related.awt import AWT,SAM
import torchvision.models as models
from transferattack.attack import Attack
warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MBA_CHECKPOINT_DIR = os.path.abspath(os.path.join(_REPO_ROOT, 'ckpt'))
_MBA_CKPT_NAME = 'resnet50_morebayesian_attack.pt'


def _resolve_mba_checkpoint_dir(path) -> str:
    """Use a directory only when it contains the expected MBA checkpoint."""
    fb = _DEFAULT_MBA_CHECKPOINT_DIR
    if path is None:
        return fb
    p = str(path).strip()
    if not p:
        return fb
    cand = os.path.abspath(p)
    pt = os.path.join(cand, _MBA_CKPT_NAME)
    if os.path.isfile(pt):
        return cand
    return fb


def _default_imagenet_val_dir(model_name: str) -> str:
    """Return the default ImageNet subset directory for the selected backbone."""
    _leaf = {
        'resnet50': 'val_selected_5000_resnet50',
        'vgg19_bn': 'val_selected_5000_vgg19_bn',
    }.get(model_name, 'val_selected_5000_resnet50')
    return os.path.join(_REPO_ROOT, 'datasets', 'imagenet', _leaf)


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=str, default='16/255')
parser.add_argument('--sgm_lambda', type=float, default=1.0)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--pre_attack', type=int, default=5)
parser.add_argument('--global_amp_factor', type=float, default=10.0)
parser.add_argument('--ila_niters', type=int, default=10)
parser.add_argument('--method', type=str, default = 'fgsm')
parser.add_argument('--linbp_layer', type=str, default='3_1')
# parser.add_argument('--linbp_layer', type=int, default=23)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--ila_layer', type=str, default='2_3')
# parser.add_argument('--ila_layer', type=int, default=23)
parser.add_argument('--save_dir', type=str, default = 'data/imagenet/temp')
parser.add_argument('--target_attack', default=False, action='store_true')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--imagenet_val_dir', type=str, default=None,
                    help='ImageNet validation root; defaults to the selected-backbone subset.')
parser.add_argument('--selected_images_csv', type=str, default=None,
                    help='CSV defining the selected ImageNet subset; defaults to splits/selected_imagenet_<model_name>.csv.')
parser.add_argument('--mba_checkpoint_path', type=str, default=_DEFAULT_MBA_CHECKPOINT_DIR,
                    help='Directory containing resnet50_morebayesian_attack.pt; '
                         f'default: {_DEFAULT_MBA_CHECKPOINT_DIR}')
parser.add_argument('--use_single_surrogate', action='store_true', default=False,
                    help='Use a sampled MBA ResNet-50 subnet as the outer pseudo-surrogate; the default uses Inception-v3.')
parser.add_argument('--mba_use_ensemble', action='store_true', default=False,
                    help='For MBA methods, use the three-subnet EnsembleModel instead of the default final sampled subnet.')
parser.add_argument('--mba_task_use_ensemble', action='store_true', default=False,
                    help='Use three sampled MBA subnets to compute the outer loss; the default uses the final subnet from six samples.')
parser.add_argument('--mba_inner_standard_resnet', action='store_true', default=False,
                    help='Use standard normalized ResNet-50 as the MBA inner surrogate while retaining the MBA outer task model.')
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--state_dict_dic', type=str, default='ckpt/')
parser.add_argument('--if_distributed', type=int, default=0)
parser.add_argument('--alpha', type=str, default='1/255')
parser.add_argument('--portion', type=float, default=0.2)
parser.add_argument('--relu_silu_temperature', type=float, default=1.)
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument('--first_order', action='store_true', default=False,
                    help='Use the first-order trajectory approximation.')
parser.add_argument('--attack_exist', action='store_true', default=False,
                    help='Reuse previously generated adversarial examples when available.')
parser.add_argument('--inner_loop', type=int, default=2, help='number of inner loops for IAPTT')
parser.add_argument('--cg_steps', type=int, default=1)
parser.add_argument('--meta_steps', type=int, default=1)
parser.add_argument('--lamb', type=float, default=10.0)
parser.add_argument("--attack_lr", type=float, default=0.2, help="Tuning factor")
parser.add_argument('--max_batches', type=int, default=-1,
                    help='>=0: stop after this many batches; -1 = full dataset.')
# parser.add_argument('--beta', default=0.95, type=float,
#                     help="Max inner lr for cyclic scheduler. This is for cyclic scheduler only.")
# parser.add_argument('--temperature', default=6., type=float,
#                     help="Max inner lr for cyclic scheduler. This is for cyclic scheduler only.")
args = parser.parse_args()
# Treat an empty shell variable as an unspecified ImageNet directory.
if not args.imagenet_val_dir:
    args.imagenet_val_dir = None
_mba_raw = args.mba_checkpoint_path
args.mba_checkpoint_path = _resolve_mba_checkpoint_dir(args.mba_checkpoint_path)
if _mba_raw is not None and str(_mba_raw).strip() and os.path.abspath(str(_mba_raw).strip()) != args.mba_checkpoint_path:
    print(f"[WARN] --mba_checkpoint_path={_mba_raw!r} lacks {_MBA_CKPT_NAME}; using {args.mba_checkpoint_path}")
if not os.path.isabs(args.state_dict_dic):
    args.state_dict_dic = os.path.join(_REPO_ROOT, args.state_dict_dic.lstrip('./'))
if not str(args.state_dict_dic).endswith(os.sep):
    args.state_dict_dic = args.state_dict_dic + os.sep


class MBATriTackSurrogate(nn.Module):
    """Wrap MBA's Sequential(normalize, resnet) without treating p_cur as a layer."""

    def __init__(self, seq_norm_backbone):
        super().__init__()
        if not isinstance(seq_norm_backbone, nn.Sequential) or len(seq_norm_backbone) != 2:
            raise ValueError('MBA surrogate expects nn.Sequential with exactly 2 modules')
        self.add_module('norm', seq_norm_backbone[0])
        self.add_module('backbone', seq_norm_backbone[1])

    def forward(self, x):
        return self.backbone(self.norm(x))


def vec_to_grad(vec,model):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res

def distance_loss(params, init_prams):
    loss = 0
    for p,ip in zip(params,init_prams):
        loss += torch.norm(p - ip, 2) ** 2
    return loss

#######Backup#############
# def hv_prod(in_grad, x, params):
#     hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
#     hv = torch.nn.utils.parameters_to_vector(hv).detach()
#     # precondition with identity matrix
#     return hv/args.lamb + x
#
#
# def CG(in_grad, outer_grad, params, model, cg_steps):
#     x = outer_grad.clone().detach()
#     r = outer_grad.clone().detach() - hv_prod(in_grad, x, params)
#     p = r.clone().detach()
#     for i in range(cg_steps):
#         Ap = hv_prod(in_grad, p, params)
#         alpha = (r @ r) / (p @ Ap)
#         x = x + alpha * p
#         r_new = r - alpha * Ap
#         beta = (r_new @ r_new) / (r @ r)
#         p = r_new + beta * p
#         r = r_new.clone().detach()
#     return vec_to_grad(x,model)

def hv_prod(in_grad, x, params, lamb, retain_graph=True):
    hv = torch.autograd.grad(
        in_grad, params, retain_graph=retain_graph, grad_outputs=x)
    hv = torch.nn.utils.parameters_to_vector(hv).detach()
    # precondition with identity matrix
    return hv / lamb + x


def CG(in_grad, outer_grad, params, model, cg_steps, lamb):
    x = outer_grad.clone().detach()
    r = outer_grad.clone().detach() - hv_prod(
        in_grad, x, params, lamb, retain_graph=(cg_steps > 0))
    p = r.clone().detach()
    for i in range(cg_steps):
        last = i == cg_steps - 1
        Ap = hv_prod(in_grad, p, params, lamb, retain_graph=not last)
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new.clone().detach()
    return vec_to_grad(x, model)


def normalize(x, ms=None):
    if ms == None:
        ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - ms[0][i]) / ms[1][i]
    return x

def grad_inv(grad):
    return grad.neg()


def generate_pert(phi, return_entropy=False):
    adv_mean = phi[:, :3, :, :]
    adv_std = F.softplus(phi[:, 3:, :, :])
    rand_noise = torch.randn_like(adv_std)
    adv = torch.tanh(adv_mean + rand_noise * adv_std)
    # omit the constants in -logp
    negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
    entropy = negative_logp.mean()  # entropy
    if return_entropy:
        return adv, entropy
    else:
        return adv


def trans_all_224_no_normalize(x):
    return x.data


def trans_pnas(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    x = F.interpolate(x, size=(331,331))
    return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data


def trans_se(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    return normalize(x, ms = None).data


def trans_ori(x):
    if 'incep' in args.save_dir:
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
        return normalize(x, ms = None).data
    else:
        return normalize(x, ms = None).data


def test(model, trans, if_minus=False):
    target = torch.from_numpy(np.load(args.save_dir + '/labels.npy')).long()
    if args.target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()
        target = label_switch[target]
    img_num = 0
    count = 0
    advfile_ls = os.listdir(args.save_dir)
    for advfile_ind in range(len(advfile_ls)-1):
        adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
        if advfile_ind == 0:
            adv_batch_size = adv_batch.shape[0]
        img = adv_batch
        img_num += img.shape[0]
        label = target[advfile_ind * adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
            if if_minus:
                pred = pred -torch.tensor(1, dtype=torch.long)
        count += (label != pred.squeeze(0)).sum().item()
        del pred, img
        del adv_batch
    return round(100. - 100. * count / img_num, 2) if args.target_attack else round(100. * count / img_num, 2)

def DI_transform(x,diversity_prob=0.5,resize_rate=1.1):
    """
    Random transform the input images
    """
    # do not transform the input image
    if torch.rand(1) > diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    # resize the input image to random size
    rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

    # randomly add padding
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    # resize the image back to img_size
    return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

def SI_transform(x, num_scale=3):
    """
    Scale the input for SIM
    """
    return torch.cat([x / (2**i) for i in range(num_scale)])

def generate_kernel(self, kernel_type, kernel_size, nsig=3):
    """
    Generate the gaussian/uniform/linear kernel

    Arguments:
        kernel_type (str): the method for initilizing the kernel
        kernel_size (int): the size of kernel
    """
    if kernel_type.lower() == 'gaussian':
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
    elif kernel_type.lower() == 'uniform':
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    elif kernel_type.lower() == 'linear':
        kern1d = 1 - np.abs(
            np.linspace((-kernel_size + 1) // 2, (kernel_size - 1) // 2, kernel_size) / (kernel_size ** 2))
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
    else:
        raise Exception("Unspported kernel type {}".format(kernel_type))

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

def trans_incep(x):
    """Use 224-pixel input for Inception attack runs and 299-pixel input otherwise."""
    if 'incep' in args.save_dir:
        return normalize(x, ms=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data
    else:
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x[:, :, (256 - 224) // 2: (256 - 224) // 2 + 224, (256 - 224) // 2: (256 - 224) // 2 + 224]
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return normalize(x, ms=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data


def trans_ens_adv_inception_v3(x):
    """Ens3/Ens4 models require 299-pixel input for their auxiliary logits path."""
    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    x = x[:, :, (256 - 224) // 2: (256 - 224) // 2 + 224, (256 - 224) // 2: (256 - 224) // 2 + 224]
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return normalize(x, ms=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data

from scipy.ndimage import interpolation
from PIL import Image
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_MAX_SCALE = 1.1
_MIN_SCALE = 0.75
_VGG_MAX_SCALE = 0.8
_MAX_ANGLE = 4
_MAX_NOISE = 4

def inc_distort(image):
  image += np.random.randn(*image.shape).reshape(image.shape)*_MAX_NOISE/255.
  np.clip(image, -1, 1, out=image)
  angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
  if np.random.randint(2) == 0:
    angle = -angle

  rotated = interpolation.rotate(image, angle, reshape=True)
  zoom = (np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE), np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE), 1)
  zoomed = interpolation.zoom(rotated, zoom)

  starts = [(zoomed.shape[i] - image.shape[i]) // 2 for i in range(2)]
  ends = [starts[i] + image.shape[i] for i in range(2)]
  result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
  return result


def vgg_distort(image, out_shape):
  angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
  if np.random.randint(2) == 0:
    angle = -angle
  rotated = interpolation.rotate(image, angle, reshape=True)

  zoom = (np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE),
          np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE), 1)
  zoomed = interpolation.zoom(rotated, zoom)

  starts = [(zoomed.shape[i] - out_shape[i]) // 2 for i in range(2)]
  ends = [starts[i] + out_shape[i] for i in range(2)]
  result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
  return result


def jitter(num):
  scale = np.random.randint(95, 106) / 100.
  return num*scale


def transform_matrix_offset_center(matrix, x, y):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


def ens_distort(image):
  h = image.shape[0]
  w = image.shape[1]
  channel_axis = 2
  shear = jitter(np.pi/128)
  theta = jitter(-np.pi/128)
  tx = int(jitter(w*0.02))
  ty = int(jitter(h*0.02))
  zx = jitter(0.95)
  zy = jitter(0.95)

  # shear
  transform_matrix = np.array(
    [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  # shift
  transform_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
  image = apply_transform(image, transform_matrix, channel_axis)

  # zoom
  transform_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  # rotation
  transform_matrix = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  return image


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='constant',
                    cval=0.):
  """Apply the image transformation specified by a matrix.

  Arguments:
      x: 2D numpy array, single image.
      transform_matrix: Numpy array specifying the geometric transformation.
      channel_axis: Index of axis for channels in the input tensor.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.

  Returns:
      The transformed version of the input.
  """
  x = np.rollaxis(x, channel_axis, 0)
  final_affine_matrix = transform_matrix[:2, :2]
  final_offset = transform_matrix[:2, 2]
  channel_images = [
      interpolation.affine_transform(
          x_channel,
          final_affine_matrix,
          final_offset,
          order=0,
          mode=fill_mode,
          cval=cval) for x_channel in x
  ]
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, channel_axis + 1)
  return x


def transcode(image):
  im = Image.fromarray(image)
  im.save('/dev/shm/tmp.jpg', quality=50)
  transcoded = Image.open('/dev/shm/tmp.jpg')
  return np.asarray(transcoded)

batch_shape = [25, 224, 224, 3]
vgg_batch_shape = [25, 224, 224, 3]
def nips_test(model_list, ens_name_list):
    # trans_list=[trans_ori,trans_ori,trans_incep,trans_incep]
    trans_list = [trans_incep, trans_incep]
    target = torch.from_numpy(np.load(args.save_dir + '/labels.npy')).long()
    if args.target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()
        target = label_switch[target]
    img_num = 0
    count = 0
    weights = [[0.7, 0.1], [0.2, 0.1]]
    advfile_ls = os.listdir(args.save_dir)
    for advfile_ind in range(len(advfile_ls)-1):
        print('batch_'+str(advfile_ind)+'\n')
        logits=None
        adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
        if advfile_ind == 0:
            adv_batch_size = adv_batch.shape[0]
        img = adv_batch
        img_num += img.shape[0]
        label = target[advfile_ind * adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
        label = label.to(device)
        img = img.to(device)
        ens_images = np.zeros(batch_shape)
        inc_images = np.zeros(batch_shape)
        # vgg_images = np.zeros(batch_shape)
        adv_batch_rgb=adv_batch.transpose(1, 2).transpose(2, 3).cpu().numpy()
        for _ in range(adv_batch_size):
            image=adv_batch_rgb[_]
            # vgg_image = vgg_distort(image, vgg_batch_shape[1:3])
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = (image) * 2.0 - 1.0
            ens_images[_] =ens_distort(image)
            # Resize and mean subtract for VGG
            # vgg_image -= np.array((_R_MEAN, _G_MEAN, _B_MEAN)).reshape((1, 1, 3))
            # vgg_images[_] = vgg_image
            inc_images[_] = inc_distort(image)
        all_inputs=[[torch.from_numpy(ens_images).transpose(3, 2).transpose(2, 1).to(torch.float).to(device),img],[torch.from_numpy(inc_images).transpose(3, 2).transpose(2, 1).to(torch.float).to(device),img]]
        # for model_idx, model in enumerate(model_list[:2]):
        for idx, inputs in enumerate(all_inputs[0]):
            with torch.no_grad():
                pred = model_list[0](inputs)
                if logits == None:
                    logits = pred * weights[0][idx]
                else:
                    logits += pred * weights[0][idx]

        for idx, inputs in enumerate(all_inputs[1]):
            with torch.no_grad():
                pred = model_list[1](inputs)
                logits += pred * weights[1][idx]

        weight = 0.1
        result = model_list[2](trans_ori(img))
        logits += result * weight
        weight = 0.05
        result = model_list[3](trans_ori(img))
        logits += result * weight
        # predictions = torch.softmax(logits)
        with torch.no_grad ():
            pred = torch.argmax(logits, dim=1).view(1,-1)
        count += (label != pred.squeeze(0)).sum().item()
        del pred, img,ens_images,inc_images,all_inputs
        del adv_batch
    return round(100. - 100. * count / img_num, 2) if args.target_attack else round(100. * count / img_num, 2)

if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(0)

    os.makedirs(args.save_dir, exist_ok=True)
    epsilon = eval(args.epsilon)
    batch_size = args.batch_size
    method = args.method

    def _meta_in(t):
        """MBA subnets include ImageNet normalization; other meta models do not."""
        return t if ('mba' in method) else Normalize(t)

    ila_layer = args.ila_layer
    save_dir = args.save_dir
    niters = args.niters
    ila_niters = args.ila_niters
    target_attack = args.target_attack
    sgm_lambda = args.sgm_lambda
    device_id = args.device_id
    if args.imagenet_val_dir:
        imagenet_val_dir = args.imagenet_val_dir
    else:
        imagenet_val_dir = _default_imagenet_val_dir(args.model_name)
    if args.selected_images_csv:
        selected_images_csv = os.path.abspath(args.selected_images_csv)
    else:
        selected_images_csv = os.path.join(
            os.path.dirname(_REPO_ROOT), 'splits', 'selected_imagenet_{}.csv'.format(args.model_name))
    model_name=args.model_name
    state_dict_dic = args.state_dict_dic
    if_distributed = args.if_distributed
    linbp_layer = args.linbp_layer
    alpha = eval(args.alpha)
    portion = args.portion
    ReLU_SiLU_Function.temperature = args.relu_silu_temperature

    if if_distributed:
        args.local_run = 0
        device_id = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(device_id)
        print(f"=> set cuda device = {device_id}")

        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        print(f"local_run:{args.local_run}")
        if args.local_run == 0:
            print("==> start init_process_group")
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=24)
            )
        rank_id=int(os.environ["RANK"])
        world_size=int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_id))
    else:
        device = torch.device('cpu')
    print(device)

    if not args.attack_exist:
        print('Generate Adversarial images!')
        if args.max_batches >= 0:
            print(f'[max_batches] will stop after {args.max_batches} batches (smoke / subset).')
        trans = T.Compose([
            T.Resize((256,256)),
            T.CenterCrop((224,224)),
            T.ToTensor()
        ])
        dataset = SelectedImagenet(imagenet_val_dir=imagenet_val_dir,
                                   selected_images_csv=selected_images_csv,
                                   transform=trans
                                    )
        if if_distributed:
            my_sampler=distributed.DistributedSampler(
                    dataset, shuffle=False)
            ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=my_sampler, num_workers = 8, pin_memory = False)
        else:
            ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)


        task_model_list = []
        if args.use_single_surrogate:
            _mba_task_ens = MBA_Tri_Tack(
                epsilon=epsilon, alpha=alpha, epoch=niters, decay=1.0, targeted=False,
                random_start=True, norm='linfty', loss='crossentropy', device=device, attack='MBA',
                checkpoint_path=args.mba_checkpoint_path).load_model(
                    'resnet50', num_subnets=3 if args.mba_task_use_ensemble else 6)
            if args.mba_task_use_ensemble:
                for task_model in _mba_task_ens.models:
                    task_model.to(device)
                    task_model.eval()
                    task_model_list.append(task_model)
            else:
                # Use a distinct final subnet from six samples for the outer task.
                pseudo_model = _mba_task_ens.models[-1].to(device)
                pseudo_model.eval()
                task_model_list.append(pseudo_model)
            del _mba_task_ens
        else:
            pseudo_model = MODEL.inceptionv3.Inception3().to(device)
            pseudo_model.load_state_dict(torch.load(state_dict_dic + 'inception_v3_google-1a9a5a14.pth'))
            pseudo_model.eval()
            task_model_list.append(pseudo_model)
        if model_name=='vgg19_bn':
            if 'ghost' in method:
                meta_model = MODEL.vgg19_bn(pretrained=True)
            else:
                meta_model = torchvision.models.vgg19_bn(pretrained=True)
        elif model_name=='resnet50':
            if 'ghost' in method:
                meta_model = MODEL.resnet50(state_dict_dir=state_dict_dic+'resnet50-19c8e357.pth')
            else:
                # Match the standard ResNet-50 loading path used in the original experiments.
                meta_model = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError('Non-existent model name!')
        meta_model.eval()

        if 'sgm' in method:
            if model_name=='vgg19_bn':
                pass
            elif model_name=='resnet50':
                for i in range(1,3):
                    meta_model.layer1[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,4):
                    meta_model.layer2[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,6):
                    meta_model.layer3[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
                for i in range(1,3):
                    meta_model.layer4[i].conv1.register_full_backward_hook(resnet_weight_backward_hook)
            else:
                raise ValueError('Non-existent model name!')
        # batch norm 2d
        if 'batch_norm_2d' in method:
            meta_model.features[50]=MyBatchNorm2d(meta_model.features[50].weight.to(device), meta_model.features[50].bias, meta_model.features[50].running_mean.to(device), meta_model.features[50].running_var.to(device), meta_model.features[50].eps)

        if 'ti' in method:
            kernel = gkern(5, 3).astype(np.float32)
            gaussian_kernel = np.stack([kernel, kernel, kernel])
            gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
            gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)
        # if 'mba' in method:
        #     model=MBA_Tri_Tack(epsilon=epsilon,alpha=alpha,epoch=niters,decay=1.0, targeted=False,
        #          random_start=True,
        #          norm='linfty', loss='crossentropy', device=device, attack='MBA',
        #          checkpoint_path='D:/Code_2023/Code_new/BETAK/ckpt/').load_model('resnet50')
        #
        # meta_model.to(device)
        # if 'mba' not in method:
        #     model = nn.Sequential(
        #             Normalize,
        #             meta_model
        #         )
        # # else:
        # #     model=meta_model
        if 'mba' in method:
            _mba_ens = MBA_Tri_Tack(
                epsilon=epsilon, alpha=alpha, epoch=niters, decay=1.0, targeted=False,
                random_start=True, norm='linfty', loss='crossentropy', device=device, attack='MBA',
                checkpoint_path=args.mba_checkpoint_path).load_model('resnet50')
            if args.mba_use_ensemble:
                # load_model returns three wrapped subnets; EnsembleModel averages logits.
                meta_model = _mba_ens.to(device)
            else:
                del _mba_ens.models[:-1]
                # A raw Sequential would treat p_cur as a third layer.
                _mba_subnet = _mba_ens.models[-1].to(device)
                meta_model = MBATriTackSurrogate(_mba_subnet).to(device)

        if 'dra' in method:
            model=DRA(epsilon=epsilon,alpha=alpha,epoch=niters,decay=1.0, targeted=False,
                 random_start=True,
                 norm='linfty', loss='crossentropy', device=device, attack='DRA',
                 checkpoint_path=args.mba_checkpoint_path).load_model('resnet50')
            # del model.models[:-1]
            meta_model = model.to(device)
            model=model.to(device)
        # MBA subnets include normalization and must not be wrapped a second time.
        if 'dra' not in method:
            if 'mba' in method:
                model = meta_model
            else:
                model = nn.Sequential(
                    Normalize,
                    meta_model
                )
        if 'awt' in method:
            awt_attack = AWT(model_name='resnet50', epsilon=epsilon, alpha=alpha, epoch=niters, decay=1.0,
                             targeted=target_attack,
                             random_start=False,
                             norm='linfty', loss='crossentropy', device=device, attack='AWT')
        model.to(device)
        if target_attack:
            label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
        label_ls = []

        start_time = time.perf_counter()
        meta_iter_times = [[] for _ in range(args.meta_steps)]
        meta_iter_mem_peaks = [[] for _ in range(args.meta_steps)]
        inner_adapt_times = [[] for _ in range(args.meta_steps)]
        post_meta_times = []
        batch_wall_times = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for ind, (ori_img, label)in tqdm(enumerate(ori_loader)):
            if args.max_batches >= 0 and ind >= args.max_batches:
                break

            pmax_list = []
            time_bp_list=[]
            label_ls.append(label.detach().cpu().clone())
            if target_attack:
                label = label_switch[label]

            ori_img = ori_img.to(device)
            img = ori_img.clone()
            _t_batch0 = time.perf_counter()
            m = 0
            variance = 0
            number = 1#20
            beta = 1.5
            grad_t=0
            momentum,averaged_gradient=0,0

            for i in range(args.meta_steps):
                _t0_meta = time.perf_counter()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                if 'mdi2fgsm' in method:
                    if i == 0:
                        img_x = img
                    else:
                        img_x = input_diversity(img)
                elif 'pgd' in method and i==0:
                    img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
                elif 'nifgsm' in method:
                    img_x = img + alpha * 1 * m
                elif 'pifgsm' in method:
                    img_x = img + alpha * 1 * grad_t
                else:
                    img_x = img
                img_x.requires_grad_(True)

                img_x_temp=img_x

                
                if not ('admix' in method):
                    input_grad = 0
                    img_x.requires_grad_(True)
                    ###################1.3
                    loss_outer_list = []
                    loss_inner_list = []
                    ModelDelta = ModelTensorDelta(img_x).requires_grad_(True)
                    meta_model.p_cur = ModelDelta
                    if args.target_attack:
                        params = [
                            {'params': [param for param in list(meta_model.parameters()) if
                                        param is not ModelDelta()], 'lr': -args.attack_lr* 5e-5},#args.attack_lr* 4e-5},#
                            {'params': [meta_model.p_cur()], 'lr': args.attack_lr},
                        ]
                    else:
                        params = [
                            {'params': [param for param in list(meta_model.parameters()) if
                                        param is not ModelDelta()], 'lr': args.attack_lr * 5e-5},
                            # args.attack_lr* 4e-5},#
                            {'params': [meta_model.p_cur()], 'lr': -args.attack_lr},
                        ]
                    inner_opt = torch.optim.SGD(params, lr=args.attack_lr)  # args.max_epsilon / 255.0)
                    with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=True,
                                                track_higher_grads=False if args.first_order else True) as (
                            f_model, diffopt):
                        for j in range(args.inner_loop):
                                # print('inner_loop!')
                            if 'di' in method:
                                output_clean = f_model(_meta_in(DI_transform(ori_img)))
                                output_v3 = f_model(_meta_in(DI_transform(f_model.p_cur())))  # trans_incep(delta()))
                            elif 'si' in method:
                                output_clean = f_model(_meta_in(SI_transform(ori_img)))
                                output_v3 = f_model(_meta_in(SI_transform(f_model.p_cur())))  # trans_incep(delta()))
                            else:
                                output_clean = f_model(_meta_in(ori_img))
                                output_v3 = f_model(_meta_in(f_model.p_cur()))  # trans_incep(delta()))

                            if 'si' in method:
                                loss = nn.CrossEntropyLoss()(output_v3,
                                                                label.repeat(3).to(device)) + 0.1 * nn.CrossEntropyLoss()(
                                    output_clean, label.repeat(3).to(device))
                            else:
                                loss = nn.CrossEntropyLoss()(output_v3, label.to(device))+0.1*nn.CrossEntropyLoss()(output_clean, label.to(device))
                            if torch.isnan(loss):
                                break
                            else:
                                loss_inner_list.append(loss.item())
                                # model.zero_grad()
                                diffopt.step(loss)

                        for _ in range(len(loss_inner_list)):
                            if np.isnan(loss_inner_list[_]):
                                loss_inner_list = loss_inner_list[:_]
                                break

                        # print(loss_outer_list)
                        # pmax = loss_outer_list.index(min(loss_outer_list)) if len(loss_inner_list) > 0 else -1
                        pmax=len(loss_inner_list)-1 if len(loss_inner_list)>0 else -1
                        pmax_list.append(pmax)
                        for task, task_model in enumerate(task_model_list):
                            # logits = task_model(f_model[1].p_cur.parameters(
                            #         time=pmax + 1)) if not args.first_order else task_model(f_model[1].p_cur())
                            logits = task_model(f_model.p_cur())
                            loss_val = nn.CrossEntropyLoss()(logits, label.to(device))
                            if task == 0:
                                loss_outer_final = loss_val
                            else:
                                loss_outer_final += loss_val
                            task_model.zero_grad()
                        time_0 = time.time()

                        if 'di' in method:
                            loss_inner = nn.CrossEntropyLoss()(
                                f_model(_meta_in(DI_transform(f_model.p_cur()))), label.to(device))
                        elif 'si' in method:
                            loss_inner = nn.CrossEntropyLoss()(
                                f_model(_meta_in(SI_transform(f_model.p_cur()))), label.repeat(3).to(device))
                        else:
                            loss_inner = nn.CrossEntropyLoss()(f_model(_meta_in(f_model.p_cur())), label.to(device))
                        grad_vec = torch.autograd.grad(loss_inner, f_model.p_cur.parameters(),
                                                        create_graph=True)
                        inner_grad = torch.nn.utils.parameters_to_vector(grad_vec)
                        outer_grad = torch.nn.utils.parameters_to_vector(
                            torch.autograd.grad(loss_outer_final, [f_model.p_cur()]))
                        grad_z = CG(inner_grad, outer_grad, [f_model.p_cur()], meta_model.p_cur, 1, 1)  # 5,0.1)
                        time_bp = time.time() - time_0
                        time_bp_list.append(time_bp)
                    input_grad += grad_z[0].data
                    del loss_inner, loss_outer_final, grad_vec, inner_grad, outer_grad, grad_z
                elif 'admix' in method:
                    input_grad=0
                    for _ in range(3):
                        random_indices = list(range(img_x.shape[0]))
                        random.shuffle(random_indices)
                        for gamma in [1., 1./2, 1./4, 1./8, 1./16]:
                            temp_img = ((img_x + portion * img_x[random_indices]) * gamma).detach().requires_grad_()
                            att_out = model(temp_img)
                            pred = torch.argmax(att_out, dim=1).view(-1)
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                            model.zero_grad()
                            loss.backward()
                            input_grad += temp_img.grad.data
                    input_grad /= 15
                model.zero_grad()
                if 'ti' in method:
                    input_grad = F.conv2d(input_grad, gaussian_kernel, bias=None, stride=1,
                                          padding=((5 - 1) // 2, (5 - 1) // 2), groups=3)
                if 'mdi2fgsm' in method or ('mifgsm' in method and not 'vmifgsm' in method) or 'admix' in method or 'nifgsm' in method  or 'vmifgsm' in method:
                    input_grad = 1 * m + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
                    m = input_grad
                if target_attack:
                    input_grad = - input_grad
                if method == 'fgsm' or '_fgsm' in method:
                    img = img.data + 2 * epsilon * torch.sign(input_grad)
                else:
                    img = img.data + alpha * torch.sign(input_grad)
                img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                img = torch.clamp(img, min=0, max=1)
                if 'awt' in method:
                    delta = torch.zeros_like(img).to(device)
                    delta.requires_grad_()
                    averaged_gradient = awt_attack.get_averaged_gradient(img, delta, label.to(device))
                    # Calculate the momentum
                    momentum = awt_attack.get_momentum(averaged_gradient, momentum)
                    delta = awt_attack.update_delta(delta, img, momentum, alpha)
                    img = img.data + delta
                    img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                    img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                    img = torch.clamp(img, min=0, max=1)
                    del delta
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                _elapsed_meta = time.perf_counter() - _t0_meta
                _peak_meta = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                meta_iter_times[i].append(_elapsed_meta)
                meta_iter_mem_peaks[i].append(_peak_meta)
            _t_post_meta = time.perf_counter()
            img_backup = img.clone()
            # m = 0
            # variance = 0c
            # number = 1  # 20
            # beta = 1.5
            # grad_t = 0ccccc
            if 'awt' in method:
                input_grad = awt_attack.forward(data=img,label=label)
                img = img.data + input_grad
                img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                img = torch.clamp(img, min=0, max=1)
            else:
                if not ('admix' in method):
                    for j in range(niters):
                        if 'mdi2fgsm' in method:
                            if j == 0:
                                img_x = img
                            else:
                                img_x = input_diversity(img)
                        elif 'pgd' in method or ('mifgsm' in method and 'vmifgsm' not in method and 'mba' in method):
                            img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
                        elif 'nifgsm' in method:
                            img_x = img + alpha * 1 * m
                        else:
                            img_x = img
                        img_x.requires_grad_(True)

                        img_x_temp = img_x
                        if 'di' in method:
                            att_out = model(DI_transform(img_x_temp))
                        elif 'si' in method:
                            att_out = model(SI_transform(img_x_temp))
                        else:
                            att_out = model(img_x_temp)
                        if 'si' in method:
                            loss = nn.CrossEntropyLoss()(att_out, label.repeat(3).to(device))
                        else:
                            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                        model.zero_grad()
                        loss.backward()
                        input_grad = img_x.grad.data

                        if 'ti' in method:
                            input_grad = F.conv2d(input_grad, gaussian_kernel, bias=None, stride=1,
                                                  padding=((5 - 1) // 2, (5 - 1) // 2), groups=3)
                        if 'mdi2fgsm' in method or ('mifgsm' in method and not 'vmifgsm' in method) or 'admix' in method or 'nifgsm' in method:
                            input_grad = 1 * m + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
                            m = input_grad

                        if 'vmifgsm' in method:
                            current_grad = input_grad + variance
                            current_grad = 1 * m + (current_grad) / torch.norm(current_grad, dim=(1, 2, 3), p=1,
                                                                               keepdim=True)
                            m = current_grad
                            variance = 0
                            for _ in range(number):
                                temp_img = (img_x + (torch.rand_like(img_x) * 2 * beta * epsilon - beta * epsilon).to(
                                    device)).detach()
                                temp_img.requires_grad_()
                                if 'relu_linear' in method and model_name == 'resnet50':
                                    att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(
                                        model, temp_img, True, linbp_layer)
                                    loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                                    model.zero_grad()
                                    variance += linbp_backw_resnet50(temp_img, loss, conv_out_ls, ori_mask_ls, relu_out_ls,
                                                                     conv_input_ls, xp=sgm_lambda)
                                else:
                                    temp_out = model(temp_img)
                                    temp_loss = nn.CrossEntropyLoss()(temp_out, label.to(device))
                                    model.zero_grad()
                                    temp_loss.backward()
                                    variance += temp_img.grad.data
                            model.zero_grad()
                            variance = variance / number - input_grad
                            input_grad = m
                        if target_attack:
                            input_grad = - input_grad
                        if method == 'fgsm' or '_fgsm' in method:
                            img = img.data + 2 * epsilon *torch.sign(input_grad)
                        else:
                            if 'gi' in method and (j < args.pre_attack):
                                img = img.data + args.global_amp_factor * alpha * torch.sign(input_grad)
                            else:
                                if 'gi' in method and (j == args.pre_attack):
                                    img = img_backup.clone()
                                img = img.data + alpha*torch.sign(input_grad)
                        img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                        img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                        img = torch.clamp(img, min=0, max=1)

            att_out = model(img)
            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
            model.zero_grad()
            att_out = task_model_list[0](img)
            loss = nn.CrossEntropyLoss()(att_out, label.to(device))
            task_model_list[0].zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            post_meta_times.append(time.perf_counter() - _t_post_meta)
            batch_wall_times.append(time.perf_counter() - _t_batch0)
            np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data*255).cpu().numpy().astype(np.uint8()))
            del img, ori_img, input_grad#,ModelDelta
            if device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        run_time = time.perf_counter() - start_time
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            global_peak_bytes = torch.cuda.max_memory_allocated(device)
        else:
            global_peak_bytes = 0
        print("[Profiling] Attack generation wall time: {:.4f} s".format(run_time))
        print("[Profiling] Note: meta_step time = full outer step (includes inner_loop adapt + surrogate/CG + sign-PGD update).")
        print("[Profiling] Inner-loop-only (Higher adapt steps, inner_loop) time (mean ± std over batches, seconds):")
        for mi in range(args.meta_steps):
            if inner_adapt_times[mi]:
                arr = np.array(inner_adapt_times[mi])
                print("  meta_step {}: mean={:.6f}, std={:.6f}, n_batch={}".format(
                    mi, float(arr.mean()), float(arr.std()), len(arr)))


        if label_ls:
            labels = torch.cat(label_ls)
            np.save(save_dir + '/labels.npy', labels.numpy())
            print('images saved')
        else:
            print('No attack batches were processed; model initialization completed.')
        del model
    else:
        # --------------------------------------test---------------------------------
        inceptionv3 = MODEL.inceptionv3.Inception3()
        inceptionv3.to(device)
        inceptionv3.load_state_dict(torch.load(state_dict_dic+'inception_v3_google-1a9a5a14.pth'))
        inceptionv3.eval()
        print('inceptionv3:', test(model = inceptionv3, trans = trans_incep))
        del inceptionv3

        inception_resnet_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2.eval()
        inception_resnet_v2.to(device)
        print('inception_resnet_v2:', test(model=inception_resnet_v2, trans = trans_incep))
        del inception_resnet_v2

        densenet = torchvision.models.densenet121(pretrained=False)
        densenet.to(device)
        import re
        pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(state_dict_dic+'densenet121-a639ec97.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        densenet.load_state_dict(state_dict)
        densenet.eval()
        print('densenet:', test(model = densenet, trans = trans_se))
        del densenet

        mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
        mobilenet.to(device)
        mobilenet.load_state_dict(torch.load(state_dict_dic+'mobilenet_v2-b0353104.pth'))
        mobilenet.eval()
        print('mobilenet:', test(model = mobilenet, trans = trans_se))
        del mobilenet

        pnasnet = MODEL.pnasnet.pnasnet5large(ckpt_dir =state_dict_dic+'pnasnet5large-bf079911.pth', num_classes=1000, pretrained='imagenet')
        pnasnet.to(device)
        pnasnet.eval()
        print('pnasnet:', test(model = pnasnet, trans = trans_pnas))
        del pnasnet

        senet = MODEL.senet.senet154(ckpt_dir =state_dict_dic+'senet154-c7b49a05.pth')
        senet.to(device)
        senet.eval()
        print('senet:', test(model = senet, trans = trans_se))
        del senet

        ens3_adv_inc_v3 = MODEL.ens3_adv_inc_v3.KitModel(state_dict_dic+'tf2torch_ens3_adv_inc_v3.npy', aux_logits=False)
        ens3_adv_inc_v3.eval()
        ens3_adv_inc_v3.to(device)
        print('ens3_adv_inc_v3:', test(model=ens3_adv_inc_v3, trans=trans_ens_adv_inception_v3, if_minus=True))
        del ens3_adv_inc_v3

        ens4_adv_inc_v3 = MODEL.ens4_adv_inc_v3.KitModel(state_dict_dic+'tf2torch_ens4_adv_inc_v3.npy', aux_logits=False)
        ens4_adv_inc_v3.eval()
        ens4_adv_inc_v3.to(device)
        print('ens4_adv_inc_v3:', test(model=ens4_adv_inc_v3, trans=trans_ens_adv_inception_v3, if_minus=True))
        del ens4_adv_inc_v3

        ens_adv_inception_resnet_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        ens_adv_inception_resnet_v2.eval()
        ens_adv_inception_resnet_v2.to(device)
        print('ens_adv_inception_resnet_v2:', test(model=ens_adv_inception_resnet_v2, trans = trans_incep))
        del ens_adv_inception_resnet_v2

        def load_pretrained_model(cnn_model=[], vit_model=[]):
            for model_name in cnn_model:
                yield model_name, models.__dict__[model_name](weights="DEFAULT")
                # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
            for model_name in vit_model:
                yield model_name, timm.create_model(model_name, pretrained=True)

        def wrap_model(model):
            """
            Add normalization layer with mean and std in training configuration
            """
            if hasattr(model, 'default_cfg'):
                """timm.models"""
                mean = model.default_cfg['mean']
                std = model.default_cfg['std']
            else:
                """torchvision.models"""
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean, std)
            return torch.nn.Sequential(normalize, model)


        cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                         'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
        vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                         'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

        vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                           'visformer_small', 'swin_tiny_patch4_window7_224']

        def load_single_model(model_name):
            if model_name in models.__dict__.keys():
                print('=> Loading model {} from torchvision.models'.format(model_name))
                model = models.__dict__[model_name](weights="DEFAULT")
            elif model_name in timm.list_models():
                print('=> Loading model {} from timm.models'.format(model_name))
                model = timm.create_model(model_name, pretrained=True)
            else:
                raise ValueError('Model {} not supported'.format(model_name))
            return wrap_model(model.eval().cuda())

        for model_name_test in vit_model_paper:
            model_test = load_single_model(model_name_test).eval().to(device)
            print(model_name_test+':', test(model=model_test, trans = trans_all_224_no_normalize))
            del model_test


        if model_name=='vgg19_bn':
            source_model = torchvision.models.vgg19_bn(pretrained=True)
        elif model_name=='resnet50':
            source_model = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError('Non-existent model name!')
        source_model.eval()
        source_model.to(device)
        print('source_model:', test(model = source_model, trans = trans_ori))
        del source_model
