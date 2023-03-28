import os
import sys
import json
import torch
from scipy.stats import truncnorm


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

      
def is_conditional(G):
    return 'biggan' in G.__class__.__name__.lower()


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec


def save_command_run_params(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')


def truncated_noise(size, truncation=1.0):
    return truncnorm.rvs(-truncation, truncation, size=size)

"""Utility functions."""

import os
import subprocess
import numpy as np

import torch

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type

__all__ = ['postprocess', 'load_generator', 'factorize_weight',
           'HtmlPageVisualizer']

CHECKPOINT_DIR = 'checkpoints'


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`.

    Args:
      array: The input array to convert.

    Returns:
      A `torch.Tensor` with dtype `torch.FloatTensor` on cuda device.
    """
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def load_generator(model_name):
    """Loads pre-trained generator.

    Args:
        model_name: Name of the model. Should be a key in `models.MODEL_ZOO`.

    Returns:
        A generator, which is a `torch.nn.Module`, with pre-trained weights
            loaded.

    Raises:
        KeyError: If the input `model_name` is not in `models.MODEL_ZOO`.
    """
    if model_name not in MODEL_ZOO:
        raise KeyError(f'Unknown model name `{model_name}`!')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')
    return generator

# 解析索引
def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Args:
        obj: The input object to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this value. (default: None)

    Returns:
        A list of integers.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
        # 
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    # 如果obj是字符串
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
            else:
                raise ValueError(f'Unable to parse the input!')

    else:
        raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

# 分解权重
# 传入了generator

def factorize_weight(generator, layer_idx='all'):
    """Factorizes the generator weight to get semantics boundaries.

    Args:
        generator: Generator to factorize.
        layer_idx: Indices of layers to interpret, especially for StyleGAN and
            StyleGAN2. (default: `all`)

    Returns:
    #    (要解释的layers,语义边界，特征值)
        A tuple of (layers_to_interpret, semantic_boundaries, eigen_values).

    Raises:
        ValueError: If the generator type is not supported.
    """
    # Get GAN type.
    gan_type = parse_gan_type(generator)

    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            # 解析索引
            layers = parse_indices(layer_idx,
                                   min_val=0,
                                   max_val=generator.num_layers - 1)
    else:
        layers=[0]
    print(generator)
    # Factorize semantics from weight.
    # 从权重中分解语义
    weights = []
    for idx in layers:
        # f'{}' 相当于format的用法 
        layer_name = f'layer{idx}'
        if gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            layer_name = f'output{idx // 2}'
        if gan_type == 'pggan':
            print(type(generator.__getattr__(layer_name)))
            weight = generator.__getattr__(layer_name).weight
            print(weight.shape)
            weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)
            print(weight.shape)
            # torch.Size([512, 512, 4, 4])
            # torch.Size([512, 8192])
        if gan_type=="SN_MNIST":
            weight = generator[0].weight.T
            print(type(weight))
            print(weight.shape)
            # weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)
        elif gan_type in ['stylegan', 'stylegan2']:
            weight = generator.synthesis.__getattr__(layer_name).style.weight.T
        # 
        weights.append(weight.cpu().detach().numpy())
    # 获取所有权重的值
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    # 在这里求得特征向量和特征值
    
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    
    return layers, eigen_vectors.T, eigen_values


