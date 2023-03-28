from latent_deformator import DeformatorType
from trainer import ShiftDistribution


HUMAN_ANNOTATION_FILE = 'human_annotation.txt'


DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}

# weights
WEIGHTS = {
    'BigGAN': 'models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN': 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'models/pretrained/generators/SN_Anime',
    'StyleGAN2': 'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
    'stylegan2-ada_cifar10':'models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl',
    'SN_GAN_MNIST':'models/pretrained/generators/SN_GAN_MNIST',
    'SN_GAN_gaussian':'models/pretrained/generators/SN_GAN_gaussian'
}
# stylegan2-ada is a newer version of the StyleGAN2
# some links about these gans
# https://github.com/NVlabs/stylegan2
# 
# https://github.com/rosinality/stylegan2-pytorch 
# usage of the stylegan2-pytorch
# python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl
# 
# read more readme.md which is the most important
# 
# 
