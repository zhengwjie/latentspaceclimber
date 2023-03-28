import os
import json
import torch

from constants import DEFORMATOR_TYPE_DICT, HUMAN_ANNOTATION_FILE, WEIGHTS
from latent_deformator import LatentDeformator
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from models.gan_load import make_big_gan, make_proggan, make_sngan,make_style_gan2_ada,get_discriminator


def load_generator(args, G_weights):
    gan_type = args['gan_type']
    if gan_type == 'BigGAN':
        G = make_big_gan(G_weights, args['target_class']).eval()
    elif gan_type in ['ProgGAN']:
        G = make_proggan(G_weights)
    elif gan_type =='stylegan2-ada':
        return load_G_D(args, G_weights)
    elif 'StyleGAN2' in gan_type:
        from models.gan_load import make_style_gan2
        G = make_style_gan2(args['gan_resolution'], G_weights, args['w_shift'])
    else:
        G = make_sngan(G_weights)

    return G,None
def load_G_D(args, G_weights):
    G = make_style_gan2_ada(G_weights,args["directions_count"])
    D=get_discriminator(G_weights)
    return G,D

def load_from_dir(root_dir, model_index=None, G_weights=None, shift_in_w=True,bias=True):
    args = json.load(open(os.path.join(root_dir, 'args.json')))
    args['w_shift'] = shift_in_w

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('deformator')])
    # print(model_index)
    if 'gan_weights' in args.keys() and G_weights is None:
        G_weights = args['gan_weights']
    if G_weights is None or not os.path.isfile(G_weights):
        print('Using default local G weights')
        G_weights = WEIGHTS[args['gan_type']]
        if isinstance(G_weights, dict):
            G_weights = G_weights[str(args['resolution'])]

    if 'resolution' not in args.keys():
        args['resolution'] = 128

    G,D = load_generator(args, G_weights)
    deformator = LatentDeformator(
        shift_dim=G.dim_shift,
        # input_dim是就是方向数
        # max_latent_dim表示的是浅空间的维度
        input_dim=args['directions_count'] if 'directions_count' in args.keys() else None,
        out_dim=args['max_latent_dim'] if 'max_latent_dim' in args.keys() else None,
        type=DEFORMATOR_TYPE_DICT[args['deformator']],bias=bias)
    # print(deformator.input_dim)
    # print(deformator.out_dim)
    if 'shift_predictor' not in args.keys() or args['shift_predictor'] == 'ResNet':
        shift_predictor = LatentShiftPredictor(G.dim_shift)
    elif args['shift_predictor'] == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            # 这里的1和3表示的是图像的通道数
            G.dim_shift, 1 if args['gan_type'] == 'SN_MNIST' else 3)

    deformator_model_path = os.path.join(models_dir, 'deformator_{}.pt'.format(model_index))
    shift_model_path = os.path.join(models_dir, 'shift_predictor_{}.pt'.format(model_index))
    if os.path.isfile(deformator_model_path):
        deformator.load_state_dict(
            torch.load(deformator_model_path, map_location=torch.device('cpu')))
    if os.path.isfile(shift_model_path):
        shift_predictor.load_state_dict(
            torch.load(shift_model_path, map_location=torch.device('cpu')))
    setattr(deformator, 'annotation',
            load_human_annotation(os.path.join(root_dir, HUMAN_ANNOTATION_FILE)))
    return deformator.eval(), G.eval(), shift_predictor.eval()


def load_human_annotation(txt_file, verbose=False):
    annotation_dict = {}
    if os.path.isfile(txt_file):
        with open(txt_file) as source:
            for line in source.readlines():
                indx_str, annotation = line.split(': ')
                if len(annotation) > 0:
                    i = 0
                    annotation_unique = annotation
                    while annotation_unique in annotation_dict.keys():
                        i += 1
                        annotation_unique = f'{annotation} ({i})'
                    annotation_unique = annotation_unique.replace('\n', '').replace(' ', '_')
                    annotation_dict[annotation_unique] = int(indx_str)
        if verbose:
            print(f'loaded {len(annotation_dict)} annotated directions from {txt_file}')

    return annotation_dict

def load_G_D_CIFAR10():
    from constants import WEIGHTS
    import sys
    import pickle
    from models.gan_load import UnConditionedStyleGAN2ADA,UnConditionedStyleD2ADA
    sys.path.append("/home/zhengwanjie/GANLatentDiscovery-master/models/stylegan2_ada/")
    G_weights=WEIGHTS.get('stylegan2-ada_cifar10')
    with open(G_weights, 'rb') as f:
        models=pickle.load(f)
        G = models['G_ema']
        D= models["D"]
    setattr(G, 'dim_z', G.z_dim)
    return UnConditionedStyleGAN2ADA(G).eval(),UnConditionedStyleD2ADA(D).eval()

def load_G_D_MNIST():
    from constants import WEIGHTS
    from models.gan_load import make_sn_gan_mnist
    G_D_path=WEIGHTS['SN_GAN_MNIST']
    G,D=make_sn_gan_mnist(G_D_path)
    return G,D

def load_G_D_gaussian():
    from constants import WEIGHTS
    from models.gan_load import make_sn_gan_gaussian
    G_D_path=WEIGHTS['SN_GAN_gaussian']
    G,D=make_sn_gan_gaussian(G_D_path)
    return G,D 
