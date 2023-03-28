import os
import tarfile
import argparse

# 下载的数据来源
SOURCES = {
    'mnist': 'https://www.dropbox.com/s/rzurpt5gzb14a1q/pretrained_mnist.tar',
    'anime': 'https://www.dropbox.com/s/9aveavgbluvjeu6/pretrained_anime.tar',
    'biggan': 'https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar',
    'proggan': 'https://www.dropbox.com/s/707xjn1rla8nwqc/pretrained_proggan.tar',
    'stylegan2': 'https://www.dropbox.com/s/c3aaq7i6soxmpzu/pretrained_stylegan2_ffhq.tar',
}

# 下载文件
def download(source, destination):
    print(destination)
    # tmp_tar = os.path.join(destination, 'tars',"pretrained_mnist.tar")
    tmp_tar = os.path.join(destination, 'tars',"pretrained_stylegan2_ffhq.tar")
    
    print(tmp_tar)
    # # urllib has troubles with dropbox
    # os.system(f'wget {source} -O {tmp_tar}')
    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(destination)
    #
    # os.remove(tmp_tar)


# 都下载下来
def main():
    parser = argparse.ArgumentParser(description='Pretrained models loader')
    # 添加参数
    parser.add_argument('--models', nargs='+', type=str,
                        choices=list(SOURCES.keys()) + ['all'], default=['all'])
    # 添加参数
    parser.add_argument('--out', type=str, help='root out dir')

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    models = args.models
    if 'all' in models:
        models = list(SOURCES.keys())
    # 逐个下载
    for model in set(models):
        source = SOURCES[model]
        print(f'downloading {model}\nfrom {source}')
        download(source, args.out)


if __name__ == '__main__':
    main()
