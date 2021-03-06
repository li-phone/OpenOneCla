from train import main as train_main
from utils import *
import glob
import os


def batch_train(cfgs, gpus='0'):
    for cfg in cfgs:
        if os.path.exists(cfg):
            cfg = import_module(cfg)
            cfg.gpus = gpus
            train_main(cfg)


def main(dataset_name='cassava'):
    root = 'config/{}/'.format(dataset_name)
    paths = glob.glob(os.path.join(root, 'resnet50*.py'))
    cfgs = [root + os.path.basename(x) for x in paths]
    batch_train(cfgs)


if __name__ == '__main__':
    main()
    pass
