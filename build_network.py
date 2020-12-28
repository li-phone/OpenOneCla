import torch
import torchvision.models
import torch
import torch.nn as nn
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import *
import torchvision


def build_optimizer(model, cfg):
    # for param in model.parameters():
    #     param.requires_grad = True

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, **cfg.optimizer['SGD'])
    lr_scheduler = CosineAnnealingLR(optimizer, **cfg.lr_scheduler['CosineAnnealingLR'])

    return optimizer, lr_scheduler


def build_loss(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)


def resume_network(model, cfg):
    last_epoch = -1
    if cfg.resume_from is not None and (os.path.islink(cfg.resume_from) or os.path.exists(cfg.resume_from)):
        resume_from = cfg.resume_from
        if os.path.islink(resume_from):
            resume_from = os.readlink(resume_from)
        checkpoint = torch.load(resume_from, map_location='cpu')
        optimizer, lr_scheduler = checkpoint['optimizer'], checkpoint['lr_scheduler']
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint['state']['last_epoch']
        logger.info("resume from {}".format(resume_from))
    else:
        optimizer, lr_scheduler = build_optimizer(model, cfg)
        logger.info("warning... resume from {} failed!".format(cfg.resume_from))

    model.train()
    return model, optimizer, lr_scheduler, last_epoch


def build_network(num_classes, gpus=[], load_from=None, **kwargs):
    model = torch.hub.load('pytorch/vision', **kwargs)
    obj_list = dir(model)

    if load_from is not None:
        state_dict = torch.load(load_from, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info("load from {}".format(load_from))

    if 'fc' in obj_list:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'classifier' in obj_list:
        model.classifier = nn.Sequential(nn.Dropout(p=0.8, inplace=True),
                                         nn.Linear(1280, num_classes))
    # model = FocalModel(model)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(gpus[0])
        model_gpu = nn.DataParallel(module=model, device_ids=gpus)
        model = model_gpu.cuda(gpus[0])
    elif torch.cuda.is_available():
        model.cuda()
    return model


if __name__ == '__main__':
    model = build_network(type='resnet', name='resnet50', num_classes=100, pretrained=False)
    print(model)
