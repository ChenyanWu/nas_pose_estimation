from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import logging
from pathlib import Path

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger


import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

def find_model_path(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        raise ValueError('root does not exist')

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    final_output_dir = root_output_dir / dataset / model / cfg_name
    return final_output_dir

def main():
    args = parse_args()
    update_config(cfg, args)

    model_dir = find_model_path(cfg, args.cfg, 'valid')
    model_path = os.path.join(model_dir, 'model_best.pth')
    print(model_path)
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    searched_model = torch.load(model_path)
    dict_coeff = {}
    for name in searched_model:
        if 'coeff' in name:
            dict_coeff[name] = searched_model[name]
    torch.save(dict_coeff, os.path.join(model_dir, 'coeff.pth'))
    coeff_search_model = torch.load(os.path.join(model_dir, 'coeff.pth'))
    model.load_state_dict(coeff_search_model, strict=False)
    for name, param in model.named_parameters():
        if "coeff" in name:
            print(name, param)
    print('ok')



if __name__ == '__main__':
    main()
