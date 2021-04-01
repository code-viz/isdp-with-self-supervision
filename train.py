import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler_torch

from networks import ResNet50Fc
from loss import loss_dict

def arg_parse():
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=str, nargs='?', default='poi_3.0', help="source data")
    parser.add_argument('--target', type=str, nargs='?', default='ant_1.6', help="target data")
    parser.add_argument('--network', type=str, nargs='?', default='ResNet50Fc', help="network name")
    parser.add_argument('--loss_name', type=str, nargs='?', default='DAN', help="loss name")
    parser.add_argument('--tradeoff', type=float, nargs='?', default=1, help="tradeoff")
    return parser.parse_args()

if __name__ =="__main__":

    args = arg_parse()

    path = 'data/txt/'

    config = {}
    config['epoch'] = 100
    config['prep'] = [
        {'name':'source', 'resize_size':256, 'crop_size':256},
        {'name':'target', 'resize_size':256, 'crop_size':256}
    ]
    config['loss'] = {'name':args.loss_name, 'tradeoff':args.tradeoff}
    config['data'] =[
        {
            'name':'source',
            'path': path + args.source + '.txt',
            'batch_size':{'train':64, 'test':64}
        },
        {
            'name':'target',
            'path':path + args.target + '.txt',
            'batch_size':{'train':64, 'test':64}
        }]
    config['network'] = {'name':args.network}
    print(config['loss'])
