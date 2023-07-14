import logging
import os
import time
import torch
from distutils.dir_util import copy_tree

from parser.model import CompoundPCFG
from parser.model import FastNBLPCFG
from parser.model import FastTNPCFG
from parser.model import NeuralBLPCFG
from parser.model import NeuralLPCFG
from parser.model import NeuralPCFG
from parser.model import TNPCFG


def get_model(args, dataset):
    if args.model_name == 'NPCFG':
        return NeuralPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'CPCFG':
        return CompoundPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'TNPCFG':
        return TNPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'NLPCFG':
        return NeuralLPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'NBLPCFG':
        return NeuralBLPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'FastTNPCFG':
        return FastTNPCFG(args, dataset).to(dataset.device)

    if args.model_name == 'FastNBLPCFG':
        return FastNBLPCFG(args, dataset).to(dataset.device)

    raise KeyError


def get_optimizer(args, model):
    if args.name == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))

    if args.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu),
                                 weight_decay=args.weight_decay)

    raise NotImplementedError


def get_logger(args, log_name='train', path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = f"/{model_name}" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print(f'Creating {args.save_dir}')
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return saved_name
