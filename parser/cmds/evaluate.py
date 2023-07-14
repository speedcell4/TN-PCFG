# -*- coding: utf-8 -*-


import click
import numpy as np
import torch
import torch
from datetime import datetime
from datetime import timedelta

from parser.cmds.cmd import CMD
from parser.cmds.cmd import CMD
from parser.helper.data_module import DataModule
from parser.helper.loader_wrapper import DataPrefetcher
from parser.helper.metric import Metric
from parser.helper.util import *


class Evaluate(CMD):

    def __call__(self, args, eval_dep=False, decode_type='mbr'):
        super(Evaluate, self).__call__(args)
        self.device = args.device
        self.args = args
        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        best_model_path = self.args.load_from_dir + "/best.pt"
        self.model.load_state_dict(torch.load(str(best_model_path)))
        print('successfully load')

        test_loader = dataset.test_dataloader
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        if not eval_dep:
            metric_f1, likelihood = self.evaluate(test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type)
        else:
            metric_f1, metric_uas, likelihood = self.evaluate(test_loader_autodevice, eval_dep=eval_dep,
                                                              decode_type=decode_type)
            print(metric_uas)
        print(metric_f1)
        print(likelihood)
