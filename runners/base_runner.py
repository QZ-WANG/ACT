from os.path import join
import torch
from torch.utils.tensorboard import SummaryWriter
import os

import warnings


class BaseRunner:
    def __init__(self, x_all, split, model, optimiser_parm, device, out_paths, resume_from, ckpt_dir=None):
        self.x_all = x_all
        self.optim_parm = optimiser_parm
        self.model = model
        self.optimiser = None
        self.split = split
        self.device = device
        self.out_paths = out_paths
        self.ckpt_dir = ckpt_dir
        if self.out_paths is not None:
            self.tb_writer = self._set_tb()
        self.resume_from =resume_from

    def resume_train(self, state_keys):
        if self.resume_from is None:
            raise RuntimeError("Checkpoint name is None!")

        if self.ckpt_dir is None:
            self.ckpt_dir = join(self.out_paths['ckpt_dir'], self.resume_from)

        ckpt_path = join(self.ckpt_dir, "last.pt")
        checkpoint = torch.load(ckpt_path)

        state_dict = dict()

        for key in state_keys:
            state_dict[key] = checkpoint[key]

        self.model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.optimiser.load_state_dict(checkpoint.pop('optimizer_state_dict'))

        return state_dict


    def _set_tb(self):
        tf_dir = join(self.out_paths['tb_data'], "runs_" + self.out_paths['task_name'] + "_" + self.out_paths['timestamp'])

        if not os.path.isdir(tf_dir):
            os.makedirs(tf_dir)
        else:
            warnings.warn("Tensorboard dir already existed!")

        writer = SummaryWriter(log_dir=tf_dir)
        return writer

    def write_tb(self, state_dict, step):
        for key in state_dict.keys():
            if isinstance(state_dict[key], dict):
                self.tb_writer.add_scalars(key, state_dict[key], step)
            else:
                self.tb_writer.add_scalar(key, state_dict[key], step)

    def save_state(self, state_dict, state_type):
        if self.out_paths is None:
            return

        # if state_type not in ["best", "last"]:
        #     raise NotImplementedError("State type not supported!")

        if self.ckpt_dir is None:
            self.ckpt_dir = join(self.out_paths['ckpt_dir'], self.out_paths['task_name'] + "_" + self.out_paths['timestamp'])

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        ckpt_fn = join(self.ckpt_dir, state_type+".pt")

        torch.save(state_dict, ckpt_fn)


    def write_tb_fig(self, tag, fig, epoch):
        self.tb_writer.add_figure(tag, fig, epoch)


    def train(self):
        pass

    def test(self):
        pass