import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 17777

# 获取当前工作目录
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = os.path.realpath(".")

# Dataset config
class Config:
    def __init__(self):
        self.dataset_path = '/home/siat/ycy/Diffusion/datasets/Ruijin-data'  # 根数据集路径
        self.base_path = '/home/siat/ycy/Diffusion/datasets/Ruijin-data'

        # 指定训练集、验证集和测试集的文本文件路径
        self.train_source = os.path.join(self.dataset_path, "train.txt")
        self.eval_source = os.path.join(self.dataset_path, "val.txt")
        self.test_source = os.path.join(self.dataset_path, "test.txt")

        # 文件格式
        self.UNCORRECTED_format = '.dcm'
        self.NORMAL_format = '.dcm'
        self.x_is_single_channel = True

# 使用示例
config = Config()
C.train_source = config.train_source
C.eval_source = config.eval_source
C.test_source = config.test_source
C.dataset_path = config.dataset_path
C.base_path = config.base_path

# Image Config
# Settings for network
C.pretrained_model = None
C.optimizer = 'AdamW'

# Train Config
C.lr = 1e-4
C.momentum = 0.9
C.weight_decay = 1e-5
C.batch_size = 16
C.nepochs = 300
C.num_workers = 0
C.train_scale_array = [1]

C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Eval Config
C.eval_stride_rate = 1
C.eval_scale_array = [1]
C.eval_flip = False

# Store Config
C.checkpoint_start_epoch = 1
C.checkpoint_step = 50

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_pet/' + 'log_')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()