import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=False, default='./dataset/train',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--load_model', type=bool, default=False, help='whether to load model')
        parser.add_argument('--cuda', type=bool, default=True, help='using cuda')
        parser.add_argument('--save_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='residual_physics',
                            help='chooses which model to use. [residual_physics | two_stream | dehazenet | our_model]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        # dataset parameters
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--image_size', type=int, default=16, help='scale images to this size')
        # additional parameters
        parser.add_argument('--load_epoch', type=int, default=0,
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.initialized = True
        return parser

    def parse(self):
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        return parser.parse_args()
