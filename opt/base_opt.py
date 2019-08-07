import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=False, default='./dataset/train',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--load_model', type=bool, default=True, help='whether to load model')
        parser.add_argument('--cuda', type=bool, default=True, help='using cuda')
        # model parameters
        parser.add_argument('--model', type=str, default='residual_physics',
                            help='chooses which model to use. [residual_physics | two_stream | dehazenet | our_model]')
        # dataset parameters
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--image_size', type=int, default=256, help='scale images to this size')
        # additional parameters
        parser.add_argument('--load_epoch', type=int, default=0, help='which iteration to load')
        parser.add_argument('--load_exp', type=int, default=1, help='load experiment id')
        self.initialized = True
        return parser

    def parse(self):
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        return parser.parse_args()
