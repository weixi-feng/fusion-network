import argparse


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False, default='./dataset/test', help='testing dataset')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model')
    parser.add_argument('--cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='residual_physics',
                        help='chooses which model to use. [residual_physics | two_stream | dehazenet | our_model]')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--image_size', type=int, default=16, help='scale images to this size')
    parser.add_argument('--load_epoch', type=int, default=0, help='which iteration to load')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--load_epoch', type=int, default=0, help='which iteration to load')
    parser.add_argument('--load_exp', type=int, default=1, help='load experiment id')
    opt = parser.parse_args()
    return opt
