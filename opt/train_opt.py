import argparse


def train_parser():
    parser = argparse.ArgumentParser(description='arguments for training')
    parser.add_argument('--dataroot', required=False, default='./dataset/train', help='path to training dataset')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='residual_physics',
                        help='chooses which model to use. [residual_physics | two_stream | dehazenet | our_model]')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--image_size', type=int, default=128, help='scale images to this size')
    parser.add_argument('--load_epoch', type=int, default=0, help='which iteration to load')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    parser.add_argument('--save_epoch_freq', type=int, default=20,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--test_freq', type=int, default=5, help='frequency of testing models on test set')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use, either adam or sgd')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    opt = parser.parse_args()
    return opt
