from opt.base_opt import *


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        # rewrite devalue values
        parser.set_defaults(model='test')

        self.isTrain = False
        self.print = False
        return parser
