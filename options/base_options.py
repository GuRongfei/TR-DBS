import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # file parameters
        parser.add_argument('--root_path', type=str, default='./dr_results', help='root path to the file package')
        parser.add_argument('--package_name', type=str, required=True)

        # train parameters
        parser.add_argument('--num_iteration', type=int, default=1000, help='*1024*8? will be #samples')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--num_env', type=int, default=8, help='#env sampled each iteration')
        parser.add_argument('--num_epoch', type=int, default=4, help='#reuse of each sample')

        # generator parameters
        parser.add_argument('--generator_mode', type=str, default='uniform', help='mode of parameter generator')
        parser.add_argument('--random_params', type=str, default='0,1', help='0: amplitude_rate, 1: frequency_rate')
        parser.add_argument('--stable_params', type=str, default='0', help='0: len_state')

        # env parameters
            #stable
        parser.add_argument('--len_state', type=int, default=250, help='length of observation')
            #random
        parser.add_argument('--range_ar_low', type=float, default=0.5, help='lower bound of amplitude rate')
        parser.add_argument('--range_ar_up', type=float, default=2, help='upper bound of frequency rate')
        parser.add_argument('--range_fr_low', type=float, default=0.5, help='lower bound of amplitude rate')
        parser.add_argument('--range_fr_up', type=float, default=2, help='upper bound of frequency rate')

        # agent parameters
        parser.add_argument('--policy_lr', type=float, default=2.5e-4, help='learning rate for policy')
        parser.add_argument('--value_lr', type=float, default=1e-4, help='learning rate for value')

        #gpu setup
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.initialized = True
        return parser

    def save_options(self, opt):
        if not os.path.exists(opt.root_path):
            os.mkdir(opt.root_path)
        assert not os.path.exists(opt.root_path + '/' + opt.package_name), 'file already exist'
        os.mkdir(opt.root_path + '/' + opt.package_name)
        message = ''
        message += '---------------Options-----------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += '-----------------End-------------------\n'
        file_name = opt.root_path + '/' + opt.package_name + '/' + 'options.txt'
        with open(file_name, 'wt') as f:
            f.write(message)
            f.write('\n')

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        self.save_options(opt)
        
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
