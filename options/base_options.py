import argparse
import os
import torch


class TrainOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # file parameters
        parser.add_argument('--train_root_path', type=str, default='./dr_train', help='root path to the file package')
        parser.add_argument('--train_package_name', type=str, required=True)

        # train parameters
        parser.add_argument('--num_iteration', type=int, default=1000, help='*1024*8? will be #samples')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--num_env', type=int, default=8, help='#env sampled each iteration')
        parser.add_argument('--num_epoch', type=int, default=4, help='#reuse of each sample')

        # generator parameters
        parser.add_argument('--generator_mode', type=str, default='uniform', help='mode of parameter generator')
        parser.add_argument('--random_params', type=str, default='0,1', help='0: amplitude_rate, 1: frequency_rate')
        parser.add_argument('--stable_params', type=str, default='2', help='0: amplitude_rate, 1:frequency_rate, 2: len_state')

        # env parameters
            #stable
        parser.add_argument('--len_state', type=int, default=250, help='length of observation')
        parser.add_argument('--amplitude_rate', type=float, default=1, help='value of amplitude rate')
        parser.add_argument('--frequency_rate', type=float, default=1, help='value of frequency rate')
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
        if not os.path.exists(opt.train_root_path):
            os.mkdir(opt.train_root_path)
        assert not os.path.exists(opt.train_root_path + '/' + opt.train_package_name), 'file already exist'
        os.mkdir(opt.train_root_path + '/' + opt.train_package_name)
        message = ''
        message += '---------------Options-----------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += '-----------------End-------------------\n'
        file_name = opt.train_root_path + '/' + opt.train_package_name + '/' + 'options.txt'
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


class TestOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # file parameters
        parser.add_argument('--train_root_path', type=str, default='./dr_train', help='root path to the train file package')
        parser.add_argument('--train_package_name', type=str, required=True)
        parser.add_argument('--test_root_path', type=str, default='./dr_test', help='root path to the test file package')
        parser.add_argument('--test_package_name', type=str, required=True)

        #test parameters
        parser.add_argument('--init_timestep', type=int, default=1000, help='#steps before stimulation')
        parser.add_argument('--stimulation_timestep', type=int, default=2000, help='#steps for stimulation')
        parser.add_argument('--rest_timestep', type=int, default=500, help='#steps after stimulation')

        # test env parameters
        parser.add_argument('--generator_mode', type=str, default='uniform', help='mode of parameter generator')
        parser.add_argument('--random_params', type=str, default='0,1', help='0: amplitude_rate, 1: frequency_rate')
        parser.add_argument('--stable_params', type=str, default='2', help='0: amplitude_rate, 1: frequency_rate, 2: len_state')

        # env parameters
        # stable
        parser.add_argument('--len_state', type=int, default=250, help='length of observation')
        parser.add_argument('--amplitude_rate', type=float, default=1, help='value of amplitude rate')
        parser.add_argument('--frequency_rate', type=float, default=1, help='value of frequency rate')
        # random
        parser.add_argument('--range_ar_low', type=float, default=0.5, help='lower bound of amplitude rate')
        parser.add_argument('--range_ar_up', type=float, default=2, help='upper bound of frequency rate')
        parser.add_argument('--range_fr_low', type=float, default=0.5, help='lower bound of amplitude rate')
        parser.add_argument('--range_fr_up', type=float, default=2, help='upper bound of frequency rate')

        # gpu setup
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.initialized = True
        return parser

    def save_options(self, opt):
        if not os.path.exists(opt.test_root_path):
            os.mkdir(opt.test_root_path)
        assert not os.path.exists(opt.test_root_path + '/' + opt.test_package_name), 'file already exist'
        os.mkdir(opt.test_root_path + '/' + opt.test_package_name)
        message = ''
        message += '---------------Options-----------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += '-----------------End-------------------\n'
        file_name = opt.test_root_path + '/' + opt.test_package_name + '/' + 'options.txt'
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
