import argparse
import os

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # init params
    parser.add_argument('--mode', default='train', help='whether to train or test')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--use-dr', action='store_true', default=False, help='whether to use domain randomization')
    parser.add_argument('--use-aug', action='store_true', default=False, help='whether to adopt augmented data')
    parser.add_argument('--use-cl', action='store_true', default=False, help='whether to use contrastive learning')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')

    # algo params
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--num-processes', type=int, default=8, help='how many training CPU processes to use')
    parser.add_argument('--num-steps', type=int, default=128, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')

    # train params
    parser.add_argument('--num-env-steps', type=int, default=10e6, help='number of environment steps to train')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--save-dir', default='./result', help='directory to save agent logs (default: ./result)')
    parser.add_argument('--file-name', required=True, help='name of this experiment, saved under save-dir')

    # record intervals
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates')
    parser.add_argument('--eval-interval', type=int, default=None, help='eval interval, one eval per n updates')

    # env params
    parser.add_argument('--env-name', default='oscillator-v0', help='environment to train on')
    parser.add_argument('--shallow-dim', type=int, default=256, help='dimension of shallow feature')
    parser.add_argument('--deep-dim', default=64, help='dimension of deep feature')
        # uniform random params
    parser.add_argument('--uniform_random_params', type=str, default='', help='0: amplitude_rate, 1: frequency_rate')
    parser.add_argument('--range_ar_low', type=float, default=0.8, help='lower bound of amplitude rate')
    parser.add_argument('--range_ar_up', type=float, default=1.2, help='upper bound of frequency rate')
    parser.add_argument('--range_fr_low', type=float, default=0.8, help='lower bound of amplitude rate')
    parser.add_argument('--range_fr_up', type=float, default=1.2, help='upper bound of frequency rate')
        # augmentation params
    parser.add_argument('--magnitude', type=float, default=0, help='magnitude parameter in augmentation')
    parser.add_argument('--sampling', type=float, default=0, help='magnitude parameter in augmentation')
    parser.add_argument('--noise', type=float, default=0, help='noise parameter in augmentation')
    parser.add_argument('--timeshift', type=int, default=0, help='timeshift parameter in augmentation')
    parser.add_argument('--permutation', type=int, default=1, help='magnitude parameter in augmentation')
    #magnitude=0., sampling=0., noise=0., timeshift=0, permutation=1

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.env_params = summarize_params(args)

    write_options(args, parser)

    return args


def summarize_params(args):
    params = []
    uniform_random_param_ids = args.uniform_random_params.split(',')

    amplitude_rate = {"name": "amplitude_rate",
                      "num": args.num_processes,
                      "lower_bound": 1.0}
    if '0' in uniform_random_param_ids:
        amplitude_rate["lower_bound"] = args.range_ar_low
        amplitude_rate["upper_bound"] = args.range_ar_up
        amplitude_rate["mode"] = "Uniform"
    params.append(amplitude_rate)

    frequency_rate = {"name": "frequency_rate",
                      "num": args.num_processes,
                      "lower_bound": 1.0}
    if '1' in uniform_random_param_ids:
        frequency_rate["lower_bound"] = args.range_fr_low
        frequency_rate["upper_bound"] = args.range_fr_up
        frequency_rate["mode"] = "Uniform"
    params.append(frequency_rate)

    return params


def write_options(args, parser):
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    if not args.mode == 'train':
        assert os.path.exists(args.save_dir + '/' + args.file_name), 'file doesn\'t exist'
        return

    assert not os.path.exists(args.save_dir + '/' + args.file_name), 'file already exist'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    os.mkdir(args.save_dir + '/' + args.file_name)
    message = ''
    message += '---------------Options-----------------\n'
    for key, value in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(key)
        if value != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
    message += '-----------------End-------------------\n'
    file_name = args.save_dir + '/' + args.file_name + '/' + 'options.txt'
    with open(file_name, 'wt') as f:
        f.write(message)
        f.write('\n')
