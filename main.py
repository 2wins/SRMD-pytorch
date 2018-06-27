import argparse
import torch
from data_loader import get_loader
import os
from solver import Solver
try:
    import nsml
    USE_NSML = True
except ImportError:
    USE_NSML = False


if USE_NSML:
    DATA_PATH = nsml.DATASET_PATH
else:
    DATA_PATH = './Database'


def str2bool(v):
    return v.lower() in ('true')


def main(config, scope):
    # Create directories if not exist.
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if config.mode == 'test':
        config.batch_size = config.test_size

    # Data loader
    data_loader = get_loader(config.image_path, config)

    # Solver
    solver = Solver(data_loader, config)

    def load(filename, *args):
        solver.load(filename)

    def save(filename, *args):
        solver.save(filename)

    def evaluate(test_data, output):
        pass

    def decode(input):
        return input

    if USE_NSML:
        nsml.bind(save, load)

        if config.pause:
            nsml.paused(scope=scope)
    
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=40)
    parser.add_argument('--num_blocks', type=int, default=11)
    parser.add_argument('--num_channels', type=int, default=18)
    parser.add_argument('--conv_dim', type=int, default=4)
    parser.add_argument('--scale_factor', type=int, default=2)

    # Training settings
    parser.add_argument('--total_step', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--trained_model', type=int, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')

    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)

    # NSML setting
    parser.add_argument('--pause', type=int, default=0)

    config = parser.parse_args()

    # Device selection
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data path
    config.image_path = os.path.join(DATA_PATH, config.mode)

    print(config)
    main(config, scope=locals())
