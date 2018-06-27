import torch
import torch.nn as nn
import os
from torchvision.utils import save_image, make_grid
from model import SRMD
import numpy as np

try:
    import nsml
    USE_NSML = True
except ImportError:
    USE_NSML = False

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.num_blocks = config.num_blocks
        self.num_channels = config.num_channels
        self.conv_dim = config.conv_dim
        self.scale_factor = config.scale_factor

        # Training settings
        self.total_step = config.total_step
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.trained_model = config.trained_model
        self.use_tensorboard = config.use_tensorboard

        # Path and step size
        self.log_path = config.log_path
        self.result_path = config.result_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Device configuration
        self.device = config.device

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        # model and optimizer
        self.model = SRMD(self.num_blocks, self.num_channels, self.conv_dim, self.scale_factor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2])

        self.model.to(self.device)

    def load_trained_model(self):
        self.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.trained_model)))
        print('loaded trained models (step: {})..!'.format(self.trained_model))

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S['SR'])

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.optimizer.zero_grad()

    def detach(self, x):
        return x.data

    def train(self):
        self.model.train()

        # Reconst loss
        reconst_loss = nn.MSELoss()

        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)

        # Start with trained model
        if self.trained_model:
            start = self.trained_model + 1
        else:
            start = 0

        for step in range(start, self.total_step):
            # Reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                data_iter = iter(self.data_loader)

            x, y = next(data_iter)

            # from PIL import Image
            # tmp = np.transpose(np.squeeze(x.data.numpy()), (1,2,0))
            # tmp = (255 * tmp).astype(np.uint8)
            # Image.fromarray(tmp).save('test_lr.png')
            # tmp = np.transpose(np.squeeze(y.data.numpy()), (1,2,0))
            # tmp = (255 * tmp).astype(np.uint8)
            # Image.fromarray(tmp).save('test_hr.png')

            x, y = x.to(self.device), y.to(self.device)
            y = y.to(torch.float64)

            out = self.model(x)
            loss = reconst_loss(out, y)

            self.reset_grad()

            # For decoder
            loss.backward(retain_graph=True)

            self.optimizer.step()

            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] loss: {:.4f}".format(step+1, self.total_step, loss.item()))

                if USE_NSML:
                    if self.use_tensorboard:
                        info = {
                            'loss/loss': loss.item(),
                            # 'misc/lr': lr
                        }

                        for key, value in info.items():
                            self.logger.scalar_summary(key, value, step + 1, scope=locals())

            # Sample images
            if (step+1) % self.sample_step == 0:
                self.model.eval()
                reconst = self.model(x)

                def to_np(x):
                    return x.data.cpu().numpy()

                if USE_NSML:
                    tmp = nn.Upsample(scale_factor=2)(x.data[:,0:3,:])
                    pairs = torch.cat((tmp.data[0:2,:], reconst.data[0:2,:], y.data[0:2,:]), dim=3)
                    grid = make_grid(pairs, 2)
                    tmp = 255 * grid.cpu().numpy()
                    # tmp = (255 * tmp).astype(np.uint8)
                    self.logger.images_summary('recons', tmp, step + 1)
                else:
                    tmp = nn.Upsample(scale_factor=2)(x.data[:,0:3,:])
                    pairs = torch.cat((tmp.data[0:2,:], reconst.data[0:2,:], y.data[0:2,:]), dim=3)
                    grid = make_grid(pairs, 2)
                    from PIL import Image
                    tmp = np.squeeze(grid.numpy().transpose((1, 2, 0)))
                    tmp = (255 * tmp).astype(np.uint8)
                    Image.fromarray(tmp).save('./samples/test_%d.jpg' % (step + 1))

            # Save check points
            if (step+1) % self.model_save_step == 0:
                if USE_NSML:
                    nsml.save(step)
                else:
                    pass

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'SR': model}, filename)

    def test(self):
        # Load trained params
        self.model.eval()
        S = torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.trained_model)))
        self.model.load_state_dict(S['SR'])

        # Sampling
        reconst = self.model(x)
        save_image(reconst.data, 'reconst.png')

        return self.denorm(fake.data)
