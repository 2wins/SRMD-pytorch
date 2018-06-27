import torch
import numpy as np
from PIL import Image
from scipy import signal
from scipy.ndimage import convolve
import h5py, os


class Kernels(object):
    def __init__(self, kernels, proj_matrix):
        self.kernels = kernels
        self.P = proj_matrix

        # kernels.shape == [H, W, C, N], C: no. of channels / N: no. of kernels
        self.kernels_proj = np.matmul(self.P,
                                      self.kernels.reshape(self.P.shape[-1],
                                                           self.kernels.shape[-1]))

        self.indices = np.array(range(self.kernels.shape[-1]))
        self.randkern = self.RandomKernel(self.kernels, [self.indices])

    def RandomBlur(self, image):
        kern = next(self.randkern)
        return Image.fromarray(convolve(image, kern, mode='nearest'))

    def ConcatDegraInfo(self, image):
        image = np.asarray(image)   # PIL Image to numpy array
        h, w = list(image.shape[0:2])
        proj_kernl = self.kernels_proj[:, self.randkern.index - 1]  # Caution!!
        n = len(proj_kernl)  # dim. of proj_kernl

        maps = np.ones((h, w, n))
        for i in range(n):
            maps[:, :, i] = proj_kernl[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image

    class RandomKernel(object):
        def __init__(self, kernels, indices):
            self.len = kernels.shape[-1]
            self.indices = indices
            np.random.shuffle(self.indices[0])
            self.kernels = kernels[:, :, :, self.indices[0]]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if (self.index == self.len):
                np.random.shuffle(self.indices[0])
                self.kernels = self.kernels[:, :, :, self.indices[0]]
                self.index = 0

            n = self.kernels[:, :, :, self.index]
            self.index += 1
            return n


def load_kernels(file_path='kernels/', scale_factor=2):
    f = h5py.File(os.path.join(file_path, 'SRMDNFx%d.mat' % scale_factor), 'r')

    directKernel = None
    if scale_factor != 4:
        directKernel = f['net/meta/directKernel']
        directKernel = np.array(directKernel).transpose(3, 2, 1, 0)

    AtrpGaussianKernels = f['net/meta/AtrpGaussianKernel']
    AtrpGaussianKernels = np.array(AtrpGaussianKernels).transpose(3, 2, 1, 0)

    P = f['net/meta/P']
    P = np.array(P)
    P = P.T

    if directKernel is not None:
        K = np.concatenate((directKernel, AtrpGaussianKernels), axis=-1)
    else:
        K = AtrpGaussianKernels

    return K, P


"""The functions below are not used currently"""


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)

    v, w = torch.eig(torch.mm(X, torch.t(X)), eigenvectors=True)
    return torch.mm(w[:k, :], X)


def isogkern(kernlen, std):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d


def anisogkern(kernlen, std1, std2, angle):
    gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
    gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_1, gkern1d_2)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d
