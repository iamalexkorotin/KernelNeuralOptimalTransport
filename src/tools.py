import pandas as pd
import numpy as np

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from ot.backend import get_backend
from ot.optim import cg

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm_notebook as tqdm
from .fid_score import calculate_frechet_distance
from .distributions import LoaderSampler
import torchvision.datasets as datasets
import h5py
from torch.utils.data import TensorDataset, ConcatDataset

import gc

from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder

def load_dataset(name, path, img_size=64, batch_size=64, device='cuda'):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, img_size)
    elif name in ['celeba_female', 'aligned_anime_faces']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['dtd']:
        transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception('Unknown dataset')
        
    if name in ['celeba_female']:
        with open('../datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
    else:
        idx = list(range(len(dataset)))
    
    test_ratio=0.1
    test_size = int(len(idx) * test_ratio)
    if name == 'dtd':
        np.random.seed(0x000000); np.random.shuffle(idx)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    else:
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
#     print(len(train_idx), len(test_idx))

    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler
import random

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')    

    return TensorDataset(dataset, torch.zeros(len(dataset)))

def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False, use_Y=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if not use_Y:
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    else:
                        batch = ((Y[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_Z_pushed_loader_stats(T, loader, ZC=1, Z_STD=0.1, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                Z = torch.randn(len(X), ZC, 1, 1) * Z_STD
                XZ = (X, Z)
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(
                        XZ[0][start:end].type(torch.FloatTensor).to(device),
                        XZ[1][start:end].type(torch.FloatTensor).to(device)
                    ).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def kernel_weak_optimal_transport(Xa, Xb, a=None, b=None, gamma=1., verbose=False, log=False, G0=None, **kwargs):
    # Custom implementation based on ot.weak source code from pot
    # gamma - coefficient of the kernel variance
    nx = get_backend(Xa, Xb)

    Xa2 = nx.to_numpy(Xa)
    Xb2 = nx.to_numpy(Xb)

    if a is None:
        a2 = np.ones((Xa.shape[0])) / Xa.shape[0]
    else:
        a2 = nx.to_numpy(a)
    if b is None:
        b2 = np.ones((Xb.shape[0])) / Xb.shape[0]
    else:
        b2 = nx.to_numpy(b)

    # init uniform
    if G0 is None:
        T0 = a2[:, None] * b2[None, :]
    else:
        T0 = nx.to_numpy(G0)

    # weak OT loss with torch (CPU)
    def f_torch(T):
        T_t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        Xa2_t, Xb2_t = torch.tensor(Xa2, dtype=torch.float32), torch.tensor(Xb2, dtype=torch.float32)

        cost_t = (torch.cdist(Xa2_t, Xb2_t) * T_t).sum()
        T2_t = (T_t.T / T_t.sum(axis=1)).T
        T3_t = torch.matmul(T2_t.reshape(T2_t.shape[0], T2_t.shape[1], 1), T2_t.reshape(T2_t.shape[0], 1, T2_t.shape[1]))
        T4_t = (T3_t * T_t.sum(axis=1).reshape(-1, 1, 1)).sum(axis=0)

        cvar_t = (torch.cdist(Xb2_t, Xb2_t) * T4_t).sum()
        f_t = cost_t - (gamma / 2) * cvar_t
        f_t.backward()
        return T_t.grad.cpu().detach().numpy(), f_t.item()
    
    # kernel weak OT cost
    def f(T):
        return f_torch(T)[1]

    # kernel weak OT gradient
    def df(T):
        return f_torch(T)[0]

    # solve with conditional gradient and return solution
    if log:
        res, log = cg(a2, b2, 0, 1, f, df, T0, log=log, verbose=verbose, **kwargs)
        log['u'] = nx.from_numpy(log['u'], type_as=Xa)
        log['v'] = nx.from_numpy(log['v'], type_as=Xb)
        return nx.from_numpy(res, type_as=Xa), log
    else:
        return nx.from_numpy(cg(a2, b2, 0, 1, f, df, T0, log=log, verbose=verbose, **kwargs), type_as=Xa)