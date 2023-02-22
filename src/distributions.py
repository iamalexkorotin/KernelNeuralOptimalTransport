import torch
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)


class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)
    
class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, device='cuda'
    ):
        super(CubeUniformSampler, self).__init__(device=device)
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = np.eye(self.dim, dtype=np.float32) if self.normalized else np.eye(self.dim, dtype=np.float32) / 12
        self.mean = np.zeros(self.dim, dtype=np.float32) if self.centered else .5 * np.ones(self.dim, dtype=np.float32)

        self.bias = torch.tensor(self.mean, device=self.device)

    def sample(self, size=10):
        with torch.no_grad():
            sample = np.sqrt(self.var) * (torch.rand(
                size, self.dim, device=self.device
            ) - .5) / np.sqrt(self.dim / 12)  + self.bias
        return sample
    
class MixN2GaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, device='cuda'):
        super(MixN2GaussiansSampler, self).__init__(device=device)
        
        assert dim == 2
        self.dim = 2
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(centers, device=self.device,)
        
    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        indices = random.choices(range(len(self.centers)), k=batch_size)
        with torch.no_grad():
            batch *= self.std
            batch += self.step * self.centers[indices, :]
        return batch 

class MixNGaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, device='cuda'):
        super(MixNGaussiansSampler, self).__init__(device=device)
        
        assert dim == 1
        self.dim = 1
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        self.centers = torch.tensor(grid_1d, device=self.device,)
        
    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        indices = random.choices(range(len(self.centers)), k=batch_size)
        with torch.no_grad():
            batch *= self.std
            batch += self.step * self.centers[indices, None]
        return batch
    
    
class Mix8GaussiansSampler(Sampler):
    def __init__(self, with_central=False, std=1, r=12, dim=2, device='cuda'):
        super(Mix8GaussiansSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch
    
class SphereUniformSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(SphereUniformSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device
        )
        batch /= torch.norm(batch, dim=1)[:, None]
        return torch.tensor(batch, device=self.device)

class Transformer(object):
    def __init__(self, device='cuda'):
        self.device = device
        

class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()
        
        mean, cov = np.mean(batch, axis=0), np.matrix(np.cov(batch.T))
        self.mean = torch.tensor(
            mean, device=self.device, dtype=torch.float32
        )
        
        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier),
            device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(self.base_sampler.sample(batch_size), device=self.device)
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch
    
class LinearTransformer(Transformer):
    def __init__(
        self, base_sampler, weight, bias=None,
        device='cuda'
    ):
        super(LinearTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler
        
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32)
        else:
            self.bias = torch.zeros(self.weight.size(0), device=device, dtype=torch.float32)
        
    def sample(self, size=4):        
        batch = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch
    
class NormalNoiseTransformer(Transformer):
    def __init__(
        self, base_sampler, std=0.01,
        device='cuda'
    ):
        super(NormalNoiseTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler
        self.std = std
        
    def sample(self, batch_size=4):
        batch = self.base_sampler.sample(batch_size)
        with torch.no_grad():
            batch = batch + self.std * torch.randn_like(batch)
        return batch