import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from backpack import backpack, extend
from backpack.extensions import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np
from loss import Entropy

def get_hessian(args, model, train_loader):
    W = list(model.parameters())[-1]
    m, n = W.shape
    lossfunc = nn.CrossEntropyLoss()

    extend(lossfunc, debug=False)
    extend(model.fc, debug=False)

    with backpack(KFAC()):
        U, V = torch.zeros(m, m).cuda(args.gpu), torch.zeros(n, n).cuda(args.gpu)
        for i, (x, y, _) in tqdm(enumerate(train_loader)):
            x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac

                rho = min(1-1/(i+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
    
    n_data = len(train_loader.dataset)
    M_W = W.t()
    #U = sqrt(n_data)*U
    #V = sqrt(n_data)*V

    return [M_W, U, V]

def estimate_variance(args, var0, hessians, invert=True):
    if not invert:
        return hessians
    
    tau = 1/var0

    with torch.no_grad():
        M_W, U, V = hessians
    
    m, n = U.shape[0], V.shape[0]

    # Add priors
    U_ = U + torch.sqrt(tau)*torch.eye(m).cuda(args.gpu)
    V_ = V + torch.sqrt(tau)*torch.eye(n).cuda(args.gpu)

    # Covariances for Laplace
    U_inv = torch.inverse(V_)  # Interchanged since W is transposed
    V_inv = torch.inverse(U_)

    return [M_W, U_inv, V_inv]

@torch.no_grad()
def predict_lap(args, model, val_loader, M_W, U, V, n_samples=100, apply_softmax=True):
    M_W, U, V = torch.from_numpy(M_W).cuda(), \
                torch.from_numpy(U).cuda(), \
                torch.from_numpy(V).cuda()
    model.eval()

    py = []
    preds = []
    targets = []
    max_probs = []

    for x, y, _ in val_loader:
        x = x.cuda(args.gpu)
        targets.append(y)
        phi = model.feature_extractor(x).view(x.shape[0], -1)

        mu_pred = phi @ M_W
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample() / 1. # {(0.1, 0.5), (0.235, 0.4)}
            py_ += torch.softmax(f_s, 1) if apply_softmax else f_s
        
        py_ /= n_samples
        max_prob, pred = torch.max(py_, dim=1)
        preds.append(pred)
        max_probs.append(max_prob.view(-1))
        py.append(py_)
    
    py = torch.cat(py, dim=0)
    preds = torch.cat(preds, dim=0).detach().cpu()
    max_probs = torch.cat(max_probs, dim=0).detach().cpu().numpy()
    targets = torch.cat(targets, dim=0)
    correct = preds.eq(targets).numpy() # boolean vector
    entropy = Entropy(py.detach().cpu(), reduction='sum').numpy()

    out = {}
    out['entropies'] = entropy
    out['max_probs'] = max_probs
    out['correct'] = correct
    out['targets'] = targets

    return out

