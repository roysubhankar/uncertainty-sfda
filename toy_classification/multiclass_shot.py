import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
import matplotlib.cm as cm
from sklearn import datasets
from math import *
from argparse import ArgumentParser
import seaborn as sns; sns.set_style('white')
from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal
import tikzplotlib
from matplotlib import rc
rc("text", usetex=False)

matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['font.size'] = 14
#matplotlib.rcParams['font.family'] = "serif"
#matplotlib.rcParams['font.serif'] = 'Times'
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.linewidth'] = 1.0
plt = matplotlib.pyplot

cmap = np.asarray(["b", "r", "g"])

def shift_data(x_in, ti=None, ri=None, si=None):
    x_out = x_in
    
    xmean = np.mean(x_out, axis=0)
    x_out -= xmean
    
    if si is not None and si > 0:
        s_mat = si * np.eye(x_in.shape[1])
        x_out = x_out @ s_mat
    
    if ti is not None:
        x_out = x_out + ti
    
    if ri is not None:
        theta = np.radians(-ri)
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        x_out = x_out @ rot_mat
    
    return x_out + xmean

class Model(nn.Module):
    
    def __init__(self, n, h, k):
        super(Model, self).__init__()

        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(), 
            nn.Linear(h, h), 
            nn.BatchNorm1d(h),
            nn.ReLU()
        )

        self.clf = nn.Linear(h, k, bias=False)
    
    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)

def plot(X, Y, X1_test, X2_test, Z, class_mask, test_range, fname='figs/original_multiclass.png',
         source=False):
    """
    Plots the data points and the confidence in predictions
    """
    cmap_contour = 'Blues'
    plt.figure(figsize=(6, 5))

    plt.imshow((Z) * (class_mask == 0), cmap='Blues',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 0),
        vmin=0.5, vmax=1.0)
    plt.imshow((Z) * (class_mask == 1), cmap='Reds',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 1),
        vmin=0.5, vmax=1.0)
    plt.imshow((Z) * (class_mask == 2), cmap='Greens',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 2),
        vmin=0.5, vmax=1.0)
    
    if source: # solid source data points
        plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], c=cmap[0], edgecolors='k', linewidths=0.5)
        plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], c=cmap[1], edgecolors='k', linewidths=0.5)
        plt.scatter(X[Y==2][:, 0], X[Y==2][:, 1], c=cmap[2], edgecolors='k', linewidths=0.5)
    else: # hollow target data points
        plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], facecolor='none', edgecolors=cmap[0], linewidths=0.5)
        plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], facecolor='none', edgecolors=cmap[1], linewidths=0.5)
        plt.scatter(X[Y==2][:, 0], X[Y==2][:, 1], facecolor='none', edgecolors=cmap[2], linewidths=0.5)

    plt.xlim(test_range)
    plt.ylim(test_range)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(fname, bbox_inches='tight')

def plot_decision_boundary(X, Y, X1_test, X2_test, Z, class_mask, test_range, 
                           fname='figs/original_multiclass.png',
                           source=False):
    """
    Plots the data points and the confidence in predictions
    """
    plt.figure(figsize=(6, 5))
    
    plt.imshow((Z) * (class_mask == 0), cmap='Blues',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 0),
        vmin=0.5, vmax=1.0)
    plt.imshow((Z) * (class_mask == 1), cmap='Reds',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 1),
        vmin=0.5, vmax=1.0)
    plt.imshow((Z) * (class_mask == 2), cmap='Greens',origin='lower', 
        extent=[-15, 15, -15, 15], alpha = 0.4 * (class_mask == 2),
        vmin=0.5, vmax=1.0)
    
    if source: # solid source data points
        plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], c=cmap[0], edgecolors='k', linewidths=0.5)
        plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], c=cmap[1], edgecolors='k', linewidths=0.5)
        plt.scatter(X[Y==2][:, 0], X[Y==2][:, 1], c=cmap[2], edgecolors='k', linewidths=0.5)
    else: # hollow target data points
        plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], facecolor='none', edgecolors=cmap[0], linewidths=0.5)
        plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], facecolor='none', edgecolors=cmap[1], linewidths=0.5)
        plt.scatter(X[Y==2][:, 0], X[Y==2][:, 1], facecolor='none', edgecolors=cmap[2], linewidths=0.5)

    plt.xlim(test_range)
    plt.ylim(test_range)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(fname, bbox_inches='tight')

def Entropy(input_, reduction='sum'):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if reduction == 'sum':
        entropy = torch.sum(entropy, dim=1)
    else:
        return entropy
    return entropy 

def source_train(X, Y, model, opt):
    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(Y).long()
    model.train()

    for it in range(5000):
        y = model(X_train)
        l = F.cross_entropy(y, y_train)
        l.backward()
        opt.step()
        opt.zero_grad()
            
    print(f'Loss: {l.item():.3f}')

    return model

def shot(X, model, opt):
    X_train = torch.from_numpy(X).float()
    model.train()

    for it in range(500):
        y = model(X_train)
        y_prob = nn.Softmax(dim=1)(y)
        ent_loss = torch.mean(Entropy(y_prob, reduction='sum'))
        msoftmax = y_prob.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        ent_loss -= gentropy_loss

        opt.zero_grad()
        ent_loss.backward()
        opt.step()
    
    print(f'IM Loss: {ent_loss.item():.3f}')

    return model

def get_laplace(X, Y, model, weight_decay):
    """
    Compute the laplace approximation
    """
    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(Y).long()

    W = list(model.parameters())[-1]
    shape_W = W.shape

    # Use BackPACK to get the Kronecker-factored last-layer covariance
    extend(model.clf)
    loss_func = extend(nn.CrossEntropyLoss(reduction='sum'))

    loss = loss_func(model(X_train), y_train)
    with backpack(extensions.KFAC()):
        loss.backward()
    
    # The Kronecker-factored Hessian of the negative log-posterior
    A, B = W.kfac
    #A, B = np.sqrt(X_train.shape[0]) * A, np.sqrt(X_train.shape[0]) * B

    # The weight decay used for training is the Gaussian prior's precision
    prec0 = weight_decay

    # The posterior covariance's Kronecker factors
    U = torch.inverse(A + sqrt(prec0)*torch.eye(shape_W[0]))
    V = torch.inverse(B + sqrt(prec0)*torch.eye(shape_W[1]))

    return U, V

def get_laplace_predictions(X, Y, X_test, model, weight_decay):
    """
    Computes the laplace model predictions
    """
    W = list(model.parameters())[-1]
    U, V = get_laplace(X, Y, model, weight_decay)

    with torch.no_grad():
        phi = model.feature_extr(X_test)

        # MAP prediction
        m = phi @ W.T

        # v is the induced covariance. 
        # See Appendix B.1 of https://arxiv.org/abs/2002.10118 for the detail of the derivation.
        v = torch.diag(phi @ V @ phi.T).reshape(-1, 1, 1) * U

        # The induced distribution over the output (pre-softmax)
        output_dist = MultivariateNormal(m, v)

        # MC-integral
        n_sample = 10000
        py = 0

        for _ in range(n_sample):
            out_s = output_dist.rsample()
            py += torch.softmax(out_s, 1)
        
        py /= n_sample
    
    return py


def shot_lap(X, Y, Xt, model, opt, weight_decay):

    # get Laplace predictions for the Xt
    Xt = torch.from_numpy(Xt).float()
    py = get_laplace_predictions(X, Y, Xt, model, weight_decay)
    wt = torch.exp(-Entropy(py, reduction='sum'))

    # train with shot
    model.train()

    for it in range(500):
        y = model(Xt)
        y_prob = nn.Softmax(dim=1)(y)

        ent_loss = torch.sum(Entropy(y_prob) * wt.view(-1, 1).detach(), dim=1).mean()
        msoftmax = y_prob.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        ent_loss -= gentropy_loss

        opt.zero_grad()
        ent_loss.backward()
        opt.step()
    
    print(f'IM Loss: {ent_loss.item():.3f}')

    return model

if __name__ == '__main__':
    parser = ArgumentParser(description='toy')
    parser.add_argument('--strong_shift', action='store_true', default=False, help='simulate strong data shift')
    parser.add_argument('--method', choices=['shot', 'usfan'], default='usfan')
    args = parser.parse_args()

    np.random.seed(777)
    torch.manual_seed(99999)

    # source data points
    tr_size = 500
    train_range = (-10, 10)
    test_range = (-15, 15)
    X, Y = datasets.make_blobs(n_samples=tr_size, centers=3, cluster_std=1.3, 
                           center_box=train_range, random_state=37) #11
    
    # target data points
    Xt = np.zeros((X.shape[0],2))
    
    if args.strong_shift: # for strong data shift
        Xt[Y==0] = shift_data(X[Y==0], ti=-10, ri=10, si=1.2) + np.random.normal(0, 1.5, (X[Y==0].shape[0], 2)) # magenta
        Xt[Y==1] = shift_data(X[Y==1], ti=2.3, ri=-10, si=1.2) + np.random.normal(0, 1.5, (X[Y==1].shape[0], 2)) # red
        Xt[Y==2] = shift_data(X[Y==2], ti=3.3, ri=-10, si=1.2) + np.random.normal(0, 1.5, (X[Y==2].shape[0], 2)) # gold
    else:
        Xt[Y==0] = shift_data(X[Y==0], ti=-1, ri=10, si=1.2) + np.random.normal(0, 1.5, (X[Y==0].shape[0], 2)) # magenta
        Xt[Y==1] = shift_data(X[Y==1], ti=1.3, ri=-10, si=1.2) + np.random.normal(0, 1.5, (X[Y==1].shape[0], 2)) # red
        Xt[Y==2] = shift_data(X[Y==2], ti=1.5, ri=-10, si=1.2) + np.random.normal(0, 1.5, (X[Y==2].shape[0], 2)) # gold
    Yt = Y

    ######### Test data grid #########
    size = 100
    test_rng = np.linspace(*test_range, size)

    X1_test, X2_test = np.meshgrid(test_rng, test_rng)
    X_test = np.stack([X1_test.ravel(), X2_test.ravel()]).T
    X_test = torch.from_numpy(X_test).float()

    # source training
    M, N = X.shape
    H = 20  # num. hidden units
    K = 3  # num. classes
    WEIGHT_DECAY = 3e-3
    LR = 1e-3
    MOMENTUM = 0.9

    model = Model(n=N, h=H, k=K) 
    opt = optim.SGD(model.parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    model = source_train(X, Y, model, opt)
    
    ############ ReLU Networks ##############
    #### evaluate and visualize source model predictions ####
    model.eval()

    with torch.no_grad():
        py_map = F.softmax(model(X_test), dim=1).squeeze()

    conf_relu, class_mask = torch.max(py_map, 1)
    conf_relu, class_mask = conf_relu.numpy(), class_mask.numpy()

    folder = args.method
    os.makedirs(folder, exist_ok=True)

    # plot source points and confidence
    plot(X, Y, X1_test, X2_test, 
        conf_relu.reshape(size, size),
        class_mask.reshape(size, size),
        test_range,
        fname=os.path.join(folder, 'source_relu.png'),
        source=True)
    # plot target points and confidence
    plot(Xt, Yt, X1_test, X2_test, 
        conf_relu.reshape(size, size), 
        class_mask.reshape(size, size),
        test_range,
        fname=os.path.join(folder,'target_relu.png'))

    ############ Laplace Approximation ##############
    py_map = get_laplace_predictions(X, Y, X_test, model, WEIGHT_DECAY)
    conf_relu, class_mask = torch.max(py_map, 1)
    conf_relu, class_mask = conf_relu.numpy(), class_mask.numpy()

    # plot source points and confidence
    plot(X, Y, X1_test, X2_test, 
        conf_relu.reshape(size, size),
        class_mask.reshape(size, size),
        test_range,
        fname=os.path.join(folder, 'source_laplace.png'),
        source=True)
    # plot target points and confidence
    plot(Xt, Yt, X1_test, X2_test, 
        conf_relu.reshape(size, size), 
        class_mask.reshape(size, size),
        test_range,
        fname=os.path.join(folder, 'target_laplace.png'))

    #############################
    ##### Target adaptation #####
    #############################

    # freeze the hypothesis
    for k, v in model.named_parameters():
        if 'clf' in k:
            v.requires_grad = False
    # exlcude the hypothesis from the optimizer
    opt = optim.SGD(model.feature_extr.parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    if args.method == 'shot':
        model = shot(Xt, model, opt)
    elif args.method == 'usfan':
        # uncertainty guided shot
        model = shot_lap(X, Y, Xt, model, opt, WEIGHT_DECAY)
    
    ##### eval and visualize with the target model #####
    model.eval()
    with torch.no_grad():
        py_map = F.softmax(model(X_test), dim=1).squeeze()

    conf_relu, class_mask = torch.max(py_map, 1)
    conf_relu, class_mask = conf_relu.numpy(), class_mask.numpy()

    # plot target points and confidence
    plot_decision_boundary(Xt, Yt, X1_test, X2_test, 
        conf_relu.reshape(size, size), 
        class_mask.reshape(size, size),
        test_range, fname=os.path.join(folder, 'target_adapted_' + args.method +  '.png'))
    

    