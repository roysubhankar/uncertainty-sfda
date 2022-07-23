import os, sys
import numpy as np
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_list import trainval_loader
from loss import CrossEntropyLabelSmooth, Entropy
from laplace import get_hessian, estimate_variance, predict_lap

parser = argparse.ArgumentParser()
# dataset args
parser.add_argument('--s', choices=['CIFAR9', 'STL9'], default='CIFAR9', help='Source domain')
parser.add_argument('--t', choices=['STL9', 'CIFAR9'], default='STL9', help='Target domain')
parser.add_argument('--root_dir', type=str, default='data/', help='Path to the root folder')
parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
parser.add_argument('--download', action='store_true', help='Download flag for datasets')

# model args
parser.add_argument('--model', type=str, default='cnn13', help='Network architecture to use')

# training args
parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per mini-batch')
parser.add_argument('--num_workers', type=int, default=4, help='Number of pipeline threads/workers')
parser.add_argument('--epochs', type=int, default=20, help='Number of total epochs to run')
parser.add_argument('--print_freq', type=int, default=100, help='Print every print_freq steps')
parser.add_argument('--lr', type=float, default=0.01, help='Learing rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
parser.add_argument('--var0', type=float, default=1e-3, help='precision')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--mode', type=str, default='source', choices=['source', 'target'], 
                    help='Source training or Target adaptation')
parser.add_argument('--method', type=str, default='shot', choices=['shot', 'usfan'])
parser.add_argument('--smooth', type=float, default=0.1, help='Label smoothing')
parser.add_argument('--epsilon', type=float, default=1e-5)

# logging
parser.add_argument('--model_dir', type=str, default='checkpoints/', help='Folder to save the checkpoint')

def main():
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)
    
    # define the model
    print(f"==> Creating model {args.model}...")
    model = __import__(args.model).__dict__[args.model.upper()](num_classes=args.num_classes, bias=False)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    print(model)

    # dataset and dataloaders
    dset_loaders = {}
    src_train_loader, src_lap_loader, src_val_loader = trainval_loader(args, dset=args.s)
    dset_loaders['src_train_loader'] = src_train_loader
    dset_loaders['src_lap_loader'] = src_lap_loader
    dset_loaders['src_val_loader'] = src_val_loader
    tgt_train_loader, tgt_lap_loader, tgt_val_loader = trainval_loader(args, dset=args.t)
    dset_loaders['tgt_train_loader'] = tgt_train_loader
    dset_loaders['tgt_lap_loader'] = tgt_lap_loader
    dset_loaders['tgt_val_loader'] = tgt_val_loader

    if args.mode == 'source':
        # train on source domain
        if not os.path.exists(os.path.join(args.model_dir, args.s + '.pt')):
            source_train(args, dset_loaders, model)
        
        # load source trained model from checkpoint
        modelpath = os.path.join(args.model_dir, args.s + '.pt')
        print(f'Loading checkpoint of {modelpath}')
        model.load_state_dict(torch.load(modelpath))

        # compute Laplace approximation
        print(f'Computing the Laplace Approximation for Bayesian hypothesis generation...')
        compute_laplace(args, model, dset_loaders['src_lap_loader'])

        # test on target domain
        model.eval()
        acc = calc_acc(args, dset_loaders['tgt_val_loader'], model)
        print(f'Task: {args.t}, Accuracy: {acc * 100:.2f}%')

    elif args.mode == 'target':
        # load source trained model from checkpoint
        modelpath = os.path.join(args.model_dir, args.s + '.pt')
        model.load_state_dict(torch.load(modelpath))

        # train on the target domain
        if args.method == 'shot':
            tgt_model = shot(args, dset_loaders, model)
        elif args.method == 'usfan':
            tgt_model = shot_lap(args, dset_loaders, model)
        
        # test on target domain
        tgt_model.eval()
        acc = calc_acc(args, dset_loaders['tgt_val_loader'], tgt_model)
        print(f'Task: {args.t}, Accuracy: {acc * 100:.2f}%')

def shot_lap(args, dset_loaders, model):
    """
    Implements U-SFAN proposed in our paper
    """
    for k, v in model.named_parameters():
        if 'fc' in k:
            v.requires_grad = False
    
    optimizer = torch.optim.SGD(model.feature_extractor.parameters(),
                                lr=args.lr * 0.1, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizer = op_copy(optimizer)

    # switch to train mode
    model.train()
    num_batches = len(dset_loaders['tgt_train_loader'])
    max_iter = args.epochs * num_batches
    interval_iter = max_iter // 10
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["tgt_train_loader"])
            inputs_test, _, tar_idx = iter_test.next()
        
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            model.eval()
            print(f'Loading Hessians for uncertainty weights...')
            M_W, U, V = list(np.load(os.path.join(args.model_dir, args.s + "_llla.npy"), allow_pickle=True))
            out = predict_lap(args, model, dset_loaders['tgt_lap_loader'], M_W, U, V)
            model.train()
            mem_metric = torch.from_numpy(out['entropies']).cuda()
        
        inputs_test = inputs_test.cuda(args.gpu)
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)
        outputs_test = model(inputs_test)

        # get the uncertainty weights
        la_ent = mem_metric[tar_idx]
        unknown_weight = torch.exp(-la_ent).detach()

        # entropy loss
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        entropy_loss = Entropy(softmax_out, reduction='none')
        entropy_loss = torch.sum(entropy_loss * unknown_weight.view(-1, 1), dim=1).mean()
        # divergence loss
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        entropy_loss -= gentropy_loss

        optimizer.zero_grad()
        entropy_loss.backward()
        optimizer.step()

        if iter_num % args.print_freq == 0:
            print(f'Task: {args.s}->{args.t}, Iter: {iter_num}/{max_iter}, Loss: {entropy_loss.item():.4f}')
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc = calc_acc(args, dset_loaders['tgt_val_loader'], model)
            print(f'Validatiion: {args.t}, Iter: {iter_num}/{max_iter}, Accuracy: {acc * 100:.2f}%')
            model.train()
        
    save_checkpoints(args, model, fname=args.s + '_' + args.t + '.pt')

    return model

def shot(args, dset_loaders, model):
    """
    Implements the SHOT-IM baseline
    """
    for k, v in model.named_parameters():
        if 'fc' in k:
            v.requires_grad = False
    
    optimizer = torch.optim.SGD(model.feature_extractor.parameters(),
                                lr=args.lr * 0.1, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer = op_copy(optimizer)

    # switch to train mode
    model.train()
    num_batches = len(dset_loaders['tgt_train_loader'])
    max_iter = args.epochs * num_batches
    interval_iter = max_iter // 10
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["tgt_train_loader"])
            inputs_test, _, tar_idx = iter_test.next()
        
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda(args.gpu)
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_test = model(inputs_test)
        
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # entropy loss
        entropy_loss = torch.mean(Entropy(softmax_out, reduction='sum'))
        # divergence loss
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        entropy_loss -= gentropy_loss

        optimizer.zero_grad()
        entropy_loss.backward()
        optimizer.step()

        if iter_num % args.print_freq == 0:
            print(f'Task: {args.s}->{args.t}, Iter: {iter_num}/{max_iter}, Loss: {entropy_loss.item():.4f}')
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc = calc_acc(args, dset_loaders['tgt_val_loader'], model)
            print(f'Validatiion: {args.t}, Iter: {iter_num}/{max_iter}, Accuracy: {acc * 100:.2f}%')
            model.train()
    
    save_checkpoints(args, model, fname=args.s + '_' + args.t + '.pt')
    
    return model
    
def source_train(args, dset_loaders, model):
    param_group = []
    for k, v in model.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # switch to train mode
    model.train()
    num_batches = len(dset_loaders['src_train_loader'])
    max_iter = args.epochs * num_batches
    interval_iter = max_iter // 10
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(dset_loaders["src_train_loader"])
            inputs_source, labels_source, _ = iter_source.next()
        
        if inputs_source.size(0) == 1:
            continue
        
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(args.gpu), labels_source.cuda(args.gpu)
        outputs_source = model(inputs_source)

        classifier_loss = CrossEntropyLabelSmooth(
                num_classes=args.num_classes, 
                epsilon=args.smooth
            )(outputs_source, labels_source)
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % args.print_freq == 0:
            print(f'Task: {args.s}, Iter: {iter_num}/{max_iter}, Loss: {classifier_loss.item():.4f}')
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc = calc_acc(args, dset_loaders['src_val_loader'], model)
            print(f'Validatiion: {args.s}, Iter: {iter_num}/{max_iter}, Accuracy: {acc * 100:.2f}%')
            model.train()
    # save model checkpoints
    save_checkpoints(args, model, fname=args.s + '.pt')

def calc_acc(args, val_loader, model):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.cuda(args.gpu)
            output = model(images)
            output_prob = F.softmax(output, dim=1)
            preds = torch.max(output_prob, dim=1)[1]
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = torch.sum(torch.squeeze(all_preds).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(args, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = args.weight_decay
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def save_checkpoints(args, model, fname):
    torch.save(model.state_dict(), os.path.join(args.model_dir, fname))

def vanilla_pred(args, model, dset_loader):
    preds = []
    targets = []
    entropies = []
    max_probs = []
    model.eval()
    
    with torch.no_grad():
        for (images, labels, _) in tqdm(dset_loader, desc="Preparing histogram"):
            images = images.cuda()
            # compute output
            output = model(images)
            # compute entropy for every sample
            entropy = Entropy(F.softmax(output, dim=-1), reduction='sum')
            entropies.append(entropy)
            # get the model predictions
            max_prob, pred = torch.max(F.softmax(output, dim=-1), dim=1)
            preds.append(pred)
            targets.append(labels)
            max_probs.append(max_prob.view(-1))
    
    preds = torch.cat(preds, dim=0).detach().cpu()
    max_probs = torch.cat(max_probs, dim=0).detach().cpu().numpy()
    targets = torch.cat(targets, dim=0)
    entropies = torch.cat(entropies, dim=0).detach().cpu().numpy()
    correct = preds.eq(targets).numpy() # boolean vector
    out = {}
    out['max_probs'] = max_probs
    out['entropies'] = entropies
    out['correct'] = correct
    out['targets'] = targets

    return out

def compute_laplace(args, model, train_loader):
    # compute the laplace approximations
    hessians = get_hessian(args, model, train_loader)
    var0 = torch.tensor(args.var0).float().cuda() # 5e-4, 0.001, 0.01, 0.2656
    M_W, U, V = estimate_variance(args, var0, hessians)
    print(f'Saving the hessians...')
    M_W, U, V = M_W.detach().cpu().numpy(), U.detach().cpu().numpy(), \
                        V.detach().cpu().numpy()
    np.save(os.path.join(args.model_dir, args.s + "_llla.npy"), [M_W, U, V])

if __name__ == '__main__':
    main()