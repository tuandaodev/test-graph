from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, load_DBLP, load_Yelp, load_Elliptic
from adagnn import AdaGNN
from pre_train_2_layer import pre_train
import matplotlib.pyplot as plt
import sys
print(sys.argv[0])

parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--layers', type=int, default=8,
                    help='Layer number of AdaGNN model.')
parser.add_argument('--dataset', type=str, 
                    help='dataset from {"DBLP", "Yelp"}.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,  # 0.0001
                    help='Initial learning rate.')
parser.add_argument('--mode', type=str, default='s',
                    help='Regularization of adjacency matrix in {"r", "s"}.')
parser.add_argument('--weight_decay', type=float, default=9e-12,  # 9e-12
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1norm', type=float, default=1e-6,  # 1e-6
                    help='L1 loss on Phi in each layer.')
parser.add_argument('--hidden', type=int, default=128,  # 16
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,  # 0.2
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset in ['DBLP']:
    adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label = load_DBLP(args.dataset, args.mode)
elif args.dataset in ['Yelp', 'Amazon']:
  adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label = load_Yelp(args.dataset, args.mode)
elif args.dataset in ['Elliptic']:
  adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label = load_Elliptic(args.dataset, args.mode)
else:
    print('No such dataset supported !')
    assert 0==1

model = AdaGNN(diag_dimension=features.shape[0], nfeat=features.shape[1],
                                                                 nhid=args.hidden, nlayer=args.layers,
                                                                 nclass=labels.max().item() + 1,
                                                                 dropout=args.dropout)
print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.layers >= 2:
    # Pre-train the first & last layer for faster convergence
    pre_train( args.dataset, args.hidden, adj, features, labels, idx_train, idx_val, gamma, minority_label)
    if args.mode == 's':
        model.load_state_dict(torch.load( args.dataset + '-2.pkl'),
                              strict=False)
    elif args.mode == 'r':
        model.load_state_dict(torch.load(args.dataset + '-2.pkl'),
                              strict=False)

stop_count = 0
val_loss_final = 0
last_loss = 1000
labels = labels.float()

def train(epoch):
    global val_loss_final
    global stop_count
    global last_loss

    t = time.time()
    the_l1 = 0

    for k, v in model.named_parameters():
        if 'learnable_diag' in k:
            the_l1 += torch.sum(abs(v))

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # Only consider minority labels for the target
    target = F.one_hot(labels[idx_train].long(), num_classes=2).float()
    minority_target = target[labels[idx_train] == minority_label]
    minority_output = output[idx_train][labels[idx_train] == minority_label]
    
    criterion_minority = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss_train =   F.nll_loss(output[idx_train], labels[idx_train].long()) +  criterion_minority(minority_output, minority_target) +  args.l1norm * the_l1
    
    #criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    #target = F.one_hot(labels[idx_train].long(), num_classes=2).float()
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train].long()) + criterion(output[idx_train], target) +  args.l1norm * the_l1
    acc_train, auc_roc_train, auc_pr_train = accuracy(output[idx_train], labels[idx_train])
    
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)
        output = output.float()

    target = F.one_hot(labels[idx_val].long(), num_classes=2).float()
    minority_target = target[labels[idx_val] == minority_label]
    minority_output = output[idx_val][labels[idx_val] == minority_label]

    loss_val = F.nll_loss(output[idx_val], labels[idx_val].long()) +  criterion_minority(minority_output, minority_target)   + args.l1norm * the_l1
    acc_val , auc_roc_val, auc_pr_val  = accuracy(output[idx_val], labels[idx_val])

    if loss_val.item() > last_loss:
        stop_count += 1
    else:
        stop_count = 0
    last_loss = loss_val.item()

    if epoch == 0:
        val_loss_final = loss_val.item()
    elif loss_val.item() < val_loss_final:
        val_loss_final = loss_val.item()
        torch.save(model.state_dict(), args.dataset + '-' + str(args.layers) + '.pkl')
        
    print('Epoch: {:04d}'.format(epoch+1),
      'Model training :',
      'loss_train: {:.4f}'.format(loss_train.item()),
      'acc_train: {:.4f}'.format(acc_train.item()),
      'auc_roc_train: {:.4f}'.format(auc_roc_train.item()),
      'auc_pr_train: {:.4f}'.format(auc_pr_train.item()),
      'Model validation :',
      'loss_val: {:.4f}'.format(loss_val.item()),
      'acc_val: {:.4f}'.format(acc_val.item()),
      'auc_roc_val: {:.4f}'.format(auc_roc_val.item()),
      'auc_pr_val: {:.4f}'.format(auc_pr_val.item()),
      'time: {:.4f}s'.format(time.time() - t), 
      sep='\n')
    if stop_count >= patience:  # 6
        print("Early stop  ! ")

def test():
    try:
        model.load_state_dict(torch.load(args.dataset + '-' + str(args.layers) + '.pkl'),
                              strict=True)
    except FileNotFoundError:
        model.load_state_dict(torch.load(args.dataset + '-' + str(2) + '.pkl'),
                              strict=False)
    model.eval()
    
t_total = time.time()
for epoch in range(args.epochs):
    model.train()
    train(epoch)
    if stop_count >= patience:
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()
