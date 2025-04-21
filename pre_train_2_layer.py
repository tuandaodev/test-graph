from __future__ import division
from __future__ import print_function
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import accuracy
from adagnn import AdaGNN


def pre_train(dataset, hidden, adj, features, labels, idx_train, idx_val, gamma, minority_label):

    stop_count = 0
    val_loss_final = 0
    last_loss = 1000

    model = AdaGNN(diag_dimension=features.shape[0],
                                 nfeat=features.shape[1],
                                 nhid=hidden, nlayer=2,
                                 nclass=labels.max().item() + 1,
                                 dropout=0.5)

    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=9e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)

    for epoch in range(10):
        t = time.time()
        the_l1 = 0

        for k, v in model.named_parameters():
            if 'learnable_diag' in k:
                the_l1 += torch.sum(abs(v))

        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        
        target = F.one_hot(labels[idx_train].long(), num_classes=2).float()
        minority_target = target[labels[idx_train] == minority_label]
        minority_output = output[idx_train][labels[idx_train] == minority_label]
    
        criterion_minority = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].long()) +  criterion_minority(minority_output, minority_target) +  1e-6 * the_l1
        acc_train, auc_roc_train , auc_pr_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        target = F.one_hot(labels[idx_val].long(), num_classes=2).float()
        minority_target = target[labels[idx_val] == minority_label]
        minority_output = output[idx_val][labels[idx_val] == minority_label]

        loss_val =  F.nll_loss(output[idx_val], labels[idx_val].long()) + criterion_minority(minority_output, minority_target)  + 1e-6 * the_l1
        acc_val, auc_roc_val, auc_pr_val = accuracy(output[idx_val], labels[idx_val])

        if loss_val.item() > last_loss:
            stop_count += 1
        else:
            stop_count = 0
        last_loss = loss_val.item()

        if epoch == 0:
            val_loss_final = loss_val.item()
        elif loss_val.item() < val_loss_final:  # and epoch >= 100:
            val_loss_final = loss_val.item()
            torch.save(model.state_dict(), dataset + '-' + str(2) + '.pkl')

        print('Epoch: {:04d}'.format(epoch+1),
      'loss_train: {:.4f}'.format(loss_train.item()),
      'acc_train: {:.4f}'.format(acc_train),
      'auc_roc_train: {:.4f}'.format(auc_roc_train),
      'auc_pr_train: {:.4f}'.format(auc_pr_train),
      'loss_val: {:.4f}'.format(loss_val.item()),
      'acc_val: {:.4f}'.format(acc_val),
      'auc_roc_val: {:.4f}'.format(auc_roc_val),
      'auc_pr_val: {:.4f}'.format(auc_pr_val),
      'time: {:.4f}s'.format(time.time() - t), 
      sep='\n')

        if stop_count >= 300 and epoch > 20:
            print("Early stop - pretraining process finished ! ")
            return 0
        scheduler.step()
    print("Pretraining process finished ! ")


