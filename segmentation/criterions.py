"""
cited: https://github.com/CoinCheung/pytorch-loss/blob/master/lovasz_softmax.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszSoftmax(nn.Module):
    '''
    This is the autograd version, used in the multi-category classification case
    '''
    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmax()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float() # use fp32 to avoid nan
        label = label.view(-1)

        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]

        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(0, label.unsqueeze(0), 1).detach()

        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)

        # lovasz extension grad
        with torch.no_grad():
            #  lb_one_hot_sort = lb_one_hot[
            #      torch.arange(c).unsqueeze(1).repeat(1, n_samples), errs_order
            #      ].detach()
            lb_one_hot_sort = torch.cat([
                lb_one_hot[i, ord].unsqueeze(0)
                for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1. - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1. - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]

        losses = torch.einsum('ab,ab->a', errs_sort, jacc)

        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses, errs
