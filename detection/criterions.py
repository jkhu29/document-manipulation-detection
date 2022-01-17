import torch


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels, loss=torch.nn.BCELoss()):
    loss0 = loss(d0, labels)
    loss1 = loss(d1, labels)
    loss2 = loss(d2, labels)
    loss3 = loss(d3, labels)
    loss4 = loss(d4, labels)
    loss5 = loss(d5, labels)
    loss6 = loss(d6, labels)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss
