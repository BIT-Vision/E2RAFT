import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_loss(flow_preds, flow_gt, valid, gamma):
    mask = (valid >= 0.5)

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (mask[:, None] * i_loss).mean()

    return flow_loss


def DCEI_loss(flow_preds, fmap2_gt, fmap2_pseudo, flow_gt, valid):
    gamma = 0.8
    flow_loss = 0.0
    mask = (valid >= 0.5)

    for i in range(len(flow_preds)):
        i_weight = gamma**(len(flow_preds) - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (mask[:, None] * i_loss).mean()

    if fmap2_pseudo is not None:
        if isinstance(fmap2_pseudo, list):
            for i in range(len(fmap2_pseudo)):
                i_weight = gamma**(len(fmap2_pseudo) - i - 1) if len(fmap2_pseudo) != 1 else 1.0
                i_loss = F.l1_loss(fmap2_pseudo[i], fmap2_gt[i]) * 10
                pseudo_loss += i_weight * i_loss
        else:
            pseudo_loss = F.l1_loss(fmap2_pseudo, fmap2_gt) * 10

        flow_loss += pseudo_loss

    return flow_loss
