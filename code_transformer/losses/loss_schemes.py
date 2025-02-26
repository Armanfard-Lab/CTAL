#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTaskLoss(nn.Module):
    def __init__(self, tasks, losses):
        super(SingleTaskLoss, self).__init__()
        self.tasks = tasks
        self.losses = losses
    
    def forward(self, pred, gt):
        preds = {}
        preds['initial'] = pred
        preds['final'] = pred
        
        losses = {}
        losses['initial'] = {task: self.losses[task](pred[task], gt[task]) for task in self.tasks}
        losses['final'] = losses['initial']
        return losses, preds


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, losses: dict, loss_weighting: dict):
        super(MultiTaskLoss, self).__init__()
        self.tasks = tasks
        self.losses = losses
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {}
        preds['initial'] = pred
        preds['final'] = pred

        losses = {}
        losses['initial'] = {task: self.losses[task](pred[task], gt[task]) for task in self.tasks}
        losses['final'] = losses['initial']
        losses['total'] = self.loss_weighting(losses)
        return losses, preds


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, losses: dict, loss_weighting: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.losses = losses
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= pred[f'initial_{task}'], gt[task]
            loss_ = self.losses['initial'][task](pred_, gt_)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_


        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.losses['final'][task](pred_, gt_)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds

class PAPNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, losses: dict, loss_weighting: dict):
        super(PAPNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.losses = losses
        self.loss_weighting = loss_weighting
    
    # TODO: Add pair-wise loss
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= pred[f'initial_{task}'], gt[task]
            loss_ = self.losses['initial'][task](pred_, gt_)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_


        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.losses['final'][task](pred_, gt_)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds
    
class CTALLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, losses: dict, loss_weighting: dict):
        super(CTALLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.losses = losses
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= pred[f'initial_{task}'], gt[task]
            loss_ = self.losses['initial'][task](pred_, gt_)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_

        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.losses['final'][task](pred_, gt_)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds

class MTINetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, losses: dict, loss_weighting: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.losses = losses
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {t: 0. for t in self.auxilary_tasks}}

        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup)
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)]
            pred_scale = {t: F.interpolate(pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            if scale == 0:
                preds['initial'] = pred_scale
            losses_scale = {t: self.losses['initial'][t](pred_scale[t], gt[t]) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                losses['initial'][k] += v/4

        # Losses at output
        preds['final'] = {task: pred[task] for task in self.tasks}
        losses['final'] = {task: self.losses['final'][task](preds['final'][task], gt[task]) for task in self.tasks}

        losses['total'] = self.loss_weighting(losses)

        return losses, preds