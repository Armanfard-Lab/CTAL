import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import random
from tqdm import tqdm # type: ignore
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms
from PIL import Image
from evaluation.evaluate_utils import PerformanceMeter
from torch.cuda.amp import GradScaler
#import cv2

"""
Define task metrics, loss functions and model trainer here.
"""
def get_output(output, task):
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def to_cuda(batch, device):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v, device)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(device=device, non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v, device) for v in batch]
    else:
        return batch

def multitask_trainer(train_loader, test_loader, model, device, opt):
    scaler = GradScaler() if opt.grad_scaling else None
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    for index in range(opt.epochs):
        performance_meter_test_initial = PerformanceMeter(opt, opt.TASKS.NAMES)
        performance_meter_test_final = PerformanceMeter(opt, opt.TASKS.NAMES)

        train_loss_initial = {f'mt_train_loss_{t}': 0 for t in opt.TASKS.NAMES}
        train_loss_final = {f'train_loss_{t}': 0 for t in opt.TASKS.NAMES}
        
        # iteration for all batches
        model.train()
        train_dataset = iter(train_loader)
        for k in tqdm(range(train_batch), desc='Training'):
            x = to_cuda(next(train_dataset), device)

            mt_train_loss, mt_train_pred, rf_train_loss, rf_train_pred = model.mt_step(x, scaler, train=True)

            train_loss_initial = {f'mt_train_loss_{t}': train_loss_initial[f'mt_train_loss_{t}']+mt_train_loss[t] for t in opt.TASKS.NAMES}
            train_loss_final = {f'train_loss_{t}': train_loss_final[f'train_loss_{t}']+rf_train_loss[t] for t in opt.TASKS.NAMES}   

        # evaluating test data
        model.eval()
        with torch.no_grad():
            test_loss_initial = {f'mt_test_loss_{t}': 0 for t in opt.TASKS.NAMES}
            test_loss_final = {f'test_loss_{t}': 0 for t in opt.TASKS.NAMES}
            test_dataset = iter(test_loader)
            for k in tqdm(range(test_batch), desc='Validating'):
                x = to_cuda(next(test_dataset), device)
                
                mt_test_loss, mt_test_pred, rf_test_loss, rf_test_pred = model.mt_step(x, train=False)

                test_loss_initial = {f'mt_test_loss_{t}': test_loss_initial[f'mt_test_loss_{t}']+mt_test_loss[t] for t in opt.TASKS.NAMES}
                test_loss_final = {f'test_loss_{t}': test_loss_final[f'test_loss_{t}']+rf_test_loss[t] for t in opt.TASKS.NAMES}

                performance_meter_test_initial.update({t: get_output(mt_test_pred[t], t) for t in opt.TASKS.NAMES}, {t: x[t] for t in opt.TASKS.NAMES})
                performance_meter_test_final.update({t: get_output(rf_test_pred[t], t) for t in opt.TASKS.NAMES}, {t: x[t] for t in opt.TASKS.NAMES})

        eval_results_initial = performance_meter_test_initial.get_score()
        eval_results_final = performance_meter_test_final.get_score()
        
        # print
        print_losses(train_loss_initial, train_batch, index, tag='TRAIN (IP)')
        print_losses(train_loss_final, train_batch, index, tag='TRAIN (FP)')
        print_losses(test_loss_initial, test_batch, index, tag='TEST (IP) ')
        print_losses(test_loss_final, test_batch, index, tag='TEST (FP) ')

        print_perf_metrics(opt, eval_results_initial, index, tag='mt_test')
        print_perf_metrics(opt, eval_results_final, index, tag='test')

        # checkpoint
        if opt.storage_root is not None and (index+1) % opt.save_epoch == 0:
            path = f'{opt.storage_root}/{opt.train_db_name}_{opt.model}_{opt.name}_{index}.pth'
            torch.save(model.state_dict(), path)
            print(f'Checkpoint saved to: {path}')
        
        # scheduler step
        if opt.setup == 'multi_task':
            if model.scheduler is not None:
                if opt.scheduler == 'cosine_wr':
                    model.scheduler.step(index + k/train_batch)
                else:
                    model.scheduler.step()
        else:
            for scheduler in model.scheduler.values():
                if scheduler is not None:
                    if opt.scheduler == 'cosine_wr':
                        scheduler.step(index + k/train_batch)
                    else:
                        scheduler.step()

def print_losses(losses, batches, epoch, tag):
    out = f'EPOCH: {epoch} | {tag} '
    for k, v in losses.items():
        task = k.split('_')[-1] 
        out += f'{task}: {v/batches:.4f} | '
    print(out)

def print_perf_metrics(opt, metrics, epoch, tag='test'):
    if 'semseg' in opt.TASKS.NAMES:
        stats = metrics['semseg']['mIoU']
        print(f'{tag}_semseg_miou: {stats}')
    if 'human_parts' in opt.TASKS.NAMES:
        stats = metrics['human_parts']['mIoU']
        print(f'{tag}_human_parts_mIoU: {stats}')
    if 'sal' in opt.TASKS.NAMES:
        stats = metrics['sal']['maxF']
        print(f'{tag}_sal_maxF: {stats}')
    if 'edge' in opt.TASKS.NAMES:
        stats = metrics['edge']['loss']
        print(f'{tag}_edge_val_loss: {stats}')
    if 'normals' in opt.TASKS.NAMES:
        stats = metrics['normals']['mean']
        print(f'{tag}_normals_mean: {stats}')
    if 'depth' in opt.TASKS.NAMES:
        stats = metrics['depth']['rel']
        print(f'{tag}_depth_rel_err: {stats}')
