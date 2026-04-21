# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from timm.utils import accuracy

import utils


def _normalize_mask_indices(mask, num_classes):
    mask_arr = np.asarray(mask, dtype=np.int64)
    if mask_arr.size == 0:
        return mask_arr

    # Some dataset splits store labels as 1..N. Convert to 0..N-1 when detected.
    if mask_arr.min() >= 1 and mask_arr.max() <= num_classes and not np.any(mask_arr == 0):
        mask_arr = mask_arr - 1

    mask_arr = mask_arr[(mask_arr >= 0) & (mask_arr < num_classes)]
    if mask_arr.size == 0:
        return mask_arr

    return np.unique(mask_arr)


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None,):

    model.train(set_training_mode)
    if original_model is not None:
        original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    for step, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        is_multiseg = (input.ndim == 5)
        if is_multiseg:
            batch_size, num_segments, channels, height, width = input.shape
            flat_input = input.reshape(batch_size * num_segments, channels, height, width)
        else:
            flat_input = input
            batch_size = input.shape[0]
            num_segments = 1

        with torch.no_grad():
            if original_model is not None:
                output = original_model(flat_input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = model(flat_input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        if is_multiseg:
            logits = logits.reshape(batch_size, num_segments, -1).mean(dim=1)

        if args.train_mask and class_mask is not None:
            num_classes = logits.shape[1]
            mask = _normalize_mask_indices(class_mask[task_id], num_classes)
            if mask.size > 0:
                not_mask = np.setdiff1d(np.arange(num_classes), mask, assume_unique=True)
                if not_mask.size > 0:
                    not_mask = torch.tensor(not_mask, dtype=torch.int64, device=device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        loss = loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(data_loader):
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['Acc@5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    model.eval()
    if original_model is not None:
        original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            is_multiseg = (input.ndim == 5)
            if is_multiseg:
                batch_size, num_segments, channels, height, width = input.shape
                flat_input = input.reshape(batch_size * num_segments, channels, height, width)
            else:
                flat_input = input
                batch_size = input.shape[0]
                num_segments = 1

            if original_model is not None:
                output = original_model(flat_input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output = model(flat_input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if is_multiseg:
                logits = logits.reshape(batch_size, num_segments, -1).mean(dim=1)

            if args.task_inc and class_mask is not None:
                num_classes = logits.shape[1]
                mask = _normalize_mask_indices(class_mask[task_id], num_classes)
                if mask.size > 0:
                    mask = torch.tensor(mask, dtype=torch.int64, device=device)
                    logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                    logits_mask = logits_mask.index_fill(1, mask, 0.0)
                    logits = logits + logits_mask

            loss = criterion(logits, target)
            predicts = torch.max(logits, dim=1)[1]
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks))

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2]
    )
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device, class_mask=None, args=None,):

    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            if model.module.e_prompt.prompt.grad is not None:
                                model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            if model.e_prompt.prompt.grad is not None:
                                model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()

        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = slice(cur_start, cur_end)
                    prev_idx = slice(prev_start, prev_end)

                    with torch.no_grad():
                        if args.distributed:
                            if model.module.e_prompt.prompt_key.grad is not None:
                                model.module.e_prompt.prompt_key.grad.zero_()
                            model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            if model.e_prompt.prompt_key.grad is not None:
                                model.e_prompt.prompt_key.grad.zero_()
                            model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()

        if task_id > 0 and args.reinit_optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if not args.SLCA else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device,
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
