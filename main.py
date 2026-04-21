# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import copy
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import logging

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')



def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    print("NB CLasses: ", args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        num_tasks=args.num_tasks,
        kernel_size=args.kernel_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        prompts_per_task=args.num_prompts_per_task,
        args=args
    )
    model.to(device)  
    original_model = copy.deepcopy(model)
    original_model.to(device)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    if args.freeze:
        
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                if n.find('norm1')>=0 or n.find('norm2')>=0:
                    # print(n)
                    pass
                else:
                    p.requires_grad = False
            #         print(n)

        # exit(0)
        
    
    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device,
                                  task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    criterion = torch.nn.CrossEntropyLoss().to(device)

    milestones = [18] if "CIFAR" in args.dataset else [40]
    lrate_decay = 0.1
    param_list = list(model.parameters())
 

    network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    
    if not args.SLCA:
        optimizer = create_optimizer(args, model)
        if args.sched != 'constant':
            # lr_scheduler, _ = create_scheduler(args, optimizer)
            # Create cosine lr scheduler
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.sched == 'constant':
            lr_scheduler = None
    else:
        optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    print("Started main")
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    print("Parser created: ", parser)

    print("Getting config")
    supported_configs = {
        'cifar100_convprompt',
        'imr_convprompt',
        'cub_convprompt',
        'cifar100_slca',
        'imr_slca',
        'cub_slca',
        'ucf101_convprompt',
        'activitynet_convprompt',
    }
    config = next((arg for arg in sys.argv[1:] if arg in supported_configs), None)
    if config is None:
        parser.error(f"missing config subcommand. Expected one of: {sorted(supported_configs)}")

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_convprompt':
        from configs.cifar100_convprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_convprompt', help='Split-CIFAR100 configs for ConvPrompt')
    elif config == 'imr_convprompt':
        from configs.imr_convprompt import get_args_parser
        config_parser = subparser.add_parser('imr_convprompt', help='Split-ImageNet-R configs for ConvPrompt')
    elif config == 'cub_convprompt':
        from configs.cub_convprompt import get_args_parser
        config_parser = subparser.add_parser('cub_convprompt', help='Split-CUB configs for ConvPrompt')
    elif config == 'cifar100_slca':
        from configs.cifar100_slca import get_args_parser
        config_parser = subparser.add_parser('cifar100_slca', help='Split-CIFAR100 SLCA configs')
    elif config == 'imr_slca':
        from configs.imr_slca import get_args_parser
        config_parser = subparser.add_parser('imr_slca', help='Split-ImageNet-R SLCA configs')
    elif config == 'cub_slca':
        from configs.cub_slca import get_args_parser
        config_parser = subparser.add_parser('cub_slca', help='Split-CUB SLCA configs')
    elif config == 'ucf101_convprompt':
        from configs.ucf101_convprompt import get_args_parser
        config_parser = subparser.add_parser('ucf101_convprompt', help='UCF101 ConvPrompt configs')
    elif config == 'activitynet_convprompt':
        from configs.activitynet_convprompt import get_args_parser
        config_parser = subparser.add_parser('activitynet_convprompt', help='ActivityNet ConvPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    print("Reached here")
    args, unknown_args = parser.parse_known_args()
    launcher_flags = {
        '--nproc_per_node',
        '--nnodes',
        '--node_rank',
        '--master_addr',
        '--master_port',
        '--local_rank',
        '--rdzv_backend',
        '--rdzv_endpoint',
        '--rdzv_id',
    }
    invalid_unknown = []
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        is_launcher_flag = any(
            token == flag or token.startswith(flag + '=')
            for flag in launcher_flags
        )
        if is_launcher_flag:
            if '=' not in token and (i + 1) < len(unknown_args) and not unknown_args[i + 1].startswith('-'):
                i += 2
            else:
                i += 1
            continue

        invalid_unknown.append(token)
        i += 1

    if invalid_unknown:
        parser.error('unrecognized arguments: {}'.format(' '.join(invalid_unknown)))
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Reached here")
    main(args)
    
    sys.exit(0)
