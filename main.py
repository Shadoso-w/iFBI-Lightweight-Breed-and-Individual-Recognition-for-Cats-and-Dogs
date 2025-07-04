# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.facenet import Facenet
from model.fusion_mobilenet_temp import Fusion_MobileNetV3

# from pair_dataset import build_dataset
from dataset.triplet_dataset import build_dataset
# from dataset.random_sample_dataset import build_dataset
from engine import train_one_epoch, evaluate
# from engine_temp import train_one_epoch, evaluate

from collections import OrderedDict
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import wandb

from losses import SupConLoss, SupConLoss_clear, ContrastiveLoss, TripletLoss

import sys
sys.path.insert(0, './model')

## Identify the GPU used
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=180, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='fusion_mobilenet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.6, type=float)

    # EMA related parameters # Exponential Moving Average(EMA), for optimizer
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=1e-3, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--step_size', type=int, default=20, metavar='N',
                        help='num or steps to learning rate cyclicity')
    parser.add_argument('--start_step', type=int, default=150,
                        help='the first decayed epoch of multistep scheduler')
    parser.add_argument('--middle_step', type=int, default=200,
                        help='the second decayed epoch of multistep scheduler')
    parser.add_argument('--last_step', type=int, default=240,
                        help='the last decayed epoch of multistep scheduler')
    parser.add_argument('--step_gamma', type=float, default=0.25,
                        help='the gamma of multistep or step scheduler')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--similar_threshold', type=float, default=0.76)
    parser.add_argument('--margin', type=float, default=0.6)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--face_finetune', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/head_model_298_train2_0.871_test_0.956.pth',
                        help='finetune face backbone from checkpoint')
    parser.add_argument('--body_finetune', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/body_model_277_train1_0.872_test_0.958.pth',
                        help='finetune body backbone from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_freeze', default=True, type=bool)

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/xd/HUAWEI-CUP/DogFaceNet_Dataset_224_1/after_4_bis', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='PETFINDER', choices=['CIFAR', 'IMNET', 'IMNET_LMDB', 'image_folder', 'PETFINDER'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='result_fusion_TripletLoss/sgd_multistep_lr_0.001_margin_0.6_alpha_0.6_warm_5_dogfacenet_finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='result_fusion_TripletLoss/sgd_multistep_lr_0.001_margin_0.6_alpha_0.6_warm_5_dogfacenet_finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/result_fusion_TripletLoss/triplet_petfinder_all_step/best118_0.001_sgd_Multistep_triplet/2024-08-02 10:19:57/checkpoint-best.pth',
    # parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_envir', default=False, type=bool,
                        help='resume optimizer and schedule when load ckpt')
    parser.add_argument('--auto_resume', type=str2bool, default=False) # for resume training from result folder
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='temp', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, has_face=True, has_body=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, has_face=True, has_body=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=125,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.model == "mobilenet_v3_small":
        model = MobileNetV3_Small()
    elif args.model == "mobilenet_v3_large":
        model = MobileNetV3_Large()
    elif args.model == "facenet":
        num_classes = utils.get_num_classes(args.data_path + '/train.txt')
        model = Facenet(num_classes=num_classes)
    elif args.model == "fusion_mobilenet":
        num_classes = utils.get_num_classes(args.data_path + '/train.txt')
        model = Fusion_MobileNetV3(num_classes=num_classes, dropout_keep_prob=args.dropout_keep_prob, alpha=args.alpha)

    if args.face_finetune and args.body_finetune:
        if args.face_finetune.startswith('https'):
            face_checkpoint = torch.hub.load_state_dict_from_url(
                args.face_finetune, map_location='cpu', check_hash=True)
            face_checkpoint = face_checkpoint.state_dict()
        else:
            face_checkpoint = torch.load(args.face_finetune, map_location='cpu')
            # face_checkpoint = face_checkpoint.state_dict()
        print("Load face ckpt from %s" % args.face_finetune)

        # change face ckpt name
        for name, param in list(face_checkpoint.items()):
            if "features" in name:
                # new_name = name.replace("features", "face_backbone")
                new_name = name.replace("base_model.features", "face_backbone")
                face_checkpoint[new_name] = face_checkpoint.pop(name)
        # original_state_dict = model.state_dict()
        utils.load_state_dict(model, face_checkpoint, prefix=args.model_prefix)

        if args.body_finetune.startswith('https'):
            body_checkpoint = torch.hub.load_state_dict_from_url(
                args.body_finetune, map_location='cpu', check_hash=True)
            body_checkpoint = body_checkpoint.state_dict()
        else:
            body_checkpoint = torch.load(args.body_finetune, map_location='cpu')
            # body_checkpoint = body_checkpoint.state_dict()
        print("Load body ckpt from %s" % args.body_finetune)

        # change body ckpt name
        for name, param in list(body_checkpoint.items()):
            if "base_model.features" in name:
                new_name = name.replace("base_model.features", "body_backbone")
                body_checkpoint[new_name] = body_checkpoint.pop(name)
                # print(name)
        utils.load_state_dict(model, body_checkpoint, prefix=args.model_prefix)

        if args.model_freeze == True:
            checkpoint = OrderedDict(list(face_checkpoint.items()) + list(body_checkpoint.items()))
            model = utils.freeze_model(model, checkpoint)
            # print("*--------frozen params--------*")
            # for name, param in model.named_parameters():
            #     if param.requires_grad == False:
            #         print(name)
            print("*--------trainable params--------*")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
            # loaded_state_dict = model.state_dict()

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used # essentially contains loss.backward(create_graph=create_graph) and optimizer.step()

    # print("Use Cosine LR scheduler")
    # lr_schedule_values = utils.cosine_scheduler(
    #     args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    #     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    # )

    # print("Use Step LR scheduler")
    # lr_schedule_values = utils.step_scheduler(
    #     args.lr, args.epochs, num_training_steps_per_epoch, args.step_size, gamma=0.7,
    #     warmup_epochs=args.warmup_epochs,
    # )

    print("Use Multistep LR scheduler")
    lr_schedule_values = utils.multistep_scheduler(
        base_value=args.lr, epochs=args.epochs, niter_per_ep=num_training_steps_per_epoch,
        milestones=[args.start_step, args.middle_step, args.last_step], gamma=args.step_gamma,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # todo: add our own loss
    # criterion = ContrastiveLoss(batch_size=total_batch_size, device='cuda')
    criterion = TripletLoss(batch_size=total_batch_size, margin=args.margin)



    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, similar_threshold=args.similar_threshold)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, similar_threshold=args.similar_threshold, wandb_logger=wandb_logger)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%")
            if max_accuracy < test_stats["acc"]:
                max_accuracy = test_stats["acc"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc=test_stats['acc'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, similar_threshold=args.similar_threshold)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc"]:
                    max_accuracy_ema = test_stats_ema["acc"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc_ema=test_stats_ema['acc'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if args.model_ema and args.model_ema_eval:
        utils.bn_update(data_loader_train, model_ema.ema, use_amp=args.use_amp)
        test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, similar_threshold=args.similar_threshold)
        print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc']:.1f}%")
        if max_accuracy_ema < test_stats_ema["acc"]:
            max_accuracy_ema = test_stats_ema["acc"]
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #------------------sweep config------------------#
    # sweep_configuration = {
    #     "method": "random",
    #     "metric": {"goal": "maximize", "name": "acc"},
    #     "parameters": {
    #         "batch_size": {"values": [64, 128, 256]},
    #         "lr"        : {"max": 0.5, "min": 1e-6},
    #     },
    # }
    #
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
    # wandb.agent(sweep_id, function=main(args), count=10)

    main(args)

