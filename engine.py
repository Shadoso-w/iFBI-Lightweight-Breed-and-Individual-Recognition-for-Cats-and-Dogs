# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional

import numpy as np
import torch
from timm.data import Mixup
from timm.utils import ModelEma
from utils import accuracy, img_preprocess, classify_img_preprocess, get_class_name

import utils
import os

import torch.nn.functional as F
import torch.nn as nn

import time


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('tripletloss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('celoss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    sample_num = 0

    for data_iter_step, (samples, targets, face_samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        img_anc, img_pos, img_neg = samples
        lab_anc, lab_pos, lab_neg = targets
        fac_anc, fac_pos, fac_neg = face_samples

        img_anc = img_anc.to(device, non_blocking=True)
        img_pos = img_pos.to(device, non_blocking=True)
        img_neg = img_neg.to(device, non_blocking=True)

        fac_anc = fac_anc.to(device, non_blocking=True)
        fac_pos = fac_pos.to(device, non_blocking=True)
        fac_neg = fac_neg.to(device, non_blocking=True)

        lab_anc = lab_anc.to(device)
        lab_pos = lab_pos.to(device)
        lab_neg = lab_neg.to(device)

        # if mixup_fn is not None:
        #     samples_1, targets = mixup_fn(samples_1, targets)
        #     samples_2, targets = mixup_fn(samples_2, targets)

        if use_amp:
            with ((torch.cuda.amp.autocast())):
                feat_anc, pred_anc = model(fac_anc, img_anc)
                feat_pos, pred_pos = model(fac_pos, img_pos)
                feat_neg, pred_neg = model(fac_neg, img_neg)

                pred = torch.cat((pred_anc, pred_pos, pred_neg), dim=0)
                lab = torch.cat((lab_anc, lab_pos, lab_neg), dim=0)

                TripletLoss = criterion(pred_anc, pred_pos, pred_neg)
                CELoss = 0.1 * nn.NLLLoss()(F.log_softmax(pred, dim=-1), lab)
                loss = TripletLoss + CELoss
                # loss = TripletLoss
                print("INTO AMP")
        else: # full precision
            feat_anc, pred_anc = model(fac_anc, img_anc)
            feat_pos, pred_pos = model(fac_pos, img_pos)
            feat_neg, pred_neg = model(fac_neg, img_neg)

            pred = torch.cat((pred_anc, pred_pos, pred_neg), dim=0)
            lab = torch.cat((lab_anc, lab_pos, lab_neg), dim=0)

            TripletLoss = criterion(pred_anc, pred_pos, pred_neg)
            CELoss = 0.1 * nn.NLLLoss()(F.log_softmax(pred, dim=-1), lab)

            loss = TripletLoss + CELoss
            # loss = TripletLoss

            sample_num += 1
            # print("finished process sample number: ", sample_num)

        triplet_loss = TripletLoss.item()
        ce_loss = CELoss.item()
        total_loss = loss.item()

        if not math.isfinite(total_loss): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(total_loss))
            assert math.isfinite(total_loss)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        # torch.cuda.synchronize()

        # if mixup_fn is None:
        #     class_acc = (output.max(-1)[-1] == targets).float().mean()
        # else:
        #     class_acc = None
        metric_logger.update(loss=total_loss)
        metric_logger.update(tripletloss=triplet_loss)
        metric_logger.update(celoss=ce_loss)
        # metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=total_loss, head="total_loss")
            log_writer.update(loss=triplet_loss, head="triplet_loss")
            log_writer.update(loss=ce_loss, head="ce_loss")
            # log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': total_loss,
                'Rank-0 Batch Wise/triplet_loss': triplet_loss,
                'Rank-0 Batch Wise/ce_loss': ce_loss,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            # if class_acc:
            #     wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, similar_threshold=0.8, wandb_logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # start_time = time.time()
    for batch in metric_logger.log_every(data_loader, 1, header):
        images_1, images_2 = batch[0]
        face_1, face_2 = batch[1]
        target = batch[-1]

        images_1 = images_1.to(device, non_blocking=True)
        images_2 = images_2.to(device, non_blocking=True)
        face_1 = face_1.to(device, non_blocking=True)
        face_2 = face_2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                feature_1 = model(face_1, images_1)
                feature_2 = model(face_2, images_2)
                loss = criterion(feature_1, feature_2)
                similairty = torch.cosine_similarity(feature_1, feature_2, dim=1, eps=1e-8)
                similairty[similairty >= similar_threshold] = 1
                similairty[similairty < similar_threshold] = 0
        else:
            feature_1 = model(face_1, images_1)
            feature_2 = model(face_2, images_2)
            loss = criterion(feature_1, feature_2)
            similairty = torch.cosine_similarity(feature_1, feature_2, dim=1, eps=1e-8)
            similairty[similairty >= similar_threshold] = 1
            similairty[similairty < similar_threshold] = 0

        output = similairty
        # print("output: ", output)
        # print("target: ", target)
        acc = accuracy(output, target)

        batch_size = images_1.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc, n=batch_size)

        if wandb_logger:
            wandb_logger._wandb.log({
                'acc': acc
            }, commit=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc, losses=metric_logger.loss))

    # finish_time = time.time()
    # time_cost = finish_time-start_time
    # print(f"Time Cost: {time_cost} s")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def extract_feature(dataloader, model, device, args=None, wandb_logger=None):
    # switch to evaluation mode
    model.eval()

    for (image, face_image, label) in dataloader:

        image.to(device)
        face_image.to(device)
        label.to(device)

        feature = model(face_image, image)

        # write to .txt
        file_path = args.output_dir + '/feature.txt'
        f = open(file_path, 'a')
        for i in range(0, len(label)):
            feature_list = feature[i].tolist() # avoid auto enter
            f.write("{label}+{feature}\n".format(label=label[i], feature=feature_list))
        print("{batch_size} features saved".format(batch_size=len(label)))

    print("All features saving finished")

@torch.no_grad()
def pair_inference(model, device, args, pair_img_dir, similar_threshold):
    # get images absolute path
    pair_img_path = os.listdir(pair_img_dir)
    assert len(pair_img_path) == 2
    pair_img_path[0] = pair_img_dir + '/' + pair_img_path[0]
    pair_img_path[1] = pair_img_dir + '/' + pair_img_path[1]

    # get images
    image_1, face_image_1 = img_preprocess(pair_img_path[0], args)
    image_2, face_image_2 = img_preprocess(pair_img_path[1], args)

    # switch to evaluation mode
    model.eval()

    # image_1.to(device)
    # face_image_1.to(device)
    # image_2.to(device)
    # face_image_2.to(device)

    image_1 = image_1.unsqueeze(0).to(device) # unsqueeze dimension to fit model input size (add batch_size dimension)
    face_image_1 = face_image_1.unsqueeze(0).to(device)
    image_2 = image_2.unsqueeze(0).to(device)
    face_image_2 = face_image_2.unsqueeze(0).to(device)

    feature_1 = model(face_image_1, image_1)
    feature_2 = model(face_image_2, image_2)

    similairty = torch.cosine_similarity(feature_1, feature_2, dim=1, eps=1e-8)
    if similairty >= similar_threshold:
        return True
    else:
        return False

@torch.no_grad()
def single_inference(face_model, body_model, device, args, single_img_path, alpha=0.4):
    # get image
    face_image, image = classify_img_preprocess(single_img_path, args)

    # switch to evaluation mode
    face_model.eval()
    body_model.eval()

    image = image.unsqueeze(0).to(device)  # unsqueeze dimension to fit model input size (add batch_size dimension)
    face_image = face_image.unsqueeze(0).to(device)

    face_pred = face_model(face_image)
    body_pred = body_model(image)

    avg_pred = alpha * face_pred + (1 - alpha) * body_pred
    _, final_pred = torch.max(avg_pred.data, 1)

    final_pred = final_pred.item() # tensor to int
    result = get_class_name(final_pred)

    return result

@torch.no_grad()
def classify_evaluate(data_loader, face_model, body_model, device, args, classify_data_path, alpha=0.4):
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    face_model.eval()
    body_model.eval()

    for (image, face_image, label) in metric_logger.log_every(data_loader, 10, header):
        face_image = face_image.to(device)
        image = image.to(device)
        label = label.to(device)

        face_pred = face_model(face_image)
        body_pred = body_model(image)
        avg_pred = alpha * face_pred + (1 - alpha) * body_pred # weighted average

        loss = criterion(avg_pred, label) # CE

        _, final_pred = torch.max(avg_pred.data, 1)
        acc = accuracy(final_pred, label)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    pass




