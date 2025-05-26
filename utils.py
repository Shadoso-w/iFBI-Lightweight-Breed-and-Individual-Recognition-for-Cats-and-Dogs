# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
import time

import matplotlib.pyplot as plt
import tqdm
import numpy
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
from torch._six import inf
from image_processing.yolo_test import detect_coordinate_yolo
from image_processing.landmark_test import landmark_test
from torch.autograd import Variable
# from dataset.single_dataset import build_dataset

from PIL import Image

import cv2

CLASS_NAME_LIST = ['American Bulldog',
                   'American Pit Bull Terrier',
                   'Basset Hound',
                   'Beagle',
                   'Boxer',
                   'Chihuahua',
                   'English Cocker Spaniel',
                   'English Setter',
                   'German Shorthaired',
                   'Great Pyrenees',
                   'Havanese',
                   'Japanese Chin',
                   'Keeshond',
                   'Leonberger',
                   'Miniature Pinscher',
                   'Newfoundland',
                   'Pomeranian',
                   'Pug',
                   'Saint Bernard',
                   'Samyoed',
                   'Scottish Terrier',
                   'Shiba Inu',
                   'Staffordshire Bull Terrier',
                   'Wheaten Terrier',
                   'Yorkshire Terrier',
                   'Abyssinian',
                   'Bengal',
                   'Birman',
                   'Bombay',
                   'British Shorthair',
                   'Egyptian Mau',
                   'Main Coon',
                   'Persian',
                   'Ragdoll',
                   'Russian Blue',
                   'Siamese',
                   'Sphynx']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    # if len(missing_keys) > 0:
    #     print("Weights of {} not initialized from pretrained model: {}".format(
    #         model.__class__.__name__, missing_keys))
    # if len(unexpected_keys) > 0:
    #     print("Weights from pretrained model not used in {}: {}".format(
    #         model.__class__.__name__, unexpected_keys))
    # if len(ignore_missing_keys) > 0:
    #     print("Ignored weights of {} not initialized from pretrained model: {}".format(
    #         model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cyclic_scheduler(max_value, min_value, epochs, niter_per_ep, step_size, mode='exp_range', gamma=0.99994):
    if mode == 'triangular':
        scale_fn = lambda x: 1.  # triangular
    elif mode == 'triangular2':
        scale_fn = lambda x: 1 / (2. ** (x - 1))  # triangular2
    elif mode == 'exp_range':
        scale_fn = lambda x: gamma ** (x)  # exp_range

    iters = np.arange(epochs * niter_per_ep)
    cycle = np.array([np.floor(1 + i / (2 * step_size)) for i in iters])
    x = np.array([np.abs(i / step_size - 2 * cycle[i] + 1) for i in iters])
    schedule = np.array(min_value + (max_value - min_value) * np.maximum(0, (1 - x)) * scale_fn(x))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def multistep_scheduler(base_value, epochs, niter_per_ep, milestones, gamma=0.2, warmup_epochs=0,
                        start_warmup_value=0, warmup_steps=-1):
    # warmup阶段
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    # lr scheduler
    assert (epochs * niter_per_ep - warmup_iters) > milestones[-1]
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    milestones.insert(0, 0)
    lr_every_step = [base_value * np.power(gamma, i) for i in range(0, len(milestones))]
    schedule = []
    step_record = 0
    print(lr_every_step)
    for i in iters:
        if i/niter_per_ep in milestones:
            step_record = milestones.index(i/niter_per_ep)
        schedule.append(lr_every_step[step_record])
    schedule = np.array(schedule)
    schedule = np.concatenate((warmup_schedule, schedule))

    print(schedule)
    # import matplotlib.pyplot as plt
    # plt.plot(range(len(schedule)), schedule, label='step_lr')
    # plt.show()

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def step_scheduler(base_value, epochs, niter_per_ep, step_size, gamma=0.2, warmup_epochs=0,
                        start_warmup_value=0, warmup_steps=-1):
    # warmup阶段
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    # lr scheduler
    assert (epochs * niter_per_ep - warmup_iters) > step_size
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    lr = base_value
    schedule = []
    for i in iters:
        if (i/niter_per_ep) % step_size == 0 and i != 0:
            lr = lr * gamma
        schedule.append(lr)
    schedule = np.array(schedule)
    schedule = np.concatenate((warmup_schedule, schedule))

    print(schedule)
    import matplotlib.pyplot as plt
    plt.plot(range(len(schedule)), schedule, label='step_lr')
    plt.show()

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, classifier=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    model_checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    classifier_checkpoint_paths = [output_dir / ('classifier_checkpoint-%s.pth' % epoch_name)]

    # for feature extract model
    for checkpoint_path in model_checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

    if classifier is not None:
        # for classifier
        for checkpoint_path in classifier_checkpoint_paths:
            to_save = {
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)

        if is_main_process() and isinstance(epoch, int):
            to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
            old_ckpt = output_dir / ('classifier_checkpoint-%s.pth' % to_del)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model'])
        load_state_dict(model_without_ddp, checkpoint['model'], prefix=args.model_prefix)
        print("Resume checkpoint %s" % args.resume)
        if args.resume_envir:
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if not isinstance(checkpoint['epoch'], str):  # does not support resuming with 'best', 'best-ema'
                    args.start_epoch = checkpoint['epoch'] + 1
                else:
                    assert args.eval, 'Does not support resuming with checkpoint-best'
                if hasattr(args, 'model_ema') and args.model_ema:
                    if 'model_ema' in checkpoint.keys():
                        model_ema.ema.load_state_dict(checkpoint['model_ema'])
                    else:
                        model_ema.ema.load_state_dict(checkpoint['model'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, use_amp=False):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        if use_amp:
            with torch.cuda.amp.autocast():
                model(input)
        else:
            model(input)

        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def accuracy(output, target):
    assert output.shape == target.shape

    correct = (output == target).sum().item()
    total = output.numel()
    # print(f"correct / total: {correct} / {total} \n")
    acc = (correct / total) * 100

    return acc


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    num_classes = len(dataset_path)

    return num_classes


def freeze_model(model, to_freeze_dict):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    # # show freeze statement
    # freezed_num, pass_num = 0, 0
    # for (name, param) in model.named_parameters():
    #     if param.requires_grad == False:
    #         freezed_num += 1
    #     else:
    #         pass_num += 1
    # print('\n Total {} params, miss {} \n'.format(freezed_num + pass_num, pass_num))

    return model


# 裁切面部数据
def crop_face(data_path, input_features, labels):
    coordinates = []
    with open(data_path + '/coordinates.txt', 'r') as f:
        for line in f.readlines():
            coordinates.append(line.split(' '))
    print(coordinates)


def read_img(img_path):
    image = cv2.imread(img_path)
    return image


def one_align(image, bbox, key_points):
    bbox = np.array(bbox)
    # 特殊判断，判断是否有身体数据
    none_body = np.array([-1, -1, -1, -1])
    flag = 0

    if (bbox == none_body).all():
        # 如果没有身体数据，坐标设置为图像长宽
        height, width, _ = image.shape
        # 设置标记信息
        flag = 1
    else:
        width = int(bbox[2] - bbox[0])
        height = int(bbox[3] - bbox[1])

    marks = np.array(key_points).reshape([-1, 2])

    # src = marks[[0, 1, 2, 5]]

    key_marks = np.array([int(width / 4), int(height / 5 * 3),
                          int(width / 4 * 3), int(height / 5 * 3),
                          int(width / 2), height - 5,
                          int(width / 4), int(height / 4),
                          int(width / 4 * 3), int(height / 4)
                          ]).reshape([-1, 2])

    # dst = key_marks[[0, 1, 2, 5]]

    # 获取变换矩阵
    M, _ = cv2.findHomography(marks, key_marks)
    # 执行变换操作
    transformed = cv2.warpPerspective(image, M, (width, height), borderValue=0.0)

    if flag:
        image = torch.zeros((3, 224, 224))

    return image, transformed


def transform(img, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)(img)


def classify_transform(img, coordinate):
    if coordinate is not None and any(coordinate):
        xmin, ymin, xmax, ymax = coordinate
        face_img = img.crop((xmin, ymin, xmax, ymax))

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_img = transform_test(face_img)
    img = transform_test(img)

    return img, face_img


def img_preprocess(img_path, args):
    # get data
    img = read_img(img_path)

    coordinate = detect_coordinate_yolo(img)
    assert len(coordinate) == 4
    coordinate = list(map(float, coordinate))

    landmark = landmark_test(img_path)
    assert len(landmark) == 10

    # data test preprocess
    none_face = [-1, -1, -1, -1]  # None face standard
    none_tensor = torch.zeros((3, 224, 224))

    if coordinate == none_face:  #
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img, args)
        return img, none_tensor

    img, align_img = one_align(img, coordinate, landmark)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # ndarray to PIL Image for transform
    align_img = Image.fromarray(cv2.cvtColor(align_img, cv2.COLOR_BGR2RGB))
    img = transform(img, args)
    align_img = transform(align_img, args)
    return img, align_img


def classify_img_preprocess(img_path, args):
    # get datas
    img = read_img(img_path)

    coordinate = detect_coordinate_yolo(img)
    assert len(coordinate) == 4
    coordinate = list(map(float, coordinate))

    # data test preprocess
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # ndarray to PIL Image for transform
    img, face_img = classify_transform(img, coordinate)

    return img, face_img


def get_class_name(num):
    return CLASS_NAME_LIST[num]


def get_centroid(model, device, dataloader, args):
    # pre-compute centroid
    model.eval()

    centroidsOri = {}
    centroidsNorm = {}
    centroidsCount = {}

    # # dataloader
    # if args.data_path is not None:
    #     num_tasks = get_world_size()
    #     global_rank = get_rank()
    #
    #     dataset, _ = build_dataset(is_train=True, has_face=True, has_body=True, args=args)
    #     sampler = torch.utils.data.DistributedSampler(
    #         dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, sampler=sampler,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )

    for data_iter_step, (img_anc, lab_anc, fac_anc) in enumerate(dataloader):  # here is train_dataloader

        # print("lab_anc: ", lab_anc)

        img_anc = img_anc.to(device, non_blocking=True)
        fac_anc = fac_anc.to(device, non_blocking=True)

        with torch.no_grad():
            feat_anc = model(fac_anc, img_anc)
            feat_anc_norm = feat_anc.div(torch.norm(feat_anc, p=2, dim=1, keepdim=True).expand_as(feat_anc))

        feat_anc = feat_anc.cpu().numpy()
        feat_anc_norm = feat_anc_norm.cpu().numpy()
        lab_anc = lab_anc.cpu().numpy()

        # print("feat_anc.device: ", feat_anc.device)
        # print("feat_anc_norm.device: ", feat_anc_norm.device)

        for n, k in enumerate(lab_anc):
            if k in centroidsCount.keys():
                centroidsOri[k] = centroidsOri[k] + feat_anc[n]
                centroidsNorm[k] = centroidsNorm[k] + feat_anc_norm[n]
                centroidsCount[k] = centroidsCount[k] + 1
                # print(f"len(centroidsCount): {len(centroidsCount)}")
                # print(f"Used Memory: {torch.cuda.memory_allocated()}")
                # print(f"Reserved Memory: {torch.cuda.memory_reserved()}")
            else:
                centroidsOri[k] = feat_anc[n]
                centroidsNorm[k] = feat_anc_norm[n]
                centroidsCount[k] = 1
                # print(f"len(centroidsCount): {len(centroidsCount)}")
                # print(f"Used Memory: {torch.cuda.memory_allocated()}")
                # print(f"Reserved Memory: {torch.cuda.memory_reserved()}")
        # torch.cuda.empty_cache() # free the memory of cuda every iteration

    # for k, v in centroidsCount.items():
    #     if v >= 2:
    #         print(k)

    # mean of original feature
    centroidsOri = {k: torch.tensor(centroidsOri[k] / centroidsCount[k]) for k in centroidsOri}
    # mean of normalized feature
    centroidsNorm = {k: torch.tensor(centroidsNorm[k] / centroidsCount[k]) for k in centroidsNorm}
    # normalized mean of original feature
    # centroidsOriNorm = {k: centroidsOri[k].div(torch.norm(centroidsOri[k])) for k in centroidsOri}
    # normalized mean of normalized feature
    centroidsNormNorm = {k: torch.tensor(centroidsNorm[k] / (numpy.linalg.norm(centroidsNorm[k]))) for k in
                         centroidsNorm}

    print("pre-compute centroid done!\n"
          f"{len(centroidsCount)} classes has been saved!")

    # test
    f = open("centroidsCount.txt", "w")
    for k, v in centroidsCount.items():
        f.write("{key} {value} \n".format(key=k, value=v))  # write into .txt

    return centroidsOri, centroidsNormNorm  # all is numpy type not tensor type


def get_pos_set(centroidsNorm, centroids, labels, device):
    # print("INTO get_pos_set")
    labels = labels.cpu().numpy()  # todo: need to be int or float
    # posNorm = Variable(torch.stack([centroidsNorm.get(k, None) for k in labels]), requires_grad=False)
    posNorm = Variable(torch.stack([centroidsNorm[k] for k in labels]), requires_grad=False)
    pos = Variable(torch.stack([centroids[k] for k in labels]), requires_grad=False)

    posNorm = posNorm.to(device)
    pos = pos.to(device)
    return posNorm, pos


def get_neg_set(anchorNorm, anchor, labels):
    minLen = 5
    if len(anchorNorm) == len(anchor):
        N = len(anchor)
    batchDistanceNorm = [F.pairwise_distance(anchorNorm[k].expand(N, -1), anchorNorm) for k in range(N)]
    batchDistance = [F.pairwise_distance(anchor[k].expand(N, -1), anchor) for k in range(N)]

    sortedKeyCtrdDNorm = {k: torch.sort(batchDistanceNorm[k].cpu(), dim=0)[1] for k in range(N)}
    sortedKeyCtrdD = {k: torch.sort(batchDistance[k].cpu(), dim=0)[1] for k in range(N)}

    # remove hard sample with same label in sortedKey
    sortedKeyCtrdDNorm = {k: [int(n) for n in sortedKeyCtrdDNorm[k] if labels[int(n)] != labels[k]] for k in range(N)}
    sortedKeyCtrdD = {k: [int(n) for n in sortedKeyCtrdD[k] if labels[int(n)] != labels[k]] for k in range(N)}

    minLen = min(min([len(sortedKeyCtrdDNorm[k]) for k in sortedKeyCtrdDNorm]),
                 min([len(sortedKeyCtrdD[k]) for k in sortedKeyCtrdD]), minLen)
    negFtNorm = [Variable(torch.stack([anchorNorm[sortedKeyCtrdDNorm[n][k]].data for n in range(N)]),
                          requires_grad=False) for k in range(minLen)]
    negFt = [Variable(torch.stack([anchor[sortedKeyCtrdD[n][k]].data for n in range(N)]),
                      requires_grad=False) for k in range(minLen)]
    return negFtNorm, negFt


if __name__ == '__main__':
    # classify_img_preprocess("/home/xd/HUAWEI-CUP/petfinder_extra_cats(processed)/55708276/0.png", args=None)
    step_scheduler(base_value=0.01, epochs=300, niter_per_ep=140, step_size=20, gamma=0.7,
                        warmup_epochs=10)
