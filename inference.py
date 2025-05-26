import torch
import argparse
import utils

# from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
# from model.facenet import Facenet
from model.fusion_mobilenet_temp import Fusion_MobileNetV3_inference
from model.mobiledynamic import DynamicMobileNetV3Large

# from dataset.single_dataset import build_dataset # for feature extract
# from dataset.triplet_dataset import build_dataset
from dataset.iFBI_dataset import build_dataset
from engine import evaluate, extract_feature, pair_inference, single_inference, classify_evaluate

from pathlib import Path

import sys
sys.path.insert(0, './model')

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
        raise argparse.ArgumentTypeError('Boolean value expected')

def get_args_parser():
    #------------------------------------ global config ------------------------------------#
    parser = argparse.ArgumentParser('Inference for The Dog/Cat Classification and Identification', add_help=False)
    parser.add_argument('--infer_mode', default='classify', choices=['classify', 'identify'], type=str,
                        help='Choose from classify/identify')

    # overall config
    parser.add_argument('--device', default='cpu', type=str,
                        help='Choose from cpu/cuda')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Inferring batch size')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='')
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--seed', default=81, type=int)
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--model_prefix', default='', type=str)

    # data config
    parser.add_argument('--data_set', default='OXFORD_PET',
                        choices=['OXFORD_PET', 'DOGFACENET', 'THU_DOGS', 'STANFORD_DOGS', 'PETFINDER'], type=str,
                        help='Dataset type')
    parser.add_argument('--data_path', default='/home/xd/HUAWEI-CUP/Oxford Pet', type=str,
                        help='Image dataset path')
    parser.add_argument('--pair_img_dir', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/demo/same_cat_without_face', type=str,
                        help='Image folder path for indentify inference')
    parser.add_argument('--single_img_path', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/demo/american_bulldog.jpg', type=str,
                        help='Image path for classify inference')

    #--------------------------------- classification config ---------------------------------#
    # mode config
    parser.add_argument('--classify_mode', default='single', choices=['single', 'dataset'], type=str,
                        help='Classify for single pic or dataset')

    # model config
    parser.add_argument('--classify_model', default='dynamic_mobilenet', type=str, metavar='MODEL',
                        help='Name of classify model to infer')
    parser.add_argument('--face_ckpt', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/head_model_298_train2_0.871_test_0.956.pth', type=str,
                        help='Face model inference checkpoint path')
    parser.add_argument('--body_ckpt', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/body_model_277_train1_0.872_test_0.958.pth', type=str,
                        help='Body model inference checkpoint path')

    #--------------------------------- identification config ---------------------------------#
    # mode config
    parser.add_argument('--identify_mode', default='pair', choices=['pair', 'dataset'], type=str,
                        help='Identify for pair pic or dataset')

    # model config
    parser.add_argument('--identify_model', default='fusion_mobilenet', type=str, metavar='MODEL',
                        help='Name of identify model to infer')
    parser.add_argument('--model_checkpoint', default='pretrained/checkpoint_90.96.pth',
                        help='Model inference checkpoint path')
    parser.add_argument('--identity_ckpt', default='/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/identity_layer_91.54.pth', type=str,
                        help='Identity layer inference checkpoint path')
    parser.add_argument('--similar_threshold', default=0.76, type=float,
                        help="Used only for cosine similarity")

    return parser.parse_args()

def build_dataloader(args):
    # dataloader
    if args.data_path is not None:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        dataset, _ = build_dataset(is_train=False, has_face=True, has_body=True, args=args)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    return dataloader, dataset

def infer_classify_dataset(device, dataloader, dataset, args):
    # classify model
    if args.classify_model == "dynamic_mobilenet":
        dynamic_layers = [5, 6, 8, 10, 14]
        face_model = DynamicMobileNetV3Large(num_classes=37, dynamic_layers=dynamic_layers, K=4, temperature=30)
        body_model = DynamicMobileNetV3Large(num_classes=37, dynamic_layers=dynamic_layers, K=4, temperature=30)

    # read checkpoints
    if args.face_ckpt and args.body_ckpt:
        face_checkpoint = torch.load(args.face_ckpt, map_location='cpu')
        # face_checkpoint = face_checkpoint.state_dict()
        body_checkpoint = torch.load(args.body_ckpt, map_location='cpu')
        # body_checkpoint = body_checkpoint.state_dict()

        utils.load_state_dict(face_model, face_checkpoint, prefix=args.model_prefix)
        utils.load_state_dict(body_model, body_checkpoint, prefix=args.model_prefix)

        print(f"Load face ckpt from {args.face_ckpt}\n"
              f"Load body ckpt from {args.body_ckpt}")

    face_model.to(device)
    body_model.to(device)

    # inference for evaluate
    test_stats = classify_evaluate(dataloader, face_model, body_model, device, args, args.data_path)
    print(f"Accuracy of the network on {len(dataset)} test images: {test_stats['acc']:.5f}%")
    return

def infer_classify_single(device, args):
    # classify model
    if args.classify_model == "dynamic_mobilenet":
        dynamic_layers = [5, 6, 8, 10, 14]
        face_model = DynamicMobileNetV3Large(num_classes=37, dynamic_layers=dynamic_layers, K=4, temperature=30)
        body_model = DynamicMobileNetV3Large(num_classes=37, dynamic_layers=dynamic_layers, K=4, temperature=30)

    # read checkpoints
    if args.face_ckpt and args.body_ckpt:
        face_checkpoint = torch.load(args.face_ckpt, map_location='cpu')
        # face_checkpoint = face_checkpoint.state_dict()
        body_checkpoint = torch.load(args.body_ckpt, map_location='cpu')
        # body_checkpoint = body_checkpoint.state_dict()

        utils.load_state_dict(face_model, face_checkpoint, prefix=args.model_prefix)
        utils.load_state_dict(body_model, body_checkpoint, prefix=args.model_prefix)

        print(f"Load face ckpt from {args.face_ckpt}\n"
              f"Load body ckpt from {args.body_ckpt}")

    face_model.to(device)
    body_model.to(device)

    # single image inference
    classify_result = single_inference(face_model, body_model, device, args, args.single_img_path)
    print(f"Classify Result: {classify_result}")
    return

def infer_identify_dataset(device, dataloader, dataset, args):
    # feature extract model
    if args.identify_model == "mobilenet_v3_small":
        model = MobileNetV3_Small()
    elif args.identify_model == "mobilenet_v3_large":
        model = MobileNetV3_Large()
    elif args.identify_model == "facenet":
        model = Facenet()
    elif args.identify_model == "fusion_mobilenet":
        model = Fusion_MobileNetV3_inference()

    # read checkpoints from 3 pth file
    if args.face_ckpt and args.body_ckpt:
        # load backbone
        face_checkpoint = torch.load(args.face_ckpt, map_location='cpu')
        # face_checkpoint = face_checkpoint.state_dict()
        body_checkpoint = torch.load(args.body_ckpt, map_location='cpu')
        # body_checkpoint = body_checkpoint.state_dict()

        for name, param in list(face_checkpoint.items()):
            if "base_model.features" in name:
                # new_name = name.replace("features", "face_backbone")
                new_name = name.replace("base_model.features", "face_backbone")
                face_checkpoint[new_name] = face_checkpoint.pop(name)
        utils.load_state_dict(model, face_checkpoint, prefix=args.model_prefix)

        for name, param in list(body_checkpoint.items()):
            if "base_model.features" in name:
                new_name = name.replace("base_model.features", "body_backbone")
                body_checkpoint[new_name] = body_checkpoint.pop(name)
        utils.load_state_dict(model, body_checkpoint, prefix=args.model_prefix)

        print(f"Load face ckpt from {args.face_ckpt}\n"
              f"Load body ckpt from {args.body_ckpt}")

    if args.identity_ckpt:
        # load identity layer
        identity_checkpoint = torch.load(args.identity_ckpt, map_location='cpu')
        # identity_checkpoint = identity_checkpoint.state_dict()

        utils.load_state_dict(model, identity_checkpoint, prefix=args.model_prefix)
        print(f"Load identity layer ckpt from {args.identity_ckpt}")

    model.to(device)

    # inference for evaluate
    test_stats = evaluate(dataloader, model, device, similar_threshold=args.similar_threshold)
    print(f"Accuracy of the network on {len(dataset)} test images: {test_stats['acc']:.5f}%")
    return

def infer_identify_pair(device, args):
    # feature extract model
    if args.identify_model == "mobilenet_v3_small":
        model = MobileNetV3_Small()
    elif args.identify_model == "mobilenet_v3_large":
        model = MobileNetV3_Large()
    elif args.identify_model == "facenet":
        model = Facenet()
    elif args.identify_model == "fusion_mobilenet":
        model = Fusion_MobileNetV3_inference()

    # read checkpoints from 3 pth file
    if args.face_ckpt and args.body_ckpt:
        # load backbone
        face_checkpoint = torch.load(args.face_ckpt, map_location='cpu')
        # face_checkpoint = face_checkpoint.state_dict()
        body_checkpoint = torch.load(args.body_ckpt, map_location='cpu')
        # body_checkpoint = body_checkpoint.state_dict()

        for name, param in list(face_checkpoint.items()):
            if "base_model.features" in name:
                # new_name = name.replace("features", "face_backbone")
                new_name = name.replace("base_model.features", "face_backbone")
                face_checkpoint[new_name] = face_checkpoint.pop(name)
        utils.load_state_dict(model, face_checkpoint, prefix=args.model_prefix)

        for name, param in list(body_checkpoint.items()):
            if "base_model.features" in name:
                new_name = name.replace("base_model.features", "body_backbone")
                body_checkpoint[new_name] = body_checkpoint.pop(name)
        utils.load_state_dict(model, body_checkpoint, prefix=args.model_prefix)

        print(f"Load face ckpt from {args.face_ckpt}\n"
              f"Load body ckpt from {args.body_ckpt}")

    if args.identity_ckpt:
        # load identity layer
        identity_checkpoint = torch.load(args.identity_ckpt, map_location='cpu')
        # identity_checkpoint = identity_checkpoint.state_dict()

        utils.load_state_dict(model, identity_checkpoint, prefix=args.model_prefix)
        print(f"Load identity layer ckpt from {args.identity_ckpt}")

    model.to(device)

    # pair images inference
    match_result = pair_inference(model, device, args, pair_img_dir=args.pair_img_dir,
                                  similar_threshold=args.similar_threshold)
    if match_result:
        print("Same Pet")
    else:
        print("Different Pet")
    return

def main(args):

    device = torch.device(args.device)

    if args.infer_mode == 'classify':
        if args.classify_mode == 'dataset':
            dataloader, dataset = build_dataloader(args)
            infer_classify_dataset(device, dataloader, dataset, args)
        elif args.classify_mode == 'single':
            infer_classify_single(device, args)
        else:
            print('Please choose classify mode from single/dataset')

    elif args.infer_mode == 'identify':
        if args.identify_mode == 'dataset':
            dataloader, dataset = build_dataloader(args)
            infer_identify_dataset(device, dataloader, dataset, args)
        elif args.identify_mode == 'pair':
            infer_identify_pair(device, args)
        else:
            print('Please choose indentify mode from pair/dataset')

    else:
        print('Please choose infer mode from classify/identify')

def infer_all(infer_mode, imgs):
    args = get_args_parser()
    args.infer_mode = infer_mode

    main(args)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)
