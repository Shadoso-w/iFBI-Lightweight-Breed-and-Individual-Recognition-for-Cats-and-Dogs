from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import os

from .utils.pose_dla_dcn import get_pose_net as get_dla_dcn
from .utils.decode import multi_pose_decode
from .utils.image import get_affine_transform
from .utils.post_process import multi_pose_post_process

image_ext = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

def _to_float(x):
    return float("{:.2f}".format(x))

def convert_eval_format(all_bboxes):
    # import pdb; pdb.set_trace()
    csv_results = []
    tag = -1
    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            for dets in all_bboxes[image_id][cls_ind]:
                keypoints = np.concatenate([
                    # np.array(dets[5:23], dtype=np.float32).reshape(-1, 2),
                    np.array(dets[5:19], dtype=np.float32).reshape(-1, 2),
                    # np.ones((9, 1), dtype=np.float32)], axis=1).reshape(27).tolist()
                    np.ones((7, 1), dtype=np.float32)], axis=1).reshape(21).tolist()
                keypoints = list(map(_to_float, keypoints))
                # keypoints = np.array(keypoints).reshape([9, 3]).astype(np.float32)
                keypoints = np.array(keypoints).reshape([7, 3]).astype(np.float32)
                # keypoints = np.delete(keypoints, -1, axis=1).reshape([1, 18]).tolist()
                keypoints = np.delete(keypoints, -1, axis=1).reshape([1, 14]).tolist()
                # csv_result = [int(image_id)] + keypoints[0]
                csv_result = [image_id] + keypoints[0]
                if tag != csv_result[0]:
                    csv_results.append(csv_result)

                tag = csv_result[0]

    return csv_results

def run(image_or_path_or_tensor, load_model_path, meta=None):
    # if opt.gpus[0] >= 0:
    #   device = torch.device('cuda')
    # else:
    #   device = torch.device('cpu') !!!!!!!!!!!!!!!!!!!!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')   
    # print(device)
    # print('Creating model...')

    # model = get_dla_dcn(num_layers=34, heads={'hm': 1, 'wh': 2, 'hps': 18, 'reg': 2, 'hm_hp': 9, 'hp_offset': 2}, head_conv=256)
    model = get_dla_dcn(num_layers=34, heads={'hm': 1, 'wh': 2, 'hps': 14, 'reg': 2, 'hm_hp': 7, 'hp_offset': 2}, head_conv=256)
    model = load_model(model, load_model_path)
    model = model.to(device)
    model.eval()

    image = cv2.imread(image_or_path_or_tensor)
    
    detections = []

    images, meta = pre_process(image, meta)
    images = images.to(device) # torch.Size([1, 3, 512, 512])
    torch.cuda.synchronize()
 
    output, dets = process(model, images)

    torch.cuda.synchronize()
      
    dets = post_process(dets, meta)
    torch.cuda.synchronize()

    detections.append(dets)
    
    results = merge_outputs(detections)
    torch.cuda.synchronize()
    
    return {'results': results}

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters   
    model.load_state_dict(state_dict, strict=False)
    return model

def pre_process(image, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height)
    new_width  = int(width)

    inp_height, inp_width = 512, 512
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32) # 中心点
    s = max(height, width) * 1.0 # 缩放因子
   
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height]) # 仿射变换需要的变换矩阵
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR) # 仿射变换
    new_mean = [0.408, 0.447, 0.470]
    new_std = [0.289, 0.274, 0.278]
    inp_image = ((inp_image / 255. - new_mean) / new_std).astype(np.float32) # 归一化加标准化

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // 4, 
            'out_width': inp_width // 4}
    
    return images, meta


def process(model, images):
    with torch.no_grad():
        torch.cuda.synchronize()
        output = model(images)[-1]  # 在这里讲图片送入模型,并得到返回结果  image torch.Size([1, 3, 512, 512])
        output['hm'] = output['hm'].sigmoid_()     # torch.Size([1, 1, 128, 128])
        
        output['hm_hp'] = output['hm_hp'].sigmoid_()    # torch.Size([1, 9, 128, 128])

        reg = output['reg']
        hm_hp = output['hm_hp']
        hp_offset = output['hp_offset']
        torch.cuda.synchronize()

        dets = multi_pose_decode(
            output['hm'], output['wh'], output['hps'],
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=100)

    return output, dets


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])    # [1, 100, 24]

    dets = multi_pose_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])

    for j in range(1, 2):
        # dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 23)
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 19)
        dets[0][j][:, :4] /= 1
        dets[0][j][:, 5:] /= 1
    return dets[0]

def merge_outputs(detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results

def landmark_test(image_path):

    results = {}
    
    image_names = [image_path]

    for ind, (image_name) in enumerate(image_names):
        # img_id = image_name.split("/")[-1].split(".")[0]
        if "/" in image_name:
            img_id = image_name.split("/")[-1].split(".")[0]
        else:
            img_id = image_name.split(".")[0]
        img_path = image_name
        print("image path:", img_path)
        load_model_path = os.path.dirname(os.path.abspath(__file__)) + '/model/model_best_catdog.pth'
        # ret = detector.run(img_path)
        ret = run(img_path, load_model_path)
        results[img_id] = ret['results']
        csv_results = convert_eval_format(results)
        result = csv_results[0][1: ]

        indexes_to_remove = {6, 7, 12, 13}
        key_points_5 = [item for idx, item in enumerate(result) if idx not in indexes_to_remove]

        return key_points_5

if __name__ == '__main__':
    image_path = 'data/0.gif'
    key_points = landmark_test(image_path)
    print(key_points)