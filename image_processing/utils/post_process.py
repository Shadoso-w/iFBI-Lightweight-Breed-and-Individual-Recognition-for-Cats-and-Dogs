from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ..utils.image import transform_preds



def get_pred_depth(depth):
    return depth


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)



def multi_pose_post_process(dets, c, s, h, w):
    # dets: batch x max_dets x 40
    # return list of 23 in image coord
    ret = []

    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        # pts = transform_preds(dets[i, :, 5:23].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:19].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5],
            #  pts.reshape(-1, 18)], axis=1).astype(np.float32).tolist()
            pts.reshape(-1, 14)], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})

    return ret
