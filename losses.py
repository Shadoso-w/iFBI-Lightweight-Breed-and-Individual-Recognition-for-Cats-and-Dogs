import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        ##todo
        # mask = torch.eq(labels, labels.T).float().to(device)
        mask = torch.eq(labels, labels.transpose(1, 0)).float().to(device)
        #

        ##todo
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(features, features.T),
        #     self.temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.transpose(1, 0)),
            self.temperature)
        ##
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        #todo
        # mask = torch.eq(labels, labels.T).float().to(device)
        mask = torch.eq(labels, labels.t()).float().to(device)
        # mask = torch.eq(labels, labels.transpose(1, 0)).float().to(device)
        #

        ##todo
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(features, features.T),
        #     self.temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.transpose(1, 0)),
            self.temperature)
        ##

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cpu', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        self.euclidean_metric

    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b) ** 2).sum(dim=2)
        return logits

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
        #                                         dim=2)  # simi_mat: (2*bs, 2*bs)
        similarity_matrix = self.euclidean_metric(representations, representations) # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < self.margin).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + self.margin
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    # todo: add FATriplet Loss
class FATripletLoss(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # tri_margin_loss
        tri_margin_loss = 0
        for neg in negative:
            tri_margin_loss += F.triplet_margin_loss(anchor, positive, neg, margin=self.margin)
        tri_margin_loss /= len(negative)

        # ctrd_margin_loss
        ctrd_margin_loss = torch.mean(F.pairwise_distance(positive, anchor))

        loss = tri_margin_loss + ctrd_margin_loss
        return loss




