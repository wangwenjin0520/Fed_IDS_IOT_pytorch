from torch import nn
import torch


class TripleLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripleLoss, self).__init__()
        self.margin = margin  # 阈值
        self.rank_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels, norm=False):
        dist_mat = self.euclidean_dist(inputs, inputs, norm=norm)  # 距离矩阵
        dist_ap, dist_an = self.hard_sample(dist_mat, labels)  # 取出每个anchor对应的最大
        y = torch.ones_like(dist_an)  # 系数矩阵，1/-1
        loss = self.rank_loss(dist_ap, dist_an, y)
        return loss

    @staticmethod
    def hard_sample(dist_mat, labels, ):
        # 距离矩阵的尺寸是 (batch_size, batch_size)
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # 选出所有正负样本对
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # 两两组合， 取label相同的a-p
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # 两两组合， 取label不同的a-n

        list_ap, list_an = [], []
        # 取出所有正样本对和负样本对的距离值
        for i in range(N):
            list_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            list_an.append(dist_mat[i][is_neg[i]].max().unsqueeze(0))
            dist_ap = torch.cat(list_ap)  # 将list里的tensor拼接成新的tensor
            dist_an = torch.cat(list_an)
        return dist_ap, dist_an

    def normalize(self, x, axis=1):
        x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
        return x

    def euclidean_dist(self, x, y, norm=True):
        if norm:
            x = self.normalize(x)
            y = self.normalize(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        yy = torch.t(torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m))
        dist = xx + yy  # 任意的两个样本组合， 求第二范数后求和 x^2 + y^2
        dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=torch.t(y))  # (x-y)^2 = x^2 + y^2 - 2xy
        dist = dist.clamp(min=1e-12).sqrt()
        return dist
