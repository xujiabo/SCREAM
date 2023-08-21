import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
import cv2


class RegistrationRender(nn.Module):
    def __init__(self, rho, w, view="muti"):
        super(RegistrationRender, self).__init__()
        self.rho, self.w = rho, w
        gpu = torch.device("cuda:0")

        i, j = np.arange(w * w) // w, np.arange(w * w) % w
        pix_xy = torch.from_numpy(np.concatenate([j.reshape(-1, 1), i.reshape(-1, 1)], axis=1)).float().to(gpu)
        self.pix_xy = (pix_xy - w // 2 + 0.5) / (w // 2)
        if view == "muti":
            self.eulers = [
                # 绕y转
                np.array([0, 0, 0]), np.array([0, np.pi / 2, 0]), np.array([0, np.pi, 0]), np.array([0, np.pi * 3 / 2, 0]),
                # 绕x转
                np.array([0, 0, np.pi / 2]), np.array([0, 0, np.pi * 3 / 2])
            ]
        else:
            self.eulers = [np.array([0, 0, 0])]

    def render(self, x, src_n):
        # x: (n+m) x 3, 前面n个点是src, 后面m个点是tgt
        depth = x[:, 2]
        depth_min = torch.min(depth, dim=0)[0].item()
        pix_val = 1 - (depth - depth_min) / (torch.max(depth, dim=0)[0].item() - depth_min)

        batch_w2 = 64 * 64
        batch_num = (self.w // 64) ** 2

        deep_img_src_result = []
        deep_img_tgt_result = []

        for i in range(batch_num):
            # n x ww
            start, end = i * batch_w2, (i + 1) * batch_w2
            pix_weight = ((x[:, :2].view(-1, 1, 2).repeat([1, batch_w2, 1]) - self.pix_xy[start:end].unsqueeze(0)) ** 2).sum(dim=2)
            pix_weight = torch.exp(-pix_weight / 2 * self.rho ** 2)

            # ww
            deep_img_src = torch.max(pix_val.view(-1, 1)[:src_n] * pix_weight[:src_n], dim=0)[0]
            deep_img_src_result.append(deep_img_src)

            deep_img_tgt = torch.max(pix_val.view(-1, 1)[src_n:] * pix_weight[src_n:], dim=0)[0]
            deep_img_tgt_result.append(deep_img_tgt)

        deep_img_src = torch.cat(deep_img_src_result, dim=0)
        deep_img_tgt = torch.cat(deep_img_tgt_result, dim=0)

        deep_img_src = deep_img_src.view(self.w, self.w)
        deep_img_tgt = deep_img_tgt.view(self.w, self.w)
        # return deep_img_src, deep_img_tgt
        # 2 x w x w
        return torch.stack([deep_img_src, deep_img_tgt], dim=0)

    def forward(self, src_pred, tgt_pcd):
        # n x 3, m x 3 Tensor
        registrated = torch.cat([src_pred, tgt_pcd], dim=0)
        imgs = []

        for euler in self.eulers:
            rot_ab = torch.Tensor(Rotation.from_euler('zyx', euler).as_matrix()).to(src_pred.device)
            img = self.render(torch.matmul(rot_ab, registrated.T).T, src_pred.shape[0])
            imgs.append((img-0.5)/0.5)

        imgs = torch.stack(imgs, dim=0)
        # viewpoint_num x 2 x w x w
        return imgs.contiguous()


if __name__ == '__main__':
    pass
