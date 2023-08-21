import numpy as np
import cv2
import open3d as o3d
import torch
from models.render import RegistrationRender
from datasets.three_d_match import ThreeDLoMatchTest, ThreeDZeroMatchTest
from datasets.kitti import KITTI_Test
from torch.utils.data import DataLoader
from utils import to_o3d_pcd, deep_to_img


if __name__ == '__main__':

    lo_match, zero_match = ThreeDLoMatchTest(), ThreeDZeroMatchTest()
    lo_loader, zero_loader = DataLoader(lo_match, batch_size=1, shuffle=True), DataLoader(zero_match, batch_size=1, shuffle=True)
    kitti_loader = DataLoader(KITTI_Test(), batch_size=1, shuffle=False)
    device = torch.device("cuda:0")

    # 3dmatch
    re = RegistrationRender(rho=24, w=128)
    for src_pcd, tgt_pcd, rot, trans, s, _, covariance, c, _ in lo_loader:
        src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
        src_real = (torch.matmul(rot[0].detach(), src_pcd[0].detach().t()) + trans[0].detach()).t()

        imgs = re(src_real, tgt_pcd[0])

        imgs = imgs.permute([0, 2, 3, 1])
        img_cv = np.concatenate([deep_to_img(imgs[j]) for j in range(imgs.shape[0])], axis=1)

        src_pc = to_o3d_pcd(src_real, [1, 0.706, 0])
        tgt_pc = to_o3d_pcd(tgt_pcd[0], [0, 0.651, 0.929])
        o3d.draw_geometries([src_pc, tgt_pc], window_name="src tgt", width=1000, height=800)

        cv2.imshow("deepth images", img_cv)
        cv2.waitKey(0)
    # # kitti
    # re = RegistrationRender(rho=48, w=128, view="single")
    # re.eulers = [np.array([0, np.pi, 0])]
    # for src_pcd, tgt_pcd, rot, trans, s, c in kitti_loader:
    #     src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
    #     src_real = (torch.matmul(rot[0].detach(), src_pcd[0].detach().t()) + trans[0].detach()).t()
    #
    #     imgs = re(src_real, tgt_pcd[0])
    #
    #     imgs = imgs.permute([0, 2, 3, 1])
    #     img_cv = np.concatenate([deep_to_img(imgs[j]) for j in range(imgs.shape[0])], axis=1)
    #
    #     src_pc = to_o3d_pcd(src_real, [1, 0.706, 0])
    #     tgt_pc = to_o3d_pcd(tgt_pcd[0], [0, 0.651, 0.929])
    #     o3d.draw_geometries([src_pc, tgt_pc], window_name="src tgt", width=1000, height=800)
    #
    #     cv2.imshow("deepth images", img_cv)
    #     cv2.waitKey(0)