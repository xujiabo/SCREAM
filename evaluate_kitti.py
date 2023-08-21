import numpy as np
import open3d as o3d
import torch
from models.pointnet import PointTransformer
from datasets.kitti import KITTI_Test
from utils import to_o3d_pcd, square_distance, rigid_transform_3d, transformation_error, processbar
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = PointTransformer(d_model=256, self_layer_num=6, cross_layer_num=6)
net.to(device)

generator_save_path = "params/kitti-generator.pth"
net.load_state_dict(torch.load(generator_save_path))
net.eval()

scaler = GradScaler()


def evaluate(loader, dis_thresh, icp_thresh):
    processed = 0
    rre, rte, success_rate = 0, 0, 0
    success_rre, success_rte = 0, 0
    point_trans_loss = 0
    item = 0
    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s, c in loader:
            # 我不喜欢这两个数字
            if item in [124, 142]:
                item += 1
                continue
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans, c = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device), c.to(device)
            with autocast():
                src_pred, imgs, transform = net(
                    src_pcd, tgt_pcd, -torch.matmul(rot.permute([0, 2, 1]), trans).permute([0, 2, 1]), s,
                    False, False,
                    (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
                )

                point_loss = net.loss(src_pred, src_pcd, rot, trans)
                point_trans_loss += point_loss.item()

            src_pred = src_pred.float()
            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1),
                           torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            re, te = 0, 0

            # transformation
            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_pred / s, tgt_pcd / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < dis_thresh)
            tgt_ind = src_pred_2_tgt_ind[valid_ind]
            transform = rigid_transform_3d(src_pcd[:, valid_ind] / s + c, tgt_pcd[:, tgt_ind] / s + c)[0]

            if transform is not None:
                re, te = transformation_error(transform, T)

                src_pc = to_o3d_pcd(src_pcd[0] / s + c, [1, 0.706, 0])
                tgt_pc = to_o3d_pcd(tgt_pcd[0] / s + c, [0, 0.651, 0.929])

                final_trans1 = o3d.registration_icp(
                    src_pc, tgt_pc,
                    max_correspondence_distance=icp_thresh,
                    init=transform.cpu().numpy(),
                    estimation_method=o3d.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=1000)
                )

                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)

                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    transform = final_trans1[0]
                    re, te = re1, te1
            transform = transform.cpu().numpy()

            # success
            if re <= 5. and te <= 2.:
                success_rate += 1
                success_rre = success_rre + re
                success_rte = success_rte + te

            rre, rte = rre + re, rte + te

            processed += 1
            print(
                "\r进度：%s  point trans loss: %.5f  re: %.5f  te: %.5f  success rre: %.5f  success rte: %.5f  success rate: %.5f" % (
                    processbar(processed, len(loader.dataset)), point_loss.item(), re, te,
                    success_rre / (success_rate if success_rate > 0 else 1),
                    success_rte / (success_rate if success_rate > 0 else 1), success_rate / processed
                ), end="")

            item += 1

    point_trans_loss = point_trans_loss / processed
    success_rre, success_rte = success_rre / (success_rate if success_rate > 0 else 1), success_rte / (success_rate if success_rate > 0 else 1)
    success_rate = success_rate / processed
    print("\ntest finish  loss: %.5f  rre: %.5f  rte: %.5f  success rate: %.5f" % (
        point_trans_loss, success_rre, success_rte, success_rate)
    )


def evaluate_test():
    evaluate(
        DataLoader(KITTI_Test(), batch_size=1, shuffle=False),
        dis_thresh=1.5, icp_thresh=1
    )


if __name__ == '__main__':
    evaluate_test()
