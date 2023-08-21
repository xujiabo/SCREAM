import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from models.pointnet import PointTransformer
from models.render import RegistrationRender
from datasets.three_d_match import ThreeDLoMatchTest, ThreeDZeroMatchTest, ThreeDMatchTest
from utils import to_o3d_pcd, square_distance, rigid_transform_3d, transformation_error
import nibabel.quaternions as nq
from copy import deepcopy
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


match, lo_match, zero_match = ThreeDMatchTest(), ThreeDLoMatchTest(), ThreeDZeroMatchTest()
match_loader, lo_loader, zero_loader = DataLoader(match, batch_size=1, shuffle=True), DataLoader(lo_match, batch_size=1, shuffle=True), DataLoader(zero_match, batch_size=1, shuffle=True)

net = PointTransformer(d_model=256)
net.to(device)
net.load_state_dict(torch.load("./params/point-generator.pth"))
net.eval()

blue, yellow, gray = [0, 0.651, 0.929], [1, 0.706, 0], [0.752, 0.752, 0.752]


def RMSE(trans, info):
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]
    return p.item()


def select_pc(pcd, inds):
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(np.asarray(pcd.points)[inds])
    pc.colors = o3d.Vector3dVector(np.asarray(pcd.colors)[inds])
    pc.normals = o3d.Vector3dVector(np.asarray(pcd.normals)[inds])
    return pc


def visualize_zero_match():
    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s, _, covariance, c, _ in zero_loader:
            s = s[0].item()
            c = c.to(device)
            covariance = covariance[0].numpy()

            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            src_pred, imgs, transform = net(
                src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s,
                False, False,
                (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
            )

            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1),
                           torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)

            # transformation
            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_pred / s, tgt_pcd / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < 0.2)
            tgt_ind = valid_ind
            transform = rigid_transform_3d(src_pcd[:, valid_ind] / s + c, src_pred[:, tgt_ind] / s + c)[0]

            if transform is not None:
                re, te = transformation_error(transform, T)

                src_pc = to_o3d_pcd(src_pcd[0] / s + c, [1, 0.706, 0])
                tgt_pc = to_o3d_pcd(tgt_pcd[0] / s + c, [0, 0.651, 0.929])

                final_trans1 = o3d.registration_icp(
                    src_pc, tgt_pc,
                    max_correspondence_distance=0.1,
                    init=transform.cpu().numpy()
                )
                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)

                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    print("refine")
                    transform = final_trans1[0]
                    re, te = re1, te1

            transform = transform.cpu().numpy()
            rmse = np.sqrt(RMSE(np.linalg.inv(T.cpu().numpy()) @ transform, covariance))

            print("re: %.5f  te: %.5f  rmse: %.5f" % (re, te, rmse))

            src_pred_pc = to_o3d_pcd((src_pred[0] / s + c).cpu().numpy(), [3/255, 168/255, 158/255])

            o3d.estimate_normals(src_pc)
            o3d.estimate_normals(tgt_pc)
            o3d.estimate_normals(src_pred_pc)

            # 连线
            line_pts = np.concatenate([np.asarray(src_pc.points), np.asarray(src_pred_pc.points)], axis=0)
            lines = np.concatenate([
                np.arange(src_pcd.shape[1]).reshape(-1, 1),
                np.arange(src_pcd.shape[1]).reshape(-1, 1) + src_pcd.shape[1]
            ], axis=1).tolist()
            colors = [[224/255, 238/255, 238/255]] * len(lines)
            # print(len(lines))
            # colors = [[0, 1, 0]] * len(lines)
            # 绘制直线
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.Vector3dVector(line_pts)
            line_set.lines = o3d.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # selected correspondence
            src_selected_pc = select_pc(src_pc, valid_ind.cpu().numpy())
            tgt_selected_pc = select_pc(src_pred_pc, valid_ind.cpu().numpy())

            # re小于3的视觉效果比较好，rmse小于0.2的也有配准得不太好的情况
            if re < 3:
                o3d.draw_geometries([tgt_pc, src_pc], window_name="tgt and src", width=1000, height=800)
                o3d.draw_geometries([tgt_pc, src_pred_pc], window_name="tgt and src_pred", width=1000, height=800)
                # o3d.draw_geometries([src_pc, src_pred_pc, line_set], window_name="tgt and src_pred", width=1000, height=800)

                o3d.draw_geometries([src_selected_pc, tgt_selected_pc], window_name="src and src_pred selected", width=1000, height=800)
                o3d.draw_geometries([tgt_pc, deepcopy(src_pc).transform(transform)], window_name="registration", width=1000, height=800)
                o3d.draw_geometries([tgt_pc, deepcopy(src_pc).transform(T.cpu().numpy())], window_name="GT", width=1000, height=800)


def visualize_3dmatch(loader):
    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s, _, covariance, c, _ in loader:
            s = s[0].item()
            c = c.to(device)
            covariance = covariance[0].numpy()

            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            src_pred, imgs, transform = net(
                src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s,
                False, False,
                (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
            )

            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1),
                           torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)

            # transformation
            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_pred / s, tgt_pcd / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < 0.1)
            tgt_ind = src_pred_2_tgt_ind[valid_ind]
            transform = rigid_transform_3d(src_pcd[:, valid_ind] / s + c, tgt_pcd[:, tgt_ind] / s + c)[0]

            if transform is not None:
                re, te = transformation_error(transform, T)

                src_pc = to_o3d_pcd(src_pcd[0] / s + c, [1, 0.706, 0])
                tgt_pc = to_o3d_pcd(tgt_pcd[0] / s + c, [0, 0.651, 0.929])

                final_trans1 = o3d.registration_icp(
                    src_pc, tgt_pc,
                    max_correspondence_distance=0.1,
                    init=transform.cpu().numpy()
                )
                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)

                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    print("refine")
                    transform = final_trans1[0]
                    re, te = re1, te1

            transform = transform.cpu().numpy()
            rmse = np.sqrt(RMSE(np.linalg.inv(T.cpu().numpy()) @ transform, covariance))

            print("re: %.5f  te: %.5f  rmse: %.5f" % (re, te, rmse))

            src_pred_pc = to_o3d_pcd((src_pred[0] / s + c).cpu().numpy(), [3/255, 168/255, 158/255])

            o3d.estimate_normals(src_pc)
            o3d.estimate_normals(tgt_pc)
            o3d.estimate_normals(src_pred_pc)

            # selected correspondence
            src_selected_pc = select_pc(src_pc, valid_ind.cpu().numpy())
            tgt_selected_pc = select_pc(src_pred_pc, valid_ind.cpu().numpy())

            o3d.draw_geometries([tgt_pc, src_pc], window_name="tgt and src", width=1000, height=800)
            o3d.draw_geometries([tgt_pc, src_pred_pc], window_name="tgt and src_pred", width=1000, height=800)
            o3d.draw_geometries([src_selected_pc, tgt_selected_pc], window_name="src and pred selected", width=1000, height=800)
            o3d.draw_geometries([tgt_pc, deepcopy(src_pc).transform(transform)], window_name="registration", width=1000, height=800)
            o3d.draw_geometries([tgt_pc, deepcopy(src_pc).transform(T.cpu().numpy())], window_name="GT", width=1000, height=800)

            # print("save? (y/n)")
            # op = input()
            # if op == "y":
            #     np.save("experiments/render/tgt.npy", np.asarray(tgt_pc.points))
            #     np.save("experiments/render/src_gt.npy", np.asarray(deepcopy(src_pc).transform(T.cpu().numpy()).points))
            #     np.save("experiments/render/src_pred.npy", np.asarray(src_pred_pc.points))


if __name__ == '__main__':
    visualize_3dmatch(lo_loader)
    # visualize_zero_match()