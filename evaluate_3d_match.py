import numpy as np
import open3d as o3d
import torch
from torch.utils import data
from datasets.three_d_match import ThreeDMatchTest, ThreeDLoMatchTest, ThreeDZeroMatchTest
from utils import processbar, transformation_error, square_distance
from copy import deepcopy
from utils import to_o3d_pcd, rigid_transform_3d
from models.pointnet import PointTransformer
import nibabel.quaternions as nq

device = torch.device("cuda:0")

match_set = ThreeDMatchTest()
match_set.au = False
lo_match_set = ThreeDLoMatchTest()
lo_match_set.au = False
zero_match_set = ThreeDZeroMatchTest()
zero_match_set.au = False

match_loader = data.DataLoader(dataset=match_set, batch_size=1, shuffle=False)
lo_match_loader = data.DataLoader(dataset=lo_match_set, batch_size=1, shuffle=False)
zero_match_loader = data.DataLoader(dataset=zero_match_set, batch_size=1, shuffle=False)

# net = PointTransformer(d_model=256)
# net.to(device)
# net.load_state_dict(torch.load("./params/point-generator.pth"))
# net.eval()


def RMSE(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html

    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]

    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


def evaluate_loader(net, loader, corr="tgt", dis_thresh=0.1, re_static_method="median"):
    processed = 0
    rre, rte = 0, 0
    point_trans_loss = 0
    get_transform = False

    success_pair_num = 0

    scene_names = ['Kitchen', 'Home_1', 'Home_2', 'Hotel_1', 'Hotel_2', 'Hotel_3', 'Study', 'MIT_Lab']

    metric = {
        'Kitchen': [[], [], 0, 0],  # re te success_num valid_num
        'Home_1': [[], [], 0, 0],
        'Home_2': [[], [], 0, 0],
        'Hotel_1': [[], [], 0, 0],
        'Hotel_2': [[], [], 0, 0],
        'Hotel_3': [[], [], 0, 0],
        'Study': [[], [], 0, 0],
        'MIT_Lab': [[], [], 0, 0]
    }

    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s, idx, covariance, c, scene_idx in loader:
            s = s[0].item()
            c = c.to(device)
            scene_idx = scene_idx[0].item()
            scene_name = scene_names[scene_idx]

            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            idx, covariance = idx[0].numpy().tolist(), covariance[0].numpy()
            src_pred, imgs, transform = net(
                src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s,
                False, get_transform,
                (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
            )

            # T = torch.cat([torch.cat([rot[0], trans[0]], dim=1), torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1), torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            re, te = 0, 0

            # transformation
            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_pred / s, tgt_pcd / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < dis_thresh)
            if corr == "tgt":
                tgt_ind = src_pred_2_tgt_ind[valid_ind]
                transform = rigid_transform_3d(src_pcd[:, valid_ind] / s + c, tgt_pcd[:, tgt_ind] / s + c)[0]
            else:
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
                    transform = final_trans1[0]
                    re, te = re1, te1

            transform = transform.cpu().numpy()
            rmse = np.sqrt(RMSE(np.linalg.inv(T.cpu().numpy()) @ transform, covariance))

            if rmse < 0.2:
                success_pair_num += 1
                rre, rte = rre + re, rte + te

            # follow PREDATOR, j - i > 1
            idx_diff = abs(idx[1] - idx[0])
            if idx_diff > 1:
                metric[scene_name][3] += 1
                if rmse < 0.2:
                    metric[scene_name][2] += 1
                    metric[scene_name][0].append(re.item())
                    metric[scene_name][1].append(te.item())
                else:
                    metric[scene_name][0].append(0)
                    metric[scene_name][1].append(0)

            point_loss = net.loss(src_pred, src_pcd, rot, trans)
            point_trans_loss += point_loss.item()

            processed += 1
            print("\r进度：%s  point trans loss: %.5f  re: %.5f  te: %.5f  rmse: %.5f  rr: %.5f" % (
                processbar(processed, len(loader.dataset)), point_loss.item(), re, te, rmse, success_pair_num / processed
            ), end="")
    rre, rte = rre / len(loader.dataset), rte / len(loader.dataset)
    rr = success_pair_num / len(loader.dataset)
    point_trans_loss = point_trans_loss / len(loader.dataset)
    print("\ntest finish  loss: %.5f  rre: %.5f  rte: %.5f  rr: %.5f" % (point_trans_loss, rre, rte, rr))

    rre_mean, rte_mean, rr_mean = 0, 0, 0
    for scene_name in metric.keys():
        if re_static_method == "median":
            rre = np.median(np.array(metric[scene_name][0])).item()
            rte = np.median(np.array(metric[scene_name][1])).item()
        else:
            rre = np.mean(np.array(metric[scene_name][0])).item()
            rte = np.mean(np.array(metric[scene_name][1])).item()
        rr = metric[scene_name][2] / metric[scene_name][3]

        rre_mean += rre
        rte_mean += rte
        rr_mean += rr

        print("%s: rre: %.5f  rte: %.5f  rr: %.5f" % (scene_name, rre, rte, rr))

    rre, rte, rr = rre_mean / 8, rte_mean / 8, rr_mean / 8
    print("mean: rre: %.5f  rte: %.5f  rr: %.5f" % (rre, rte, rr))

    return point_trans_loss, rre, rte, rr


def evaluate_3d_lo_match(net, dis_thresh=0.1):
    return evaluate_loader(net, lo_match_loader, dis_thresh=dis_thresh)


def evaluate_3d_match(net, dis_thresh=0.1):
    return evaluate_loader(net, match_loader, dis_thresh=dis_thresh)


def evaluate_3d_zero_match(net, dis_thresh=0.1):
    return evaluate_loader(net, zero_match_loader, corr="src_pred", dis_thresh=dis_thresh, re_static_method="mean")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PointTransformer(d_model=256)
    net.to(device)
    net.load_state_dict(torch.load("./params/point-generator.pth"))
    net.eval()

    # evaluate_3d_lo_match(net, dis_thresh=0.1)
    evaluate_3d_match(net, dis_thresh=0.1)
    # evaluate_3d_zero_match(net, dis_thresh=0.2)