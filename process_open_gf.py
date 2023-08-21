import numpy as np
import open3d as o3d
import torch
import os
# import laspy
from torch.utils import data
from scipy.spatial.transform import Rotation
from utils import to_o3d_pcd, square_distance
from copy import deepcopy
import cv2


class OpenGFTrainNpSaver(data.Dataset):
    def __init__(self, root):
        # gentle + dense
        self.filelist = ["%s/train/S7_%d.laz" % (root, i) for i in range(1, 10)]
        # steep + sparse
        self.filelist = self.filelist + ["%s/train/S8_%d.laz" % (root, i) for i in range(1, 10)]
        # steep + sparse
        self.filelist = self.filelist + ["%s/train/S9_%d.laz" % (root, i) for i in range(1, 20)]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        laz = laspy.read(self.filelist[item])
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]
        name = self.filelist[item][self.filelist[item].rfind("/") + 1:self.filelist[item].rfind(".")]

        x = np.concatenate([inp, clz.reshape(-1, 1)-1], axis=1)

        return x, name


class OpenGFValNpSaver(data.Dataset):
    def __init__(self, root):
        # 7: gentle + dense
        # 8: steep + sparse
        # 9: steep + sparse
        self.filelist = ["%s/val/S%d_v.laz" % (root, i) for i in range(7, 10)]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        laz = laspy.read(self.filelist[item])
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]
        name = self.filelist[item][self.filelist[item].rfind("/") + 1:self.filelist[item].rfind(".")]

        x = np.concatenate([inp, clz.reshape(-1, 1) - 1], axis=1)

        return x, name



class OpenGFTrainSplit(data.Dataset):
    def __init__(self, root="./OpenGF_np"):
        # gentle + dense
        self.filelist = ["%s/train/S7_%d.npy" % (root, i) for i in range(1, 10)]
        # steep + sparse
        self.filelist = self.filelist + ["%s/train/S8_%d.npy" % (root, i) for i in range(1, 10)]
        # steep + sparse
        self.filelist = self.filelist + ["%s/train/S9_%d.npy" % (root, i) for i in range(1, 20)]

    def __len__(self):
        return len(self.filelist) * 17 * 17

    def __getitem__(self, index):
        file_idx = index // (17 * 17)
        # laz = laspy.read(self.filelist[file_idx])
        laz = np.load(self.filelist[file_idx])
        inp, clz = laz[:, :3], laz[:, 3]

        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % (17 * 17)
        xrange = [[0, 100], [25, 125], [50, 150], [75, 175], [100, 200], [125, 225], [150, 250], [175, 275], [200, 300],
                  [225, 325], [250, 350], [275, 375], [300, 400], [325, 425], [350, 450], [375, 475], [400, 500]]
        yrange = [[0, 100], [25, 125], [50, 150], [75, 175], [100, 200], [125, 225], [150, 250], [175, 275], [200, 300],
                  [225, 325], [250, 350], [275, 375], [300, 400], [325, 425], [350, 450], [375, 475], [400, 500]]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 17]
        x = xrange[range_idx % 17]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        sub_xyz, sub_cls = inp[idx], clz[idx]

        # center = np.mean(sub_xyz, axis=0)
        # sub_xyz = (sub_xyz - center.reshape(1, 3))

        return sub_xyz, sub_cls


class OpenGFValSplit(data.Dataset):
    def __init__(self, root="./OpenGF_np"):
        # S7_v, S8_v, S9_v
        self.filelist = ["%s/val/S%d_v.npy" % (root, i) for i in range(7, 10)]

    def __len__(self):
        return len(self.filelist) * 5 * 5

    def __getitem__(self, index):
        file_idx = index // (5 * 5)
        # laz = laspy.read(self.filelist[file_idx])
        laz = np.load(self.filelist[file_idx])
        inp, clz = laz[:, :3], laz[:, 3]

        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % (5 * 5)
        xrange = [[0, 100], [100, 200], [200, 300], [300, 400], [400, 500]]
        yrange = [[0, 100], [100, 200], [200, 300], [300, 400], [400, 500]]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 5]
        x = xrange[range_idx % 5]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        sub_xyz, sub_cls = inp[idx], clz[idx]

        # center = np.mean(sub_xyz, axis=0)
        # sub_xyz = (sub_xyz - center.reshape(1, 3))

        return sub_xyz, sub_cls


class OpenGFTestSplit(data.Dataset):
    def __init__(self):
        laz = np.load("./OpenGF_np/test/T1.npy")
        self.inp, self.clz = laz[:, :3], laz[:, 3]
        self.coor_max, self.coor_min = np.max(self.inp, axis=0), np.min(self.inp, axis=0)

    def __len__(self):
        return 26 * 25

    def __getitem__(self, index):
        inp, clz = self.inp, self.clz

        coor_max, coor_min = self.coor_max, self.coor_min
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % (26 * 25)
        xrange = [[st, st+100] for st in range(0, 2600, 100)]
        yrange = [[st, st+100] for st in range(0, 2500, 100)]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 26]
        x = xrange[range_idx % 26]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        sub_xyz, sub_cls = inp[idx], clz[idx]

        # center = np.mean(sub_xyz, axis=0)
        # sub_xyz = (sub_xyz - center.reshape(1, 3))

        return sub_xyz, sub_cls


def save_open_gf_as_np():
    opengf = OpenGFTrainNpSaver("E:/OpenGF_Exp")
    for i in range(len(opengf)):
        x, name = opengf[i]
        np.save("./OpenGF_np/train/%s.npy" % name, x)
        print("\r%d / %d" % (i + 1, len(opengf)), end="")
    print()
    opengf = OpenGFValNpSaver("E:/OpenGF_Exp")
    for i in range(len(opengf)):
        x, name = opengf[i]
        np.save("./OpenGF_np/val/%s.npy" % name, x)
        print("\r%d / %d" % (i + 1, len(opengf)), end="")
    print()


def split_dataset_as_patch(dataset, dataset_name="train", save_center=False):
    # opengf = OpenGFTrainSplit()
    opengf = dataset
    print(len(opengf))
    save_item = 1
    resolution = 1.
    for i in range(0, len(opengf)):
        patch, patch_cls = opengf[i]
        # print(np.sum(patch_cls == 2))
        patch_pc = to_o3d_pcd(patch, [0, 0.651, 0.929])
        dem = patch[patch_cls == 1]
        dem_pc = to_o3d_pcd(dem, [104 / 255, 131 / 255, 139 / 255])
        # dem_coarse_pc = o3d.voxel_down_sample(deepcopy(dem_pc), 15)
        # np.asarray(dem_coarse_pc.colors)[:] = np.array([1, 0, 0])
        dem_pc = o3d.voxel_down_sample(dem_pc, resolution)
        dem = np.asarray(dem_pc.points)
        print("downsampled dem n: %d" % dem.shape[0])

        patch_pc = o3d.voxel_down_sample(patch_pc, resolution)
        patch = np.asarray(patch_pc.points)
        print("downsampled patch n: %d" % patch.shape[0])

        patch, dem = torch.Tensor(patch).cuda(), torch.Tensor(dem).cuda()

        dsm = []

        for j in range(dem.shape[0]):
            dis = torch.norm(patch[:, :2] - dem[j, :2].view(1, 2), dim=1)
            if torch.sum((dis <= 0.8), dim=0).item() == 0:
                dsm.append(dem[j, :].cpu().numpy())
                continue
            pts = patch[dis <= 0.8]
            highest_idx = torch.max(pts[:, 2], dim=0)[1].item()
            highest_pt = pts[highest_idx]
            dsm.append(highest_pt.cpu().numpy())
            print("\rextracting dsm: %d / %d" % (j + 1, dem.shape[0]), end="")
        print()

        dsm = np.stack(dsm, axis=0)


        all_pts = np.concatenate([dsm, dem.cpu().numpy()], axis=0)

        # print(np.mean(all_pts, axis=0))
        coor_max, coor_min = np.max(all_pts, axis=0), np.min(all_pts, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], i))

        center = (coor_min + coor_max) / 2
        center = center.reshape(1, 3)
        dsm_dem = np.concatenate([dsm-center, dem.cpu().numpy()-center], axis=1)


        all_pts = np.concatenate([dsm_dem[:, :3], dsm_dem[:, 3:]], axis=0)

        print(np.mean(all_pts, axis=0))
        coor_max, coor_min = np.max(all_pts, axis=0), np.min(all_pts, axis=0)
        print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (
            coor_min[0], coor_max[0], coor_max[0] - coor_min[0], coor_min[1], coor_max[1], coor_max[1] - coor_min[1],
            coor_min[2], coor_max[2], coor_max[2] - coor_min[2], i)
        )
        # #
        # dsm_pc = to_o3d_pcd(dsm_dem[:, :3], [0, 0.651, 0.929])
        # dem_pc = to_o3d_pcd(dsm_dem[:, 3:], [104 / 255, 131 / 255, 139 / 255])
        # o3d.draw_geometries([dsm_pc, dem_pc], width=1000, height=800, window_name="dsm dem")
        # o3d.draw_geometries([dem_pc], width=1000, height=800, window_name="dsm dem")

        np.save("./OpenGF_%s/%d.npy" % (dataset_name, i+1), dsm_dem)
        if save_center:
            np.save("./OpenGF_%s/centers/%d.npy" % (dataset_name, i + 1), center)
        save_item += 1
        print("\r%s: %d / %d" % (dataset_name, i + 1, len(opengf)), end="\n")


def check_test():
    all_patch_dsm, all_patch_dem, all_high = [], [], []
    for i in range(1, 650+1):
        dsm_dem, center = np.load("./OpenGF_test/%d.npy" % i), np.load("./OpenGF_test/centers/%d.npy" % i)
        dsm, dem = dsm_dem[:, :3], dsm_dem[:, 3:]

        all_patch_dsm.append(dsm + center)
        all_patch_dem.append(dem + center)
        all_high.append(dsm[:, 2] - dem[:, 2])

        print("\r%d / %d" % (i, 650), end="")
    print()

    all_patch_dsm, all_patch_dem, all_high = np.concatenate(all_patch_dsm, axis=0), np.concatenate(all_patch_dem, axis=0), np.concatenate(all_high, axis=0)

    max_high = np.max(all_high).item()
    all_high = all_high / max_high

    heatmap = cv2.applyColorMap(np.uint8(255 * all_high), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float).reshape(-1, 3) / 255
    heatmap = heatmap[:, [2, 1, 0]]

    dsm_pc = to_o3d_pcd(all_patch_dsm, [3/255, 168/255, 158/255])
    dem_pc = to_o3d_pcd(all_patch_dem)
    dem_pc.colors = o3d.Vector3dVector(heatmap)

    o3d.estimate_normals(dsm_pc)
    o3d.estimate_normals(dem_pc)

    o3d.draw_geometries([dsm_pc, dem_pc], width=1200, height=1000, window_name="dsm dem")
    o3d.draw_geometries([dem_pc], width=1200, height=1000, window_name="dem")


if __name__ == '__main__':
    # # process train
    # split_dataset_as_patch(OpenGFTrainSplit(), "train")
    # # process val
    # split_dataset_as_patch(OpenGFValSplit(), "val")
    # # process test
    # split_dataset_as_patch(OpenGFTestSplit(), "test", True)
    check_test()