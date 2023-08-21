import numpy as np
import open3d as o3d
from datasets.kitti import KITTI_PREDATOR
from utils import to_o3d_pcd
from copy import deepcopy


def save_icp():
    ds = KITTI_PREDATOR("C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/KITTI-Registration", mode="train",
                        data_augmentation=False)
    for i in range(len(ds)):
        src_pcd_input, tgt_pcd_input, _, _, rot, trans = ds[i]
        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        # print(src_pcd_input.shape, tgt_pcd_input.shape)
        # src_pc, tgt_pc = to_o3d_pcd(src_pcd_input, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd_input, [0, 0.651, 0.929])
        # open3d.estimate_normals(src_pc)
        # open3d.estimate_normals(tgt_pc)
        #
        # open3d.draw_geometries([src_pc, tgt_pc], window_name="src tgt", width=1000, height=800)
        # open3d.draw_geometries([src_pc.transform(T), tgt_pc], window_name="registration", width=1000, height=800)
        print("\r%d \ %d" % (i + 1, len(ds)))

    ds = KITTI_PREDATOR("C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/KITTI-Registration", mode="val",
                        data_augmentation=False)
    for i in range(len(ds)):
        src_pcd_input, tgt_pcd_input, _, _, rot, trans = ds[i]
        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
        print("\r%d \ %d" % (i + 1, len(ds)))

    ds = KITTI_PREDATOR("C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/KITTI-Registration", mode="test",
                        data_augmentation=False)
    for i in range(len(ds)):
        src_pcd_input, tgt_pcd_input, _, _, rot, trans = ds[i]
        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
        print("\r%d \ %d" % (i + 1, len(ds)))


def save_pair(mode="train"):
    save_item = 0
    ds = KITTI_PREDATOR("C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/KITTI-Registration", mode=mode,
                        data_augmentation=False)
    for i in range(len(ds)):
        if mode == "test" and i == 1:
            continue
        src_pcd_input, tgt_pcd_input, _, _, rot, trans = ds[i]
        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        # print(src_pcd_input.shape, tgt_pcd_input.shape)
        src_pc, tgt_pc = to_o3d_pcd(src_pcd_input, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd_input, [0, 0.651, 0.929])

        # o3d.estimate_normals(src_pc)
        # o3d.estimate_normals(tgt_pc)

        src_down_pc = o3d.voxel_down_sample(src_pc, 0.7)
        tgt_down_pc = o3d.voxel_down_sample(tgt_pc, 0.7)
        print("downsampled  src: %d  tgt: %d" % (np.asarray(src_down_pc.points).shape[0], np.asarray(tgt_down_pc.points).shape[0]))

        # all_pts = np.concatenate([
        #     np.asarray(deepcopy(src_pc).transform(T).points),
        #     np.asarray(tgt_pc.points)
        # ], axis=0)
        # coor_max, coor_min = np.max(all_pts, axis=0), np.min(all_pts, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], i))

        # o3d.draw_geometries([src_pc, tgt_pc], window_name="src tgt", width=1000, height=800)
        # o3d.draw_geometries([src_pc.transform(T), tgt_pc], window_name="registration", width=1000, height=800)

        # o3d.draw_geometries([src_down_pc, tgt_down_pc], window_name="src tgt", width=1000, height=800)
        # o3d.draw_geometries([deepcopy(src_down_pc).transform(T), tgt_down_pc], window_name="registration", width=1000, height=800)

        np.save("./KITTI_%s/src%d.npy" % (mode, save_item), np.asarray(src_down_pc.points))
        np.save("./KITTI_%s/tgt%d.npy" % (mode, save_item), np.asarray(tgt_down_pc.points))
        np.save("./KITTI_%s/T%d.npy" % (mode, save_item), T)
        save_item += 1

        print("\r%d \ %d" % (i + 1, len(ds)))


def check_saved_pairs():
    for i in range(553, -1, -1):
        src_pcd, tgt_pcd, T = np.load("KITTI_test/src%d.npy" % i), np.load("KITTI_test/tgt%d.npy" % i), np.load("KITTI_test/T%d.npy" % i)
        src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])

        all_pts = np.concatenate([
            np.asarray(deepcopy(src_pc).transform(T).points),
            np.asarray(tgt_pc.points)
        ], axis=0)
        coor_max, coor_min = np.max(all_pts, axis=0), np.min(all_pts, axis=0)
        print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], i))

        o3d.estimate_normals(src_pc)
        o3d.estimate_normals(tgt_pc)

        o3d.draw_geometries([src_pc, tgt_pc], window_name="src tgt", width=1000, height=800)
        o3d.draw_geometries([src_pc.transform(T), tgt_pc], window_name="registration", width=1000, height=800)


if __name__ == '__main__':
    # save_icp()
    # save_pair("train")
    # save_pair("val")
    # save_pair("test")
    check_saved_pairs()