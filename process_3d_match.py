import numpy as np
import open3d as o3d
from datasets.three_d_match import ThreeDMatchDataset_PREDATOR
from utils import to_o3d_pcd
from copy import deepcopy


def save_train():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="train")
    print(len(datas))
    item = 0
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, _, _, _ = datas[i]
        src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])
        o3d.estimate_normals(src_pc)
        o3d.estimate_normals(tgt_pc)

        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]
        # print("overlap: %.5f" % overlap_ratio)

        # 去重叠
        non_overlap_ind = np.setdiff1d(np.arange(src_pcd.shape[0]), src_overlap_ind)
        src_zero_overlap_pcd = src_pcd[non_overlap_ind]

        src_zero_overlap_pc = to_o3d_pcd(src_zero_overlap_pcd, [1, 0.706, 0])
        o3d.estimate_normals(src_zero_overlap_pc)

        voxel_size = 0.0625
        src_pc, tgt_pc = o3d.voxel_down_sample(src_pc, voxel_size=voxel_size), o3d.voxel_down_sample(tgt_pc, voxel_size=voxel_size)
        src_zero_overlap_pc = o3d.voxel_down_sample(src_zero_overlap_pc, voxel_size=voxel_size)

        # o3d.draw_geometries([deepcopy(src_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)
        # o3d.draw_geometries([deepcopy(src_zero_overlap_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)

        # save
        np.save("3DMatch_train/src%d.npy" % item, np.asarray(src_pc.points))
        np.save("3DMatch_train/tgt%d.npy" % item, np.asarray(tgt_pc.points))
        np.save("3DMatch_train/T%d.npy" % item, T)
        item += 1

        if overlap_ratio <= 0.3:
            np.save("3DMatch_train/src%d.npy" % item, np.asarray(src_zero_overlap_pc.points))
            np.save("3DMatch_train/tgt%d.npy" % item, np.asarray(tgt_pc.points))
            np.save("3DMatch_train/T%d.npy" % item, T)
            item += 1

        print("\r%d / %d" % (i+1, len(datas)), end="")


def save_val():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="val")
    print(len(datas))
    item = 0
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, _, _, _ = datas[i]
        src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])
        o3d.estimate_normals(src_pc)
        o3d.estimate_normals(tgt_pc)

        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]
        # print("overlap: %.5f" % overlap_ratio)

        # 去重叠
        non_overlap_ind = np.setdiff1d(np.arange(src_pcd.shape[0]), src_overlap_ind)
        src_zero_overlap_pcd = src_pcd[non_overlap_ind]

        src_zero_overlap_pc = to_o3d_pcd(src_zero_overlap_pcd, [1, 0.706, 0])
        o3d.estimate_normals(src_zero_overlap_pc)

        voxel_size = 0.0625
        src_pc, tgt_pc = o3d.voxel_down_sample(src_pc, voxel_size=voxel_size), o3d.voxel_down_sample(tgt_pc, voxel_size=voxel_size)
        src_zero_overlap_pc = o3d.voxel_down_sample(src_zero_overlap_pc, voxel_size=voxel_size)

        # o3d.draw_geometries([deepcopy(src_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)
        # o3d.draw_geometries([deepcopy(src_zero_overlap_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)

        # save
        np.save("3DMatch_val/src%d.npy" % item, np.asarray(src_pc.points))
        np.save("3DMatch_val/tgt%d.npy" % item, np.asarray(tgt_pc.points))
        np.save("3DMatch_val/T%d.npy" % item, T)
        item += 1

        if overlap_ratio <= 0.3:
            np.save("3DMatch_val/src%d.npy" % item, np.asarray(src_zero_overlap_pc.points))
            np.save("3DMatch_val/tgt%d.npy" % item, np.asarray(tgt_pc.points))
            np.save("3DMatch_val/T%d.npy" % item, T)
            item += 1

        print("\r%d / %d" % (i+1, len(datas)), end="")


def save_test_3DMatch():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="test 3DMatch")
    print(len(datas))
    item = 0
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, _, _, _ = datas[i]
        src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])
        o3d.estimate_normals(src_pc)
        o3d.estimate_normals(tgt_pc)

        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]
        # print("overlap: %.5f" % overlap_ratio)

        # 去重叠
        # non_overlap_ind = np.setdiff1d(np.arange(src_pcd.shape[0]), src_overlap_ind)
        # src_zero_overlap_pcd = src_pcd[non_overlap_ind]

        # src_zero_overlap_pc = to_o3d_pcd(src_zero_overlap_pcd, [1, 0.706, 0])
        # o3d.estimate_normals(src_zero_overlap_pc)

        voxel_size = 0.0625
        src_pc, tgt_pc = o3d.voxel_down_sample(src_pc, voxel_size=voxel_size), o3d.voxel_down_sample(tgt_pc, voxel_size=voxel_size)
        # src_zero_overlap_pc = o3d.voxel_down_sample(src_zero_overlap_pc, voxel_size=voxel_size)

        # o3d.draw_geometries([deepcopy(src_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)
        # o3d.draw_geometries([deepcopy(src_zero_overlap_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)

        if overlap_ratio > 0.3:
            # save
            np.save("3DMatch_test/src%d.npy" % item, np.asarray(src_pc.points))
            np.save("3DMatch_test/tgt%d.npy" % item, np.asarray(tgt_pc.points))
            np.save("3DMatch_test/T%d.npy" % item, T)
            item += 1

        print("\r%d / %d" % (i + 1, len(datas)), end="")


def save_lo_and_zero_match():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="test 3DLoMatch")
    print(len(datas))
    item = 0
    iter_zero = 0
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, _, _, _ = datas[i]
        src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])
        o3d.estimate_normals(src_pc)
        o3d.estimate_normals(tgt_pc)

        T = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]
        # print("overlap: %.5f" % overlap_ratio)

        # 去重叠
        non_overlap_ind = np.setdiff1d(np.arange(src_pcd.shape[0]), src_overlap_ind)
        src_zero_overlap_pcd = src_pcd[non_overlap_ind]

        src_zero_overlap_pc = to_o3d_pcd(src_zero_overlap_pcd, [1, 0.706, 0])
        o3d.estimate_normals(src_zero_overlap_pc)

        voxel_size = 0.0625
        src_pc, tgt_pc = o3d.voxel_down_sample(src_pc, voxel_size=voxel_size), o3d.voxel_down_sample(tgt_pc,
                                                                                                     voxel_size=voxel_size)
        src_zero_overlap_pc = o3d.voxel_down_sample(src_zero_overlap_pc, voxel_size=voxel_size)

        # o3d.draw_geometries([deepcopy(src_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)
        # o3d.draw_geometries([deepcopy(src_zero_overlap_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)

        if 0.1 < overlap_ratio:
            # save
            np.save("3DLoMatch_test/src%d.npy" % item, np.asarray(src_pc.points))
            np.save("3DLoMatch_test/tgt%d.npy" % item, np.asarray(tgt_pc.points))
            np.save("3DLoMatch_test/T%d.npy" % item, T)
            item += 1

        if overlap_ratio <= 0.3:
            np.save("3DZeroMatch_test/src%d.npy" % iter_zero, np.asarray(src_zero_overlap_pc.points))
            np.save("3DZeroMatch_test/tgt%d.npy" % iter_zero, np.asarray(tgt_pc.points))
            np.save("3DZeroMatch_test/T%d.npy" % iter_zero, T)
            iter_zero += 1

        print("\r%d / %d" % (i + 1, len(datas)), end="")


def save_test_3DMatch_info():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="test 3DMatch")
    print(len(datas))
    item = 0
    scene_names = []
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, idx, covariance, scene_name = datas[i]

        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]

        if overlap_ratio > 0.3:
            # save
            # np.save("3DMatch_test/info/idx%d.npy" % item, idx)
            # np.save("3DMatch_test/info/covariance%d.npy" % item, covariance)
            item += 1
            scene_names.append(scene_name+"\n")
        print("\r%d / %d" % (i + 1, len(datas)), end="")
    with open("3DMatch_test/info/scene_names.txt", "w") as f:
        f.writelines(scene_names)
    print()


def save_lo_and_zero_match_info():
    datas = ThreeDMatchDataset_PREDATOR(root="D:/indoor", mode="test 3DLoMatch")
    print(len(datas))
    item = 0
    iter_zero = 0
    scene_names_lo = []
    scene_names_zero = []
    for i in range(len(datas)):
        src_pcd, tgt_pcd, rot, trans, src_overlap_ind, idx, covariance, scene_name = datas[i]
        overlap_ratio = src_overlap_ind.shape[0] / src_pcd.shape[0]

        if 0.1 < overlap_ratio:
            # save
            # np.save("3DLoMatch_test/info/idx%d.npy" % item, idx)
            # np.save("3DLoMatch_test/info/covariance%d.npy" % item, covariance)
            scene_names_lo.append(scene_name+"\n")
            item += 1

        if overlap_ratio <= 0.3:
            # np.save("3DZeroMatch_test/info/idx%d.npy" % iter_zero, idx)
            # np.save("3DZeroMatch_test/info/covariance%d.npy" % iter_zero, covariance)
            scene_names_zero.append(scene_name+"\n")
            iter_zero += 1

        print("\r%d / %d" % (i + 1, len(datas)), end="")

    with open("3DLoMatch_test/info/scene_names.txt", "w") as f:
        f.writelines(scene_names_lo)

    with open("3DZeroMatch_test/info/scene_names.txt", "w") as f:
        f.writelines(scene_names_zero)


if __name__ == '__main__':
    # datas = ThreeDMatchDataset(root="E:/indoor", mode="train")
    # save_train()
    save_val()
    # save_test_3DMatch()
    # save_lo_and_zero_match()
    # save_test_3DMatch_info()
    # save_lo_and_zero_match_info()

    ## check dataset
    # for i in range(100):
    #     src_pcd, tgt_pcd, T = np.load("3DMatch_test/src%d.npy" % i), np.load("3DMatch_test/tgt%d.npy" % i), np.load("3DMatch_test/T%d.npy" % i)
    #     src_pc, tgt_pc = to_o3d_pcd(src_pcd, [1, 0.706, 0]), to_o3d_pcd(tgt_pcd, [0, 0.651, 0.929])
    #     o3d.estimate_normals(src_pc)
    #     o3d.estimate_normals(tgt_pc)
    #
    #     o3d.draw_geometries([src_pc, tgt_pc], window_name="reg", width=1000, height=800)
    #     o3d.draw_geometries([deepcopy(src_pc).transform(T), tgt_pc], window_name="reg", width=1000, height=800)