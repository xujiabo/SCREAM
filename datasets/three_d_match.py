import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import to_o3d_pcd, load_obj, get_correspondences
from lie.numpy.se3 import SE3
import random
from lie.numpy.utils import se3_init, se3_inv, se3_cat, se3_transform


def read_info_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 7
    for i in range(num_pairs):
        line_id = i * 7
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        info = []
        for j in range(1, 7):
            info.append(lines[line_id + j].split())
        info = np.array(info, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, covariance=info))
    return test_pairs


class ThreeDMatchDataset_PREDATOR(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, root, mode="train"):
        super(ThreeDMatchDataset_PREDATOR, self).__init__()
        train_info = "datasets/3DMatch/indoor/train_info.pkl"
        val_info = "datasets/3DMatch/indoor/val_info.pkl"
        test_3dmatch_info = "datasets/3DMatch/indoor/3DMatch.pkl"
        test_3dlomatch_info = "datasets/3DMatch/indoor/3DLoMatch.pkl"

        test_info = None
        if mode == "train":
            infos = load_obj(train_info)
        elif mode == "val":
            infos = load_obj(val_info)
        elif mode == "test 3DMatch":
            infos = load_obj(test_3dmatch_info)
            test_info = "datasets/3DMatch/info/3DMatch/%s/gt.info"
        else:
            infos = load_obj(test_3dlomatch_info)
            test_info = "datasets/3DMatch/info/3DLoMatch/%s/gt.info"

        self.infos = infos
        self.base_dir = root

        infos = {

        }
        if test_info is not None:
            _scene_name_to_abbr = {
                '7-scenes-redkitchen': 'Kitchen',
                'sun3d-home_at-home_at_scan1_2013_jan_1': 'Home_1',
                'sun3d-home_md-home_md_scan9_2012_sep_30': 'Home_2',
                'sun3d-hotel_uc-scan3': 'Hotel_1',
                'sun3d-hotel_umd-maryland_hotel1': 'Hotel_2',
                'sun3d-hotel_umd-maryland_hotel3': 'Hotel_3',
                'sun3d-mit_76_studyroom-76-1studyroom2': 'Study',
                'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 'MIT_Lab',
            }
            for k in _scene_name_to_abbr.keys():
                info = read_info_file(test_info % k)
                for i in range(len(info)):

                    infos["%s_%d,%d" % (k, info[i]["test_pair"][0], info[i]["test_pair"][1])] = info[i]["covariance"]
        self.covariance = infos
        self.mode = mode

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):
        # get transformation
        rot = self.infos['rot'][item]
        trans = self.infos['trans'][item]

        # get pointcloud
        src_path = os.path.join(self.base_dir, self.infos['src'][item])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][item])
        # print(src_path, tgt_path)
        src_idx = int(src_path.split('_')[-1].replace('.pth', ''))
        tgt_idx = int(tgt_path.split('_')[-1].replace('.pth', ''))
        # print(tgt_idx, src_idx)
        scene_name = src_path.split('/')[-2]
        # print(scene_name)
        # print(self.covariance["%s_%d,%d" % (scene_name, tgt_idx, src_idx)])

        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        src_pc, tgt_pc = to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd)

        corr = get_correspondences(
            src_pc, tgt_pc,
            np.concatenate([np.concatenate([rot.astype(np.float32), trans.astype(np.float32)], axis=1), np.array([[0, 0, 0, 1]])], axis=0),
            0.03
        )
        src_overlap_ind = torch.LongTensor(list(set(corr[:, 0].tolist()))).numpy()

        if trans.ndim == 1:
            trans = trans[:, None]

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        if self.mode == "train" or self.mode == "val":
            covariance = None
        else:
            covariance = self.covariance["%s_%d,%d" % (scene_name, tgt_idx, src_idx)]

        return src_pcd, tgt_pcd, rot, trans, src_overlap_ind, np.array([tgt_idx, src_idx]).astype(np.int), covariance, scene_name


def augment(src_pcd, tgt_pcd, T):
    perturb = SE3.sample_small(std=0.1).as_matrix()
    perturb_source = random.random() > 0.5  # whether to perturb source or target

    # Center perturbation around the point centroid (otherwise there's a large
    # induced translation as rotation is centered around origin)
    centroid = np.mean(src_pcd, axis=0).reshape(3, 1) if perturb_source else np.mean(tgt_pcd, axis=0).reshape(3, 1)
    center_transform = se3_init(rot=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), trans=-centroid)
    perturb = se3_cat(se3_cat(se3_inv(center_transform), perturb), center_transform)

    if perturb_source:
        T = se3_cat(T, se3_inv(perturb))
        src_pcd = se3_transform(perturb, src_pcd)

    else:
        T = se3_cat(perturb, T)
        tgt_pcd = se3_transform(perturb, tgt_pcd)

    noise = np.random.randn(src_pcd.shape[0], 3) * 0.003
    src_pcd = src_pcd + noise

    noise = np.random.randn(tgt_pcd.shape[0], 3) * 0.003
    tgt_pcd = tgt_pcd + noise

    return src_pcd, tgt_pcd, T


scene_name_to_idx = {
    '7-scenes-redkitchen': 0,
    'sun3d-home_at-home_at_scan1_2013_jan_1': 1,
    'sun3d-home_md-home_md_scan9_2012_sep_30': 2,
    'sun3d-hotel_uc-scan3': 3,
    'sun3d-hotel_umd-maryland_hotel1': 4,
    'sun3d-hotel_umd-maryland_hotel3': 5,
    'sun3d-mit_76_studyroom-76-1studyroom2': 6,
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 7,
}


class ThreeDMatchTrain(Dataset):
    def __init__(self):
        super(ThreeDMatchTrain, self).__init__()

    def __len__(self):
        return 27730

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("3DMatch_train/src%d.npy" % item), np.load("3DMatch_train/tgt%d.npy" % item), np.load("3DMatch_train/T%d.npy" % item)

        # rotate the point cloud
        src_pcd, tgt_pcd, T = augment(src_pcd, tgt_pcd, T)

        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        c = np.mean(registrated, axis=0)
        registrated = registrated - c.reshape(1, 3)
        s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s


class ThreeDMatchVal(Dataset):
    def __init__(self):
        super(ThreeDMatchVal, self).__init__()

    def __len__(self):
        return 1749

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("3DMatch_val/src%d.npy" % item), np.load("3DMatch_val/tgt%d.npy" % item), np.load("3DMatch_val/T%d.npy" % item)

        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        c = np.mean(registrated, axis=0)
        registrated = registrated - c.reshape(1, 3)
        s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s


class ThreeDMatchTest(Dataset):
    def __init__(self):
        super(ThreeDMatchTest, self).__init__()
        with open("3DMatch_test/info/scene_names.txt", "r") as f:
            self.scene_names = f.readlines()

    def __len__(self):
        return 1253

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("3DMatch_test/src%d.npy" % item), np.load("3DMatch_test/tgt%d.npy" % item), np.load("3DMatch_test/T%d.npy" % item)
        idx, covariance = np.load("3DMatch_test/info/idx%d.npy" % item), np.load("3DMatch_test/info/covariance%d.npy" % item)
        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        c = np.mean(registrated, axis=0)
        registrated = registrated - c.reshape(1, 3)
        s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.LongTensor(idx), torch.Tensor(covariance), torch.Tensor(c), scene_name_to_idx[self.scene_names[item].replace("\n", "")]


class ThreeDLoMatchTest(Dataset):
    def __init__(self):
        super(ThreeDLoMatchTest, self).__init__()
        with open("3DLoMatch_test/info/scene_names.txt", "r") as f:
            self.scene_names = f.readlines()

    def __len__(self):
        return 1518

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("3DLoMatch_test/src%d.npy" % item), np.load("3DLoMatch_test/tgt%d.npy" % item), np.load("3DLoMatch_test/T%d.npy" % item)
        idx, covariance = np.load("3DLoMatch_test/info/idx%d.npy" % item), np.load("3DLoMatch_test/info/covariance%d.npy" % item)
        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        c = np.mean(registrated, axis=0)
        registrated = registrated - c.reshape(1, 3)
        s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.LongTensor(idx), torch.Tensor(covariance), torch.Tensor(c), scene_name_to_idx[self.scene_names[item].replace("\n", "")]


class ThreeDZeroMatchTest(Dataset):
    def __init__(self):
        super(ThreeDZeroMatchTest, self).__init__()
        with open("3DZeroMatch_test/info/scene_names.txt", "r") as f:
            self.scene_names = f.readlines()

    def __len__(self):
        return 1389

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("3DZeroMatch_test/src%d.npy" % item), np.load("3DZeroMatch_test/tgt%d.npy" % item), np.load("3DZeroMatch_test/T%d.npy" % item)
        idx, covariance = np.load("3DZeroMatch_test/info/idx%d.npy" % item), np.load("3DZeroMatch_test/info/covariance%d.npy" % item)
        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        c = np.mean(registrated, axis=0)
        registrated = registrated - c.reshape(1, 3)
        s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.LongTensor(idx), torch.Tensor(covariance), torch.Tensor(c), scene_name_to_idx[self.scene_names[item].replace("\n", "")]
