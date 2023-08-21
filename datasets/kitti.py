import numpy as np
import open3d
import torch
import os, glob, random, copy
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from utils import to_o3d_pcd

from lie.numpy.se3 import SE3
from lie.numpy.utils import se3_init, se3_inv, se3_cat, se3_transform


# copy from PREDATOR
class KITTI_PREDATOR(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': [0, 1, 2, 3, 4, 5],
        'val': [6, 7],
        'test': [8, 9, 10]
    }

    def __init__(self, root, mode="train", data_augmentation=False):
        super(KITTI_PREDATOR, self).__init__()
        self.root = root + '/dataset'
        self.icp_path = root + '/icp'
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = 0.3
        self.matching_search_voxel_size = 0.45
        self.data_augmentation = data_augmentation
        self.augment_noise = 0.01
        self.IS_ODOMETRY = True
        self.max_corr = 512
        self.augment_shift_range = 2.0
        self.augment_scale_max = 1.2
        self.augment_scale_min = 0.8

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(mode)
        self.split = mode

    def prepare_kitti_ply(self, split):
        assert split in ['train', 'val', 'test']

        subset_names = self.DATA_FILES[split]
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            # print(fnames)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split=='test':
            self.files.remove((8, 15, 58))
        print(f'Num_{split}: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                            @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                        open3d.registration.TransformationEstimationPointToPoint(),
                                                        open3d.registration.ICPConvergenceCriteria(max_iteration=50000))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]


        # refined pose is denoted as trans
        tsfm = M2
        rot = tsfm[:3,:3]
        trans = tsfm[:3,3][:,None]

        # voxelize the point clouds here
        pcd0 = to_o3d_pcd(xyz0)
        pcd1 = to_o3d_pcd(xyz1)
        pcd0 = open3d.voxel_down_sample(pcd0, self.voxel_size)
        pcd1 = open3d.voxel_down_sample(pcd1, self.voxel_size)
        # pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        # pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        src_pcd = np.array(pcd0.points)
        tgt_pcd = np.array(pcd1.points)

        # # Get matches
        # matching_inds = get_correspondences(pcd0, pcd1, tsfm, self.matching_search_voxel_size)
        # if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
        #     return self.__getitem__(np.random.choice(len(self.files),1)[0])

        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        # add data augmentation
        src_pcd_input = copy.deepcopy(src_pcd)
        tgt_pcd_input = copy.deepcopy(tgt_pcd)
        if self.data_augmentation:
            # add gaussian noise
            src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
            else:
                tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
            
            # scale the pcd
            scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            src_pcd_input = src_pcd_input * scale
            tgt_pcd_input = tgt_pcd_input * scale

            # shift the pcd
            shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

            src_pcd_input = src_pcd_input + shift_src
            tgt_pcd_input = tgt_pcd_input + shift_tgt

        return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)


def augment(src_pcd, tgt_pcd, T):
    perturb = SE3.sample_small(std=0.1).as_matrix()
    perturb_source = True  # only perturb source

    centroid = np.mean(src_pcd, axis=0).reshape(3, 1) if perturb_source else np.mean(tgt_pcd, axis=0).reshape(3, 1)
    center_transform = se3_init(rot=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), trans=-centroid)
    perturb = se3_cat(se3_cat(se3_inv(center_transform), perturb), center_transform)

    if perturb_source:
        T = se3_cat(T, se3_inv(perturb))
        src_pcd = se3_transform(perturb, src_pcd)

    else:
        T = se3_cat(perturb, T)
        tgt_pcd = se3_transform(perturb, tgt_pcd)

    # # only rotate z-axis [0, 45°]
    # euler_ab = np.random.rand(3) * np.pi / 4  # anglez, angley, anglex
    # euler_ab[1:] = 0
    # rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
    # src_pcd = np.dot(rot_ab, src_pcd.T).T
    #
    # rot_gt = np.dot(T[:3, :3], rot_ab.T)
    # T = np.concatenate([np.concatenate([rot_gt, T[:3, 3:]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    # don't add noise
    # noise = np.random.randn(src_pcd.shape[0], 3) * 0.003
    # src_pcd = src_pcd + noise
    #
    # noise = np.random.randn(tgt_pcd.shape[0], 3) * 0.003
    # tgt_pcd = tgt_pcd + noise

    return src_pcd, tgt_pcd, T


def norm_pc(pc):
    coor_max, coor_min = np.max(pc, axis=0), np.min(pc, axis=0)
    c = (coor_min + coor_max) / 2
    cross = max((coor_max[0]-coor_min[0]).item(), (coor_max[1]-coor_min[1]).item(), (coor_max[2]-coor_min[2]).item())
    s = 1 / (cross / 2)
    return c, s


class KITTI_Train(Dataset):
    def __init__(self):
        super(KITTI_Train, self).__init__()

    def __len__(self):
        return 1358

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("KITTI_train/src%d.npy" % item), np.load("KITTI_train/tgt%d.npy" % item), np.load("KITTI_train/T%d.npy" % item)

        # rotate the point cloud
        src_pcd, tgt_pcd, T = augment(src_pcd, tgt_pcd, T)
        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        # c = np.mean(registrated, axis=0)
        # registrated = registrated - c.reshape(1, 3)
        # s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()
        c, s = norm_pc(registrated)

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.Tensor(c)


class KITTI_Val(Dataset):
    def __init__(self):
        super(KITTI_Val, self).__init__()

    def __len__(self):
        return 180

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("KITTI_val/src%d.npy" % item), np.load("KITTI_val/tgt%d.npy" % item), np.load("KITTI_val/T%d.npy" % item)

        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        # c = np.mean(registrated, axis=0)
        # registrated = registrated - c.reshape(1, 3)
        # s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()
        c, s = norm_pc(registrated)

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.Tensor(c)


class KITTI_Test(Dataset):
    def __init__(self):
        super(KITTI_Test, self).__init__()

    def __len__(self):
        return 554

    def __getitem__(self, item):
        src_pcd, tgt_pcd, T = np.load("KITTI_test/src%d.npy" % item), np.load("KITTI_test/tgt%d.npy" % item), np.load("KITTI_test/T%d.npy" % item)

        rot, trans = T[:3, :3], T[:3, 3:]

        registrated = np.concatenate([(rot.dot(src_pcd.T) + trans).T, tgt_pcd], axis=0)
        # c = np.mean(registrated, axis=0)
        # registrated = registrated - c.reshape(1, 3)
        # s = 1 / np.max(np.linalg.norm(registrated, axis=1)).item()
        c, s = norm_pc(registrated)

        src_pcd = s * (src_pcd - c)
        tgt_pcd = s * (tgt_pcd - c)
        trans = s * (trans - c.reshape(3, 1) + rot.dot(c.reshape(3, 1)))

        return torch.Tensor(src_pcd), torch.Tensor(tgt_pcd), torch.Tensor(rot), torch.Tensor(trans), s, torch.Tensor(c)


if __name__ == '__main__':
    pass