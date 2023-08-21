import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from utils import to_o3d_pcd


scale_factor = 50


class OpenGFTrain(Dataset):
    def __init__(self):
        self.dem_coarse_resolution = 20

    def __len__(self):
        return 10693

    def __getitem__(self, item):
        dsm_dem = np.load("./OpenGF_train/%d.npy" % (item + 1))
        dsm, dem = dsm_dem[:, :3], dsm_dem[:, 3:]

        dem_coarse = o3d.voxel_down_sample(to_o3d_pcd(dem), self.dem_coarse_resolution)
        dem_coarse = np.asarray(dem_coarse.points)

        dsm, dem_coarse, dem = dsm / scale_factor, dem_coarse / scale_factor, dem / scale_factor
        dsm, dem_coarse, dem = torch.Tensor(dsm), torch.Tensor(dem_coarse), torch.Tensor(dem)

        return dsm, dem_coarse, dem


class OpenGFVal(Dataset):
    def __init__(self):
        self.dem_coarse_resolution = 20

    def __len__(self):
        return 75

    def __getitem__(self, item):
        dsm_dem = np.load("./OpenGF_val/%d.npy" % (item + 1))
        dsm, dem = dsm_dem[:, :3], dsm_dem[:, 3:]

        dem_coarse = o3d.voxel_down_sample(to_o3d_pcd(dem), self.dem_coarse_resolution)
        dem_coarse = np.asarray(dem_coarse.points)

        dsm, dem_coarse, dem = dsm / scale_factor, dem_coarse / scale_factor, dem / scale_factor
        dsm, dem_coarse, dem = torch.Tensor(dsm), torch.Tensor(dem_coarse), torch.Tensor(dem)

        return dsm, dem_coarse, dem


class OpenGFTest(Dataset):
    def __init__(self):
        self.dem_coarse_resolution = 20

    def __len__(self):
        return 650

    def __getitem__(self, item):
        dsm_dem = np.load("./OpenGF_test/%d.npy" % (item + 1))
        dsm, dem = dsm_dem[:, :3], dsm_dem[:, 3:]

        dem_coarse = o3d.voxel_down_sample(to_o3d_pcd(dem), self.dem_coarse_resolution)
        dem_coarse = np.asarray(dem_coarse.points)

        center = np.load("./OpenGF_test/centers/%d.npy" % (item + 1))

        dsm, dem_coarse, dem = dsm / scale_factor, dem_coarse / scale_factor, dem / scale_factor
        dsm, dem_coarse, dem = torch.Tensor(dsm), torch.Tensor(dem_coarse), torch.Tensor(dem)

        return dsm, dem_coarse, dem, center


if __name__ == '__main__':
    opengf = OpenGFTest()
    for i in range(len(opengf)):
        dsm, dem_coarse, dem, _ = opengf[i]

        dsm_pc = to_o3d_pcd(dsm, [3/255, 168/255, 158/255])
        dem_coarse_pc = to_o3d_pcd(dem_coarse, [1, 0, 0])
        dem_pc = to_o3d_pcd(dem, [104/255, 131/255, 139/255])

        o3d.estimate_normals(dsm_pc)
        o3d.estimate_normals(dem_pc)

        o3d.draw_geometries([dsm_pc, dem_coarse_pc], width=1000, height=800, window_name="dsm dem_coarse")
        o3d.draw_geometries([dsm_pc, dem_pc], width=1000, height=800, window_name="dsm dem")