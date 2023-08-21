import numpy as np
import open3d as o3d
import torch
from torch import nn
from models.pointnet import DEMTransformer
from datasets.open_gf import OpenGFTest
from torch.utils import data
from utils import processbar, to_o3d_pcd, square_distance
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
test_set = OpenGFTest()
test_loader = data.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

generator_save_path = "params/dem-generator.pth"

net = DEMTransformer(d_model=256)
net.to(device)
net.load_state_dict(torch.load(generator_save_path))
net.eval()


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, f, f_):
        if f.shape[1] == 0:
            return 0
        try:
            dis = square_distance(f, f_)
            # dis = torch.sqrt(dis)
            # f2f_: patch_num x point_num   f_2f: patch_num x M
            f2f_, f_2f = dis.min(dim=2)[0], dis.min(dim=1)[0]
            # d = torch.stack([f2f_.mean(dim=1), f_2f.mean(dim=1)], dim=0).max(dim=0)[0]
            d = f2f_.mean(dim=1) + f_2f.mean(dim=1)
        except:
            print(f.shape, f_.shape)
        return d.mean()


chamfer_fn = ChamferDistance()


def evaluate_DEM_generation():
    chamfer_loss = 0
    high_loss_mae, high_loss_mse = 0, 0
    processed = 0
    scale_factor = 1000
    with torch.no_grad():
        for dsm, dem_coarse, dem, c in test_loader:
            dsm, dem_coarse, dem = dsm.to(device), dem_coarse.to(device), dem.to(device)
            dem_pred, imgs = net(dsm, dem_coarse, False)

            point_loss = chamfer_fn(dem_pred, dem) * scale_factor

            processed += 1
            chamfer_loss += point_loss.item()

            h_loss = torch.abs(dem_pred[0, :, 2].detach() - dem[0, :, 2].detach()).mean(dim=0).item()
            high_loss_mae += h_loss * scale_factor

            h_loss = ((dem_pred[0, :, 2].detach() - dem[0, :, 2].detach()) ** 2).mean(dim=0).item()
            high_loss_mse += h_loss * scale_factor

            print("\r测试进度：%s  chamfer loss: %.5f  high_loss_mae: %.5f  high_loss_mse: %.5f" % (
                processbar(processed, len(test_set)), chamfer_loss / processed, high_loss_mae / processed, high_loss_mse / processed
            ), end="")
        chamfer_loss = chamfer_loss / len(test_set)
        high_loss_mae = high_loss_mae / len(test_set)
        high_loss_mse = high_loss_mse / len(test_set)
        print("\ntest finished ! chamfer loss: %.5f  high loss mae: %.5f  hige_loss_mse: %.5f" % (chamfer_loss, high_loss_mae, high_loss_mse))
    return chamfer_loss, high_loss_mae, high_loss_mse


if __name__ == '__main__':
    evaluate_DEM_generation()