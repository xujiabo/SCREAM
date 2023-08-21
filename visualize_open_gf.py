import numpy as np
import open3d as o3d
import torch
from models.pointnet import DEMTransformer
from datasets.open_gf import OpenGFTest
from torch.utils import data
from utils import processbar, to_o3d_pcd
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


def get_heapmap_from_high(high, max_high=None):
    if max_high is None:
        max_high = np.max(high).item()
    all_high = high / max_high
    all_high[all_high <= 0] = 0
    all_high[all_high >= 1] = 1

    heatmap = cv2.applyColorMap(np.uint8(255 * all_high), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float).reshape(-1, 3) / 255
    heatmap = heatmap[:, [2, 1, 0]]

    return heatmap


def visualize_DEM_generation():
    chamfer_loss = 0
    high_loss = 0
    processed = 0
    save_item = 1
    with torch.no_grad():
        for dsm, dem_coarse, dem, c in test_loader:
            dsm, dem_coarse, dem = dsm.to(device), dem_coarse.to(device), dem.to(device)
            dem_pred, imgs = net(dsm, dem_coarse, False)

            # visualize patch generation
            patch_dsm_pc = to_o3d_pcd(dsm[0], [3/255, 168/255, 158/255])
            patch_dem_coarse_pc = to_o3d_pcd(dem_coarse[0], [1, 0, 0])
            patch_dem_pc = to_o3d_pcd(dem[0])
            patch_dem_pred_pc = to_o3d_pcd(dem_pred[0])

            o3d.estimate_normals(patch_dsm_pc)
            o3d.estimate_normals(patch_dem_pred_pc)
            o3d.estimate_normals(patch_dem_pc)

            high_pred = (dsm[0, :, 2] - dem_pred[:, :, 2]).cpu().numpy()
            high_gt = (dsm[0, :, 2] - dem[:, :, 2]).cpu().numpy()

            patch_dem_pc.colors = o3d.Vector3dVector(get_heapmap_from_high(high_gt))
            patch_dem_pred_pc.colors = o3d.Vector3dVector(get_heapmap_from_high(high_pred))

            o3d.draw_geometries([patch_dsm_pc, patch_dem_coarse_pc], width=1000, height=800, window_name="dsm")
            o3d.draw_geometries([patch_dem_pred_pc], width=1000, height=800, window_name="dem pred")
            o3d.draw_geometries([patch_dem_pc], width=1000, height=800, window_name="dem gt")

            # print("save ? (y/n)")
            # op = input()
            # if op == "y":
            #     np.save("./experiments/DSM/dsm%d.npy" % save_item, np.asarray(patch_dsm_pc.points))
            #     np.save("./experiments/DSM/dem_coarse%d.npy" % save_item, np.asarray(patch_dem_coarse_pc.points))
            #     np.save("./experiments/DSM/dem_pred%d.npy" % save_item, np.asarray(patch_dem_pred_pc.points))
            #     np.save("./experiments/DSM/dem%d.npy" % save_item, np.asarray(patch_dem_pc.points))
            #     save_item += 1

            # end patch visualize

            point_loss = net.loss(dem_pred, dem)

            processed += 1
            chamfer_loss += point_loss.item()

            h_loss = torch.abs(dem_pred[0, :, 2].detach() - dem[0, :, 2].detach()).mean(dim=0).item()
            high_loss += h_loss

            print("\r测试进度：%s  chamfer loss: %.5f  high_loss: %.5f" % (
                processbar(processed, len(test_set)), chamfer_loss / processed, high_loss / processed
            ), end="")
        chamfer_loss = chamfer_loss / len(test_set)
        high_loss = high_loss / len(test_set)
        print("\ntest finished ! chamfer loss: %.5f  high loss: %.5f" % (chamfer_loss, high_loss))
    return chamfer_loss, high_loss


def save_patch():
    test_loader = data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    save_item = 1
    max_high = 0
    with torch.no_grad():
        for dsm, dem_coarse, dem, c in test_loader:
            dsm, dem_coarse, dem = dsm.to(device), dem_coarse.to(device), dem.to(device)
            dem_pred, imgs = net(dsm, dem_coarse, False)

            high_gt = (dsm[0, :, 2] - dem[:, :, 2]).cpu().numpy()
            max_high = max(max_high, np.max(high_gt).item())

            dem_pred = dem_pred[0].cpu().numpy() * 50 + c[0].cpu().numpy()
            np.save("./OpenGG_Patch_DEM/%d.npy" % save_item, dem_pred)
            save_item += 1
            print("\r%d / %d" % (save_item-1, len(test_loader.dataset)), end="")
    print("\n%.7f" % max_high)


def visualize_patches():
    test_loader = data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    item = 1
    max_high = 0.8338001 * 50
    dem_preds = []
    for dsm, dem_coarse, dem, c in test_loader:
        dsm = dsm * 50 + c
        dem_pred = torch.Tensor(np.load("OpenGG_Patch_DEM/%d.npy" % item)).unsqueeze(0)
        dem = dem * 50 + c

        high_pred = (dsm[0, :, 2] - dem_pred[:, :, 2]).cpu().numpy()
        high_gt = (dsm[0, :, 2] - dem[:, :, 2]).cpu().numpy()

        # patch_dem_pc = to_o3d_pcd(dem[0])
        patch_dem_pred_pc = to_o3d_pcd(dem_pred[0])

        # patch_dem_pc.colors = o3d.Vector3dVector(get_heapmap_from_high(high_gt, max_high))
        patch_dem_pred_pc.colors = o3d.Vector3dVector(get_heapmap_from_high(high_pred, max_high))
        dem_preds.append(patch_dem_pred_pc)

        item += 1
        print("\r%d / %d" % (item - 1, len(test_loader.dataset)), end="")
    o3d.draw_geometries(dem_preds, width=2000, height=1600, window_name="dem pred")


if __name__ == '__main__':
    visualize_DEM_generation()
    # save_patch()
    # visualize_patches()