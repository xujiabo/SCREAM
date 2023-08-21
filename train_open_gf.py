import numpy as np
import open3d as o3d
import cv2
import torch
from models.pointnet import DEMTransformer
from datasets.open_gf import OpenGFTrain, OpenGFVal
from torch.utils import data
from loss import AdversarialLoss
from utils import processbar, transformation_error, square_distance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset

train_set = OpenGFTrain()
val_set = OpenGFVal()

train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

# network
use_GAN = False
if use_GAN:
    gan_loss = AdversarialLoss()
    gan_loss.to(device)
net = DEMTransformer(d_model=256)
net.to(device)

generator_save_path = "params/dem-generator.pth"
discriminator_save_path = "params/discriminator.pth"


lr_g, lr_d = 0.0002, 0.0001
min_learning_rate = 0.00001
learning_rate_decay_gamma = 0.5
lr_update_epoch = 15
# 优化器
optimizer_G = torch.optim.Adam(params=net.parameters(), lr=lr_g)
if use_GAN:
    optimizer_D = torch.optim.Adam(params=gan_loss.parameters(), lr=lr_d, betas=(0.5, 0.999))
epochs = 45
save_img_iter = 1000


def update_lr(optimizer, gamma=0.5):
    global lr_g
    lr_g = max(lr_g*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_g
    print("lr update finished  cur lr: %.5f" % lr_g)


def evaluate():
    chamfer_loss = 0
    high_loss = 0
    processed = 0
    # get_transform = (epoch > 7)
    with torch.no_grad():
        for dsm, dem_coarse, dem in val_loader:
            dsm, dem_coarse, dem = dsm.to(device), dem_coarse.to(device), dem.to(device)
            dem_pred, imgs = net(dsm, dem_coarse, use_GAN)

            point_loss = net.loss(dem_pred, dem)

            processed += 1
            chamfer_loss += point_loss.item()

            h_loss = torch.abs(dem_pred[0, :, 2].detach() - dem[0, :, 2].detach()).mean(dim=0).item()
            high_loss += h_loss

            print("\r测试进度：%s  chamfer loss: %.5f  high_loss: %.5f" % (
                processbar(processed, len(val_set)), chamfer_loss / processed, high_loss / processed
            ), end="")
        chamfer_loss = chamfer_loss / len(val_set)
        high_loss = high_loss / len(val_set)
        print("\ntest finished ! chamfer loss: %.5f  high loss: %.5f" % (chamfer_loss, high_loss))
    return chamfer_loss, high_loss


def train():
    min_cd = 1e8
    for epoch in range(1, epochs):
        net.train()

        chamfer_loss = 0
        high_loss = 0
        processed = 0
        iter = 0

        for dsm, dem_coarse, dem in train_loader:
            dsm, dem_coarse, dem = dsm.to(device), dem_coarse.to(device), dem.to(device)
            dem_pred, imgs = net(dsm, dem_coarse, use_GAN)

            point_loss = net.loss(dem_pred, dem)

            g_loss = torch.zeros(1).cuda()
            if use_GAN:
                # update generator
                g_loss = gan_loss(imgs, None, 0)
                point_loss = point_loss + 0.1 * g_loss

            optimizer_G.zero_grad()
            point_loss.backward()
            optimizer_G.step()

            d_loss = torch.zeros(1).cuda()
            if use_GAN:
                # update discriminator
                dem_real = dem[0]
                real = net.generator(dem_real, dem_coarse[0].detach())

                d_loss = gan_loss(imgs, real, 1)

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            processed += 1
            iter += 1
            chamfer_loss += point_loss.item()

            h_loss = torch.abs(dem_pred[0, :, 2].detach() - dem[0, :, 2].detach()).mean(dim=0).item()
            high_loss += h_loss

            if use_GAN and iter % save_img_iter == 0:
                img_cv = (imgs.detach() * 0.5 + 0.5).permute([0, 2, 3, 1])
                img_cv = (img_cv * 255).cpu().numpy().astype(np.uint8)
                img_cv = np.concatenate([img_cv[j] for j in range(img_cv.shape[0])], axis=1)
                cv2.imwrite("./sampled_imgs/epoch%d_iter%d.jpg" % (epoch, iter), img_cv)

            print("\r进度：%s  chamfer loss: %.5f  g loss: %.5f  d_loss: %.5f  high_loss: %.5f" % (
                processbar(processed, len(train_set)), point_loss.item(), g_loss.item(), d_loss.item(), h_loss
            ), end="")
        chamfer_loss = chamfer_loss / len(train_set)
        high_loss = high_loss / len(train_set)
        print("\nepoch: %d  chamfer loss: %.5f  high loss: %.5f" % (epoch, chamfer_loss, high_loss))

        net.eval()
        cd, h = evaluate()
        if min_cd > cd:
            min_cd = cd
            print("save...")
            torch.save(net.state_dict(), generator_save_path)
            print("save finished !!!")

        if epoch % lr_update_epoch == 0:
            update_lr(optimizer_G, learning_rate_decay_gamma)


if __name__ == '__main__':
    train()
    # evaluate()