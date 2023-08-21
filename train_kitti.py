import numpy as np
import open3d as o3d
import cv2
import torch
from models.pointnet import PointTransformer
from datasets.kitti import KITTI_Train, KITTI_Val
from torch.utils import data
from loss import AdversarialLoss
from utils import processbar, transformation_error, square_distance
from utils import to_o3d_pcd, rigid_transform_3d, deep_to_img
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset

train_set = KITTI_Train()
val_set = KITTI_Val()
val_set.au = False

train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

# network
use_GAN = False
if use_GAN:
    gan_loss = AdversarialLoss()
    gan_loss.to(device)
net = PointTransformer(d_model=256, self_layer_num=6, cross_layer_num=6)
net.to(device)

generator_save_path = "params/kitti-generator.pth"
discriminator_save_path = "params/discriminator.pth"

# net.load_state_dict(torch.load(generator_save_path))


# lr_g, lr_d = 0.0003, 0.0001
lr_g, lr_d = 0.00032, 0.0001
min_learning_rate = 0.00001
learning_rate_decay_gamma = 0.5
lr_update_epoch = 10
# 优化器
optimizer_G = torch.optim.Adam(params=net.parameters(), lr=lr_g)
# optimizer_G = torch.optim.SGD(params=net.parameters(), lr=lr_g)
if use_GAN:
    optimizer_D = torch.optim.Adam(params=gan_loss.parameters(), lr=lr_d, betas=(0.5, 0.999))
    net.generator.rho = 48
    net.generator.eulers = [np.array([0, np.pi, 0])]
epochs = 120
save_img_iter = 1000

scaler = GradScaler()


def update_lr(optimizer, gamma=0.5):
    global lr_g
    lr_g = max(lr_g*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_g
    print("lr update finished  cur lr: %.5f" % lr_g)


def evaluate(dis_thresh=0.9, icp_thresh=0.9, corr="tgt"):
    processed = 0
    rre, rte, success_rate = 0, 0, 0
    success_rre, success_rte = 0, 0
    point_trans_loss = 0
    net.eval()
    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s, c in val_loader:
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans, c = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device), c.to(device)
            with autocast():
                src_pred, imgs, transform = net(
                    src_pcd, tgt_pcd, -torch.matmul(rot.permute([0, 2, 1]), trans).permute([0, 2, 1]), s,
                    False, False,
                    (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
                )

                point_loss = net.loss(src_pred, src_pcd, rot, trans)
                point_trans_loss += point_loss.item()

            src_pred = src_pred.float()
            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1),
                           torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
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
                    max_correspondence_distance=icp_thresh,
                    init=transform.cpu().numpy(),
                    estimation_method=o3d.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=1000)
                )

                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)

                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    transform = final_trans1[0]
                    re, te = re1, te1
            # success
            if re <= 5. and te <= 2.:
                success_rate += 1
                success_rre = success_rre + re
                success_rte = success_rte + te

            rre, rte = rre + re, rte + te

            processed += 1
            print("\r进度：%s  point trans loss: %.5f  re: %.5f  te: %.5f  success rre: %.5f  success rte: %.5f  success rate: %.5f" % (
                processbar(processed, len(val_set)), point_loss.item(), re, te, success_rre/(success_rate if success_rate > 0 else 1), success_rte/(success_rate if success_rate > 0 else 1), success_rate/processed
            ), end="")
    rre, rte = rre / len(val_set), rte / len(val_set)
    point_trans_loss = point_trans_loss / len(val_set)
    success_rre, success_rte = success_rre/(success_rate if success_rate > 0 else 1), success_rte/(success_rate if success_rate > 0 else 1)
    success_rate = success_rate / len(val_set)
    print("\ntest finish  loss: %.5f  rre: %.5f  rte: %.5f  success rate: %.5f" % (point_trans_loss, success_rre, success_rte, success_rate))
    return success_rre, success_rte, success_rate


def train():
    global lr_update_epoch
    max_rr = 0
    for epoch in range(1, epochs):
        net.train()

        point_trans_loss = 0
        processed = 0
        rre, rte = 0, 0
        iter = 0
        get_transform = False

        for src_pcd, tgt_pcd, rot, trans, s, c in train_loader:
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans, c = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device), c.to(device)
            # print(src_pcd.shape, tgt_pcd.shape)
            with autocast():
                src_pred, imgs, transform = net(
                    src_pcd, tgt_pcd, -torch.matmul(rot.permute([0, 2, 1]), trans).permute([0, 2, 1]), s,
                    use_GAN, get_transform,
                    (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
                )

                g_loss = torch.zeros(1).cuda()
                point_loss = net.loss(src_pred, src_pcd, rot, trans)
                # t = trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))
                # point_loss = torch.abs((torch.matmul(rot[0], (src_pcd[0] / s + c).t()) + t).t() - (src_pred[0] / s + c)).sum(dim=1).mean(dim=0)

                if use_GAN:
                    # update generator
                    g_loss = gan_loss(imgs, None, 0)
                    point_loss = point_loss + 0.1 * g_loss

            T = torch.cat([torch.cat([rot[0], trans[0] / s + c.view(3, 1) - torch.matmul(rot[0], c.view(3, 1))], dim=1),
                           torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            # transformation
            src_pred = src_pred.float().detach()
            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_pred / s, tgt_pcd / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < 0.9)
            tgt_ind = src_pred_2_tgt_ind[valid_ind]
            transform = rigid_transform_3d(src_pcd[:, valid_ind] / s + c, tgt_pcd[:, tgt_ind] / s + c)[0]
            re, te = transformation_error(transform, T)
            rre, rte = rre + re, rte + te

            # optimizer_G.zero_grad()
            # point_loss.backward()
            # optimizer_G.step()
            optimizer_G.zero_grad()
            scaler.scale(point_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            d_loss = torch.zeros(1).cuda()
            if use_GAN:
                # update discriminator
                src_real = (torch.matmul(rot[0].detach(), src_pcd[0].detach().t()) + trans[0].detach()).t()
                real = net.generator(src_real, tgt_pcd[0].detach())
                with autocast():
                    d_loss = gan_loss(imgs, real, 1)

                # optimizer_D.zero_grad()
                # d_loss.backward()
                # optimizer_D.step()
                optimizer_D.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()

            processed += 1
            iter += 1
            point_trans_loss += point_loss.item()

            if use_GAN and iter % save_img_iter == 0:
                imgs = imgs.detach().permute([0, 2, 3, 1])
                img_cv = np.concatenate([deep_to_img(imgs[j]) for j in range(imgs.shape[0])], axis=1)
                cv2.imwrite("./sampled_imgs/epoch%d_iter%d.jpg" % (epoch, iter), img_cv)

            print("\r进度：%s  point trans loss: %.5f  g loss: %.5f  d_loss: %.5f  re: %.5f  te: %.5f" % (
                processbar(processed, len(train_set)), point_loss.item(), g_loss.item(), d_loss.item(), re, te
            ), end="")
        point_trans_loss = point_trans_loss / len(train_set)
        rre, rte = rre / len(train_set), rte / len(train_set)
        print("\nepoch: %d  loss: %.5f  rre: %.5f  rte: %.5f" % (epoch, point_trans_loss, rre, rte))

        net.eval()
        rre, rte, rr = evaluate(1.5, 1.)

        if max_rr < rr:
            max_rr = rr
            print("save...")
            torch.save(net.state_dict(), generator_save_path)
            print("save finished !!!")

        if epoch % lr_update_epoch == 0:
            update_lr(optimizer_G, learning_rate_decay_gamma)
            if lr_update_epoch == 10:
                lr_update_epoch = 30


if __name__ == '__main__':
    train()
    # evaluate(1.5, 1.)