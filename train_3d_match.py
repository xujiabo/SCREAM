import numpy as np
import open3d as o3d
import cv2
import torch
from models.pointnet import PointTransformer
from datasets.three_d_match import ThreeDMatchTrain, ThreeDMatchVal
from torch.utils import data
from loss import AdversarialLoss
from utils import processbar, transformation_error, square_distance, deep_to_img
from copy import deepcopy
from utils import to_o3d_pcd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset

train_set = ThreeDMatchTrain()
val_set = ThreeDMatchVal()

train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

# network
use_GAN = False
if use_GAN:
    gan_loss = AdversarialLoss()
    gan_loss.to(device)
net = PointTransformer(d_model=256)
net.to(device)

generator_save_path = "params/point-generator.pth"
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


def look():

    net.load_state_dict(torch.load(generator_save_path))
    net.eval()
    with torch.no_grad():
        rre, rte = 0, 0
        for src_pcd, tgt_pcd, rot, trans, s in train_loader:
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            src_pred, imgs, transform = net(src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s, use_GAN, True)

            src_pc = to_o3d_pcd(src_pcd[0], [1, 0.706, 0])
            tgt_pc = to_o3d_pcd(tgt_pcd[0], [0, 0.651, 0.929])
            o3d.estimate_normals(src_pc)
            o3d.estimate_normals(tgt_pc)

            T = torch.cat([torch.cat([rot[0], trans[0]], dim=1), torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            re, te = 0, 0
            if transform is not None:
                re, te = transformation_error(transform, T)
                # final_trans1 = post_refinement(initial_trans=transform[None], src_kpts=src_pcd.detach()[0][None],
                #                                tgt_kpts=src_pred.detach()[0][None], iters=20)
                final_trans1 = o3d.registration_icp(src_pc, tgt_pc, max_correspondence_distance=0.1*s, init=transform.cpu().numpy())
                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)
                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    print("refine")
                    transform = final_trans1[0]

                    re, te = re1, te1

            print(re, te)

            transform = transform.cpu().numpy()
            # print(transform.shape)
            rre, rte = rre + re, rte + te

            src_pred_pc = to_o3d_pcd(src_pred[0], [1, 0.706, 0])
            o3d.estimate_normals(src_pred_pc)

            src_pred_gt = np.asarray(deepcopy(src_pc).transform(transform).points)
            dis = np.linalg.norm(np.asarray(src_pred_pc.points) / s - src_pred_gt / s, axis=1)
            incorrect_ind = (dis > 0.15)
            np.asarray(src_pred_pc.colors)[incorrect_ind] = np.array([1, 0, 0])

            # o3d.draw_geometries([src_pc], window_name="src", width=1000, height=800)
            o3d.draw_geometries([src_pred_pc], window_name="src pred", width=1000, height=800)
            # o3d.draw_geometries([src_pred_pc, tgt_pc], window_name="src pred", width=1000, height=800)
            o3d.draw_geometries([deepcopy(src_pc).transform(transform), tgt_pc], width=1000, height=800, window_name="reg")
            o3d.draw_geometries([src_pc.transform(T.cpu().numpy()), tgt_pc], width=1000, height=800, window_name="reg gt")


def evaluate():
    processed = 0
    rre, rte = 0, 0
    point_trans_loss = 0
    get_transform = True
    net.eval()
    with torch.no_grad():
        for src_pcd, tgt_pcd, rot, trans, s in val_loader:
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            src_pred, imgs, transform = net(
                src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s,
                False, get_transform,
                (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
            )

            T = torch.cat([torch.cat([rot[0], trans[0]], dim=1), torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            re, te = 0, 0
            if transform is not None:
                re, te = transformation_error(transform, T)

                src_pc = to_o3d_pcd(src_pcd[0], [1, 0.706, 0])
                tgt_pc = to_o3d_pcd(tgt_pcd[0], [0, 0.651, 0.929])

                final_trans1 = o3d.registration_icp(
                    src_pc, tgt_pc,
                    max_correspondence_distance=0.1 * s,
                    init=transform.cpu().numpy()
                )
                final_trans1 = torch.Tensor(final_trans1.transformation).unsqueeze(0).to(device)
                re1, te1 = transformation_error(final_trans1[0], T)
                if re1 <= re and te1 <= te:
                    transform = final_trans1[0]
                    re, te = re1, te1

            rre, rte = rre + re, rte + te

            point_loss = net.loss(src_pred, src_pcd, rot, trans)
            point_trans_loss += point_loss.item()

            processed += 1
            print("\r进度：%s  point trans loss: %.5f  re: %.5f  te: %.5f" % (
                processbar(processed, len(val_set)), point_loss.item(), re, te
            ), end="")
    rre, rte = rre / len(val_set), rte / len(val_set)
    point_trans_loss = point_trans_loss / len(val_set)
    print("\ntest finish  loss: %.5f  rre: %.5f  rte: %.5f" % (point_trans_loss, rre, rte))
    return point_trans_loss, rre, rte


def train():
    min_point_loss = 1e8
    max_rr = 0
    for epoch in range(1, epochs):
        net.train()

        point_trans_loss = 0
        processed = 0
        rre, rte = 0, 0
        iter = 0
        get_transform = True
        # get_transform = (epoch > 7)
        for src_pcd, tgt_pcd, rot, trans, s in train_loader:
            s = s[0].item()
            src_pcd, tgt_pcd, rot, trans = src_pcd.to(device), tgt_pcd.to(device), rot.to(device), trans.to(device)
            src_pred, imgs, transform = net(
                src_pcd, tgt_pcd, trans.permute([0, 2, 1]), s,
                use_GAN, get_transform,
                (torch.matmul(rot, src_pcd.permute([0, 2, 1])) + trans).permute([0, 2, 1])
            )

            T = torch.cat([torch.cat([rot[0], trans[0]], dim=1), torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0)
            re, te = 0, 0
            if transform is not None:
                re, te = transformation_error(transform, T)
            rre, rte = rre + re, rte + te

            g_loss = torch.zeros(1).cuda()
            point_loss = net.loss(src_pred, src_pcd, rot, trans)

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
                src_real = (torch.matmul(rot[0].detach(), src_pcd[0].detach().t()) + trans[0].detach()).t()
                real = net.generator(src_real, tgt_pcd[0].detach())

                d_loss = gan_loss(imgs, real, 1)

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

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
        point_trans_loss, rre, rte = evaluate()

        if min_point_loss > point_trans_loss:
            min_point_loss = point_trans_loss
            print("save...")
            torch.save(net.state_dict(), generator_save_path)
            print("save finished !!!")

        if epoch % lr_update_epoch == 0:
            update_lr(optimizer_G, learning_rate_decay_gamma)


if __name__ == '__main__':
    train()
    # evaluate()
    # look()