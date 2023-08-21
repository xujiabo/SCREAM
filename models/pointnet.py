import torch
from torch import nn
from models.transformer import PositionEmbeddingCoordsSine, MHAttention, CrossAttention
from models.render import RegistrationRender
from utils import rigid_transform_3d, square_distance


class PointTransformer(nn.Module):
    def __init__(self, d_model=256, self_layer_num=6, cross_layer_num=6):
        super(PointTransformer, self).__init__()
        self.embedding = nn.Conv1d(3, d_model, kernel_size=1, stride=1)
        self.pre_norm = nn.LayerNorm(d_model)

        self.pe_func = PositionEmbeddingCoordsSine(n_dim=3, d_model=d_model)
        self.self_layer_num = self_layer_num
        self.cross_layer_num = cross_layer_num

        self.stem = nn.ModuleList()
        for i in range(self_layer_num):
            self.stem.append(MHAttention(d_model, nhead=8))

        self.cross = nn.ModuleList()
        for i in range(cross_layer_num):
            self.cross.append(MHAttention(d_model, nhead=8))
            self.cross.append(CrossAttention(d_model, nhead=8))

        self.coor_mlp = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(d_model, 3, kernel_size=1, stride=1)
        )
        # Setting w to 128 is fine, but it will consume more computing resources
        # self.generator = RegistrationRender(rho=36, w=128)
        self.generator = RegistrationRender(rho=24, w=64)

    def forward(self, src, tgt, src_center=None, s=1, get_imgs=False, get_transform=False, filter=None):
        assert src.shape[0] == 1, "batch size must 1"
        assert tgt.shape[0] == 1, "batch size must 1"
        # batch x n x 3, batch x m x 3
        # src_feats = self.pe_func(src) + self.embedding(src.permute([0, 2, 1])).permute([0, 2, 1])
        if src_center is None:
            src_center = torch.mean(src, dim=1, keepdim=True)
        src_feats = self.pe_func(src) + self.embedding((src - src_center).permute([0, 2, 1])).permute([0, 2, 1])
        tgt_feats = self.pe_func(tgt) + self.embedding(tgt.permute([0, 2, 1])).permute([0, 2, 1])

        src_feats, tgt_feats = self.pre_norm(src_feats), self.pre_norm(tgt_feats)

        for i in range(self.self_layer_num):
            tgt_feats = self.stem[i](tgt_feats, tgt_feats, tgt_feats)
            src_feats = self.stem[i](src_feats, src_feats, src_feats)
        for i in range(2*self.cross_layer_num):
            if i % 2 == 0:
                src_feats = self.cross[i](src_feats, src_feats, src_feats)
            else:
                src_feats = self.cross[i](src_feats, tgt_feats)

        # batch x n x 3
        src_ = self.coor_mlp(src_feats.permute([0, 2, 1])).permute([0, 2, 1])

        imgs = None
        if get_imgs:
            for i in range(src.shape[0]):
                imgs = self.generator(src_[i], tgt[i])
        transform = None
        if get_transform:
            if filter is None:
                filter = tgt.detach()

            src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_.detach() / s, filter / s)[0].min(dim=1)
            valid_ind = (src_pred_2_tgt_dis < 0.075)
            tgt_ind = src_pred_2_tgt_ind[valid_ind]
            transform = rigid_transform_3d(src.detach()[:, valid_ind], filter.detach()[:, tgt_ind])[0]

            # if self.training:
            #     src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_.detach() / s, filter / s)[0].min(dim=1)
            #     valid_ind = (src_pred_2_tgt_dis < 0.075)
            #     tgt_ind = src_pred_2_tgt_ind[valid_ind]
            #     transform = rigid_transform_3d(src.detach()[:, valid_ind], filter.detach()[:, tgt_ind])[0]
            #     # transform = rigid_transform_3d(src.detach(), src_.detach())[0]
            # else:
            #     src_pred_2_tgt_dis, src_pred_2_tgt_ind = square_distance(src_.detach() / s, tgt.detach() / s)[0].min(dim=1)
            #     valid_ind = (src_pred_2_tgt_dis < 0.075)
            #     tgt_ind = src_pred_2_tgt_ind[valid_ind]
            #     print("valid num: %d / %d" % (torch.sum(valid_ind, dim=0).item(), src_.shape[1]))
            #     # corr = torch.cat([src[0].detach()[valid_ind] / s, tgt[0].detach()[tgt_ind] / s], dim=1)
            #     # transform = compute_post_by_maximum_clique(corr)
            #     transform = rigid_transform_3d(src.detach()[:, valid_ind], tgt.detach()[:, tgt_ind])[0]

        return src_, imgs, transform

    def loss(self, src_pred, src_pcd, rot_gt, trans_gt):
        src_pcd = torch.matmul(rot_gt, src_pcd.permute([0, 2, 1])) + trans_gt
        src_pcd = src_pcd.permute([0, 2, 1])
        # batch x (n+m)
        l1_loss = torch.sum(torch.abs(src_pred - src_pcd), dim=-1)
        l1_loss = torch.mean(l1_loss, dim=1)
        return l1_loss.mean(dim=0)


############################ Model for OpenGF ################################
class DEMTransformer(nn.Module):
    def __init__(self, d_model=256, self_layer_num=6, cross_layer_num=6):
        super(DEMTransformer, self).__init__()
        self.embedding = nn.Conv1d(3, d_model, kernel_size=1, stride=1)
        self.pre_norm = nn.LayerNorm(d_model)

        self.pe_func = PositionEmbeddingCoordsSine(n_dim=3, d_model=d_model)
        self.self_layer_num = self_layer_num
        self.cross_layer_num = cross_layer_num

        self.stem_dsm = nn.ModuleList()
        for i in range(self_layer_num):
            self.stem_dsm.append(MHAttention(d_model, nhead=8))
        self.stem_dem = nn.ModuleList()
        for i in range(self_layer_num):
            self.stem_dem.append(MHAttention(d_model, nhead=8))

        self.cross = nn.ModuleList()
        for i in range(cross_layer_num):
            self.cross.append(MHAttention(d_model, nhead=8))
            self.cross.append(CrossAttention(d_model, nhead=8))

        self.coor_mlp = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(d_model, 3, kernel_size=1, stride=1)
        )

        self.generator = RegistrationRender(rho=24, w=64, view="single")

    def forward(self, dsm, dem_coarse, get_imgs=False):
        assert dsm.shape[0] == 1, "batch size must 1"
        assert dem_coarse.shape[0] == 1, "batch size must 1"
        # batch x n x 3, batch x m x 3
        dsm_feats = self.pe_func(dsm) + self.embedding(dsm.permute([0, 2, 1])).permute([0, 2, 1])
        dem_coarse_feats = self.pe_func(dem_coarse) + self.embedding(dem_coarse.permute([0, 2, 1])).permute([0, 2, 1])

        dsm_feats, dem_coarse_feats = self.pre_norm(dsm_feats), self.pre_norm(dem_coarse_feats)

        for i in range(self.self_layer_num):
            dsm_feats = self.stem_dsm[i](dsm_feats, dsm_feats, dsm_feats)
            dem_coarse_feats = self.stem_dem[i](dem_coarse_feats, dem_coarse_feats, dem_coarse_feats)
        for i in range(2*self.cross_layer_num):
            if i % 2 == 0:
                dsm_feats = self.cross[i](dsm_feats, dsm_feats, dsm_feats)
            else:
                dsm_feats = self.cross[i](dsm_feats, dem_coarse_feats)

        # batch x n x 3
        dem_ = self.coor_mlp(dsm_feats.permute([0, 2, 1])).permute([0, 2, 1])

        imgs = None
        if get_imgs:
            for i in range(dsm.shape[0]):
                imgs = self.generator(dem_[i], dem_coarse[i])

        return dem_, imgs

    def loss(self, dem_pred, dem):
        # batch x n
        l1_loss = torch.sum(torch.abs(dem_pred - dem), dim=-1)
        l1_loss = torch.mean(l1_loss, dim=1)
        return l1_loss.mean(dim=0)