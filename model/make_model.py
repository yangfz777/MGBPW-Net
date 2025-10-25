import torch
import torch.nn as nn
from numpy.ma.core import shape

from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import os

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier1(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.0001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class nonLocal(nn.Module):
    def __init__(self, in_dim):
        super(nonLocal, self).__init__()
        self.conv_query = nn.Linear(in_dim, in_dim)
        # self.conv_part = nn.Linear(in_dim, in_dim)
        self.conv_part1 = nn.Linear(in_dim, in_dim)
        self.conv_part2 = nn.Linear(in_dim, in_dim)
        self.conv_part3 = nn.Linear(in_dim, in_dim)
        self.conv_part4 = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.param = nn.Parameter(torch.zeros(1))  # 可学习的参数

        # 用于加权融合 query 和 part 的可学习权重
        self.query_weight = nn.Parameter(torch.ones(1))  # 权重初始化为1
        # self.part_weight = nn.Parameter(torch.ones(1))
        self.part1_weight = nn.Parameter(torch.ones(1))
        # self.part2_weight = nn.Parameter(torch.ones(1))
        # self.part3_weight = nn.Parameter(torch.ones(1))
        # self.part4_weight = nn.Parameter(torch.ones(1))# 权重初始化为1

    def forward(self, query, part1):
        # query 和 part 的形状均为 [32, 242, 768]

        # 通过卷积转换后，query 和 part 的形状仍为 [32, 242, 768]
        f_query = self.conv_query(query)  # [32, 242, 768]
        # f_part = self.conv_part(part)  # [32, 242, 768]
        f_part1 = self.conv_part1(part1)  # [32, 242, 768]
        # f_part2 = self.conv_part2(part2)  # [32, 242, 768]
        # f_part3 = self.conv_part3(part3)  # [32, 242, 768]
        # f_part4 = self.conv_part4(part4)  # [32, 242, 768]

        # 融合操作：加权加法，使用可学习的权重
        f_fused = self.query_weight * f_query + self.part1_weight * f_part1# + self.part2_weight * f_part2 + self.part3_weight * f_part3 + self.part4_weight * f_part4  # [32, 242, 768]

        # 其他可能的非局部操作，可以选择计算相似度或进一步处理
        # f_fused = some_other_operation(f_fused)  # 如果有其他操作

        return f_fused


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.img_path_list = []

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b3 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.NL = nn.ModuleList(
                [
                    nonLocal(768)
                    for _ in range(4)
                ]
            )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier2.apply(weights_init_classifier)
            self.classifier_occ1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_occ1.apply(weights_init_classifier)
            self.classifier_occ2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_occ2.apply(weights_init_classifier)
            self.classifier_occ3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_occ3.apply(weights_init_classifier)

            self.classifier_c = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_c.apply(weights_init_classifier)
            self.classifier_cp = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_cp.apply(weights_init_classifier)
            self.classifier_0 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_0.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)
            self.classifier_5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_5.apply(weights_init_classifier)

            self.classifier3_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier3_1.apply(weights_init_classifier)
            self.classifier3_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier3_2.apply(weights_init_classifier)
            self.classifier3_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier3_3.apply(weights_init_classifier)
            self.classifier3_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier3_4.apply(weights_init_classifier)

            self.classifier_bp = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp.apply(weights_init_classifier)
            self.classifier_bp1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp1.apply(weights_init_classifier)
            self.classifier_bp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp2.apply(weights_init_classifier)
            self.classifier_bp3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp3.apply(weights_init_classifier)
            self.classifier_bp4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp4.apply(weights_init_classifier)
            self.classifier_bp5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_bp5.apply(weights_init_classifier)

            self.linear_layer = nn.Linear(768, 6,bias=False).to('cuda')
            self.linear_layer.apply(weights_init_classifier1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck_occ1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_occ1.bias.requires_grad_(False)
        self.bottleneck_occ1.apply(weights_init_kaiming)

        self.bottleneck_occ2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_occ2.bias.requires_grad_(False)
        self.bottleneck_occ2.apply(weights_init_kaiming)

        self.bottleneck_occ3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_occ3.bias.requires_grad_(False)
        self.bottleneck_occ3.apply(weights_init_kaiming)

        self.bottleneck_0 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_0.bias.requires_grad_(False)
        self.bottleneck_0.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.bottleneck3_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3_1.bias.requires_grad_(False)
        self.bottleneck3_1.apply(weights_init_kaiming)
        self.bottleneck3_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3_2.bias.requires_grad_(False)
        self.bottleneck3_2.apply(weights_init_kaiming)
        self.bottleneck3_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3_3.bias.requires_grad_(False)
        self.bottleneck3_3.apply(weights_init_kaiming)
        self.bottleneck3_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3_4.bias.requires_grad_(False)
        self.bottleneck3_4.apply(weights_init_kaiming)

        self.bottleneck_bp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp.bias.requires_grad_(False)
        self.bottleneck_bp.apply(weights_init_kaiming)
        self.bottleneck_cp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_cp.bias.requires_grad_(False)
        self.bottleneck_cp.apply(weights_init_kaiming)
        self.bottleneck_bp0 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp0.bias.requires_grad_(False)
        self.bottleneck_bp0.apply(weights_init_kaiming)
        self.bottleneck_bp1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp1.bias.requires_grad_(False)
        self.bottleneck_bp1.apply(weights_init_kaiming)
        self.bottleneck_bp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp2.bias.requires_grad_(False)
        self.bottleneck_bp2.apply(weights_init_kaiming)
        self.bottleneck_bp3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp3.bias.requires_grad_(False)
        self.bottleneck_bp3.apply(weights_init_kaiming)
        self.bottleneck_bp4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp4.bias.requires_grad_(False)
        self.bottleneck_bp4.apply(weights_init_kaiming)
        self.bottleneck_bp5 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_bp5.bias.requires_grad_(False)
        self.bottleneck_bp5.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, x2, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features,output = self.base(x, cam_label=cam_label, view_label=view_label)
        features2,output2 = self.base(x2, cam_label=cam_label, view_label=view_label)
        patchs_features = features[:, 1:, :]
        batch_size, num_patches, feature_dim = patchs_features.shape
        features_reshaped = patchs_features.view(batch_size, 22, 11, feature_dim)
        output_features = self.linear_layer(features_reshaped.reshape(-1, 768)).reshape(batch_size, 22, 11, 6).to(
            'cuda')
        targets_min = output_features.min(dim=0, keepdim=True).values
        targets_min = targets_min.min(dim=1, keepdim=True).values
        targets_min = targets_min.min(dim=2, keepdim=True).values

        targets_max = output_features.max(dim=0, keepdim=True).values
        targets_max = targets_max.max(dim=1, keepdim=True).values
        targets_max = targets_max.max(dim=2, keepdim=True).values
        # 归一化
        targets_normalized = (output_features - targets_min) / (targets_max - targets_min + 1e-8)
        # 检查特征数值范围
        # 归一化身体部位注意力权重，确保每个部分的权重合理
        body_part_attention = F.softmax(targets_normalized, dim=-1)  # [32, 22, 11, 6]
        part_body = body_part_attention[:, :, :, [1, 2, 3, 4, 5]].detach()
        # 设置温度系数
        temperature = 2.0
        part_body = part_body.permute(0, 3, 1, 2)
        part_body = part_body.sum(dim=(2, 3))
        part_body = part_body / temperature
        part_body = F.softmax(part_body, dim=-1)
        part_body_weight = part_body
        patch_features = features[:, 1:, :]
        batch_size, num_patches, feature_dim = patch_features.shape
        feature_reshaped = patch_features.view(batch_size, 22, 11, feature_dim)
        # 初始化存储每个身体部位融合后的特征
        individual_channel_features = []
        mask_fore = body_part_attention[:, :, :, 1:].sum(dim=-1, keepdim=True)
        # print(mask_fore[0])

        fore_feat = mask_fore * feature_reshaped
        # fore_features = fore_feat.mean(dim=(1, 2))  # [32, 768]
        fore_features = fore_feat.reshape(batch_size, 22 * 11, 768)
        # 将当前部分的全局特征加入融合列表
        individual_channel_features.append(fore_features)

        # 遍历每个身体部位通道，融合特征
        for i in range(1, 6):
            part_attention = body_part_attention[..., i:i + 1]  # 取出第 i 个通道权重 [32, 22, 11, 1]

            # 对原始特征进行加权 [32, 22, 11, 768] * [32, 22, 11, 1]
            weighted_features = feature_reshaped * part_attention  # 加权后的特征 [32, 22, 11, 768]

            # 对空间维度进行全局平均池化，得到每个部分的全局特征
            # pooled_features = weighted_features.mean(dim=(1, 2))  # [32, 768]
            pooled_features = weighted_features.reshape(batch_size, 22 * 11, 768)
            # 将当前部分的全局特征加入融合列表
            individual_channel_features.append(pooled_features)

        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat_ori = b1_feat[:, 0]


        # global branch
        b1_feat_occ1 = self.b1(features2)  # [64, 129, 768]
        global_feat2 = b1_feat_occ1[:, 0]


        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
            # x3 = shuffle_unit(features3, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
            # x3 = features3[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]


        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]


        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        global_feat=global_feat_ori
        feat = self.bottleneck(global_feat)
        feat2 = self.bottleneck_occ1(global_feat2)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        local_feat_p0 = self.b2(torch.cat((token, individual_channel_features[0]), dim=1))
        local_feat_p0 = local_feat_p0[:, 0]

        local_feat_p1 = self.b2(torch.cat((token, individual_channel_features[1]), dim=1))
        local_feat_p1 = local_feat_p1[:, 0]

        local_feat_p2 = self.b2(torch.cat((token, individual_channel_features[2]), dim=1))
        local_feat_p2 = local_feat_p2[:, 0]

        local_feat_p3 = self.b2(torch.cat((token, individual_channel_features[3]), dim=1))
        local_feat_p3 = local_feat_p3[:, 0]

        local_feat_p4 = self.b2(torch.cat((token, individual_channel_features[4]), dim=1))
        local_feat_p4 = local_feat_p4[:, 0]

        local_feat_p5 = self.b2(torch.cat((token, individual_channel_features[5]), dim=1))
        local_feat_p5 = local_feat_p5[:, 0]

        bp_feat0_bn = self.bottleneck_bp0(local_feat_p0)
        bp_feat1_bn = self.bottleneck_bp1(local_feat_p1)
        bp_feat2_bn = self.bottleneck_bp2(local_feat_p2)
        bp_feat3_bn = self.bottleneck_bp3(local_feat_p3)
        bp_feat4_bn = self.bottleneck_bp4(local_feat_p4)
        bp_feat5_bn = self.bottleneck_bp5(local_feat_p5)

        if self.training:

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)

            else:
                cls_score = self.classifier(feat)
                cls_score_occ1 = self.classifier_occ1(feat2)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)

                cls_score_bp0 = self.classifier_0(bp_feat0_bn)
                cls_score_bp1 = self.classifier_1(bp_feat1_bn)
                cls_score_bp2 = self.classifier_2(bp_feat2_bn)
                cls_score_bp3 = self.classifier_3(bp_feat3_bn)
                cls_score_bp4 = self.classifier_4(bp_feat4_bn)
                cls_score_bp5 = self.classifier_5(bp_feat5_bn)

            return [cls_score, cls_score_occ1, cls_score_1, cls_score_2, cls_score_3,
                    cls_score_4,  cls_score_bp0, cls_score_bp1, cls_score_bp2, cls_score_bp3,
                    cls_score_bp4,cls_score_bp5, part_body_weight

                    ], [global_feat, output_features, global_feat2, local_feat_1,
                        local_feat_2, local_feat_3, local_feat_4, local_feat_p0, local_feat_p1, local_feat_p2,
                        local_feat_p3, local_feat_p4, local_feat_p5]

        else:

            if self.neck_feat == 'after':
                return torch.cat(

                     [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)

            else:
                return torch.cat(

                    [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4,local_feat_p1,local_feat_p2,local_feat_p3,local_feat_p4,local_feat_p5], dim=1)



    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)

            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
