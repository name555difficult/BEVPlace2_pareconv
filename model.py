import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.ops import point_to_node_partition, index_select
from pareconv.modules.registration import get_node_correspondences

from pareconv.modules.dual_matching import PointDualMatching

from pareconv.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)

from pareconv.modules.registration import HypothesisProposer

from backbone import PAREConvFPN


class PARE_Net(nn.Module):
    def __init__(self, init_dim, output_dim, kernel_size, share_nonlinearity, conv_way, use_xyz, use_encoder_re_feats):
        super(PARE_Net, self).__init__()

        self.backbone = PAREConvFPN(
            init_dim,
            output_dim,
            kernel_size,
            share_nonlinearity,
            conv_way,
            use_xyz,
            use_encoder_re_feats
        )

    def split_feats(self, feats_a, lengths):
        last_idx = 0
        feats_list = []
        for i in range(len(lengths)):
            feats_list.append(feats_a[last_idx:(last_idx+lengths[i].item())])
            last_idx = last_idx + lengths[i]
        
        return feats_list

    def pooling_feats(self, feats_list):
        
        global_feats = []
        for feats in feats_list:
            feat = torch.mean(feats, dim=0, keepdim=False)
            global_feats.append(feat)
        
        return global_feats

    def forward(self, data_dict):
        output_dict = {}
        
        # 2. PARE-Conv Encoder
        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(data_dict)

        feats_c_list = self.split_feats(feats_c, data_dict['lengths'][-1])
        feats_f_list = self.split_feats(feats_f, data_dict['lengths'][1])

        global_feat_c_list = self.pooling_feats(feats_c_list)
        global_feat_f_list = self.pooling_feats(feats_f_list)

        global_feats_c = torch.stack(global_feat_c_list)
        global_feats_f = torch.stack(global_feat_f_list)

        global_feats = torch.cat((global_feats_c, global_feats_f), dim=-1)

        return global_feats

class GenerateCenter(nn.Module):
    def __init__(self, VLADNet_ipt_size, cluster_size, feature_size,
                 args=None):
        super(GenerateCenter, self).__init__()
        self.VLADNet_ipt_size = VLADNet_ipt_size
        self.cluster_size = cluster_size
        self.feature_size = feature_size
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / torch.sqrt(torch.tensor(feature_size, dtype=torch.float)))

    def forward(self, feature):
        """
        :param feature: [B, f, N]
        :return: center: [B, f, cluster_size]
        """

        center = self.cluster_weights
        return center