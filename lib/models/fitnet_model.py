import torch
import torch.nn as nn
import torch.nn.functional as F
from .CifarResNet import CifarResNet
from .CifarDenseNet import DenseNet
from .CifarWideResNet import CifarWideResNet

class FeatureMatching(nn.ModuleList):
    def __init__(self, source_model, target_model):
        super(FeatureMatching, self).__init__()
        if isinstance(source_model.module, CifarResNet):
            src_list = source_model.module.channels
        elif isinstance(source_model.module, DenseNet):
            raise NotImplementedError
        elif hasattr(source_model.module, '_C') and hasattr(source_model.module, '_layerN'):
            # TinyNetwork
            C = source_model.module._C
            N = source_model.module._layerN
            src_list   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        elif isinstance(source_model.module, CifarWideResNet):
            src_list = [160, 320, 640]
        else:
            raise ValueError('invalid teacher : {:}'.format(source_model))

        if isinstance(target_model.module, CifarResNet):
            tgt_list = target_model.module.channels
        elif isinstance(target_model.module, DenseNet):
            raise NotImplementedError
        elif hasattr(target_model.module, '_C') and hasattr(target_model.module, '_layerN'):
            # TinyNetwork
            C = target_model.module._C
            N = target_model.module._layerN
            tgt_list   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        elif isinstance(target_model.module, CifarWideResNet):
            tgt_list = [160, 320, 640]
        else:
            raise ValueError('invalid student : {:}'.format(target_model))
        src_reductions = [ len(src_list) - src_list[::-1].index(src_list[0] * i) - 1 for i in [1,2,4] ]
        tgt_reductions = [ len(tgt_list) - tgt_list[::-1].index(tgt_list[0] * i) - 1 for i in [1,2,4] ]
        pairs = [ (src_f, tgt_f) for (src_f, tgt_f) in zip(src_reductions, tgt_reductions) ]

        self.pairs = pairs
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.beta = 0.5

        for src_idx, tgt_idx in pairs:
            self.append(nn.Sequential(nn.ReLU(), nn.Conv2d(tgt_list[tgt_idx], src_list[src_idx], 1)))
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
        self.apply(weights_init)
    def forward(self, source_features, target_features, beta=None):
        assert len(source_features) == len(self.src_list), "Teacher feature not match: features {:} != list {:}".format(len(source_features), len(self.src_list))
        assert len(target_features) == len(self.tgt_list), "Student feature not match: features {:} != list {:}".format(len(target_features), len(self.tgt_list))
        if beta is None:
            beta = self.beta
        matching_loss = 0.0
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            else:
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2).mean(1).mul(beta)
            matching_loss += diff
        return matching_loss
