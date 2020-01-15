import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMatching(nn.ModuleList):
    def __init__(self, source_model, target_model):
        super(FeatureMatching, self).__init__()
        if hasattr(source_model.module, 'xchannels'):
            src_list = source_model.module.xchannels
        elif hasattr(source_model.module, 'channels'):
            # CifarResNet
            src_list = source_model.module.channels
        elif hasattr(source_model.module, '_C') and hasattr(source_model.module, '_layerN'):
            C = source_model.module._C
            N = source_model.module._layerN
            src_list   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        else:
            raise ValueError('invalid teacher : {:}'.format(source_model))
        if hasattr(target_model.module, 'xchannels'):
            tgt_list = target_model.module.xchannels
        elif hasattr(target_model.module, 'channels'):
            tgt_list = target_model.module.channels
        elif hasattr(target_model.module, '_C') and hasattr(target_model.module, '_layerN'):
            # TinyNetworkSETN
            C = target_model.module._C
            N = target_model.module._layerN
            tgt_list   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
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
        # matching_loss = []
        # print("*"*150)
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
                # with torch.no_grad():
                #     print("feature difference: {}".format(diff.pow(2).mean().item()))
                #     if diff.pow(2).mean().item() > 1e+4:
                #         print("Huge loss detected")
                #         print(diff.shape)
                #         print("Teacher: {:}".format(source_features[src_idx].mean().item()))
                #         print("Student: {:}".format(target_features[tgt_idx].mean().item()))
                #         print("Student + layer: {:}".format(self[i](target_features[tgt_idx]).mean().item()))
            else:
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2).mean(1).mul(beta)
            matching_loss += diff
            # matching_loss.append(diff.mean())
        # with torch.no_grad():
        #     print("\nTotal matching loss: {:}".format(matching_loss.sum().item()))
        #     if matching_loss.mean().item() > 1e+5:
        #         print("Teacher features")
        #         for i, feature in enumerate(source_features):
        #             print(i)
        #             print(feature[0].shape)
        #             print(feature[0].detach().cpu().numpy())
        #
        #         print("Student features")
        #         for i, feature in enumerate(target_features):
        #             print(i)
        #             print(feature[0].shape)
        #             print(feature[0].detach().cpu().numpy())
        return matching_loss
