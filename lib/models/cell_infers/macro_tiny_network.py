##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################
import torch
import torch.nn as nn
from ..cell_operations import ResNetBasicblock
from .cells import InferCell


# The macro structure for architectures in NAS-Bench-201
class MacroTinyNetwork(nn.Module):

  def __init__(self, C, N, genotype, num_classes, fixed_genotype, pos):
    super(MacroTinyNetwork, self).__init__()
    self._C               = C
    self._layerN          = N

    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    gen_pos = [False, False, False]
    gen_pos[pos] = True
    gen_layers = [gen_pos[0]] * N + [False] + [gen_pos[1]] * N + [False] + [gen_pos[2]] * N

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction, gen_pos) in enumerate(zip(layer_channels, layer_reductions, gen_layers)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2, True)
      elif gen_pos:
          cell = InferCell(genotype, C_prev, C_curr, 1)
      else:
          cell = InferCell(fixed_genotype, C_prev, C_curr, 1)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer= len(self.cells)

    self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward_for_outs(self, inputs):
    all_outs = []
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)
      all_outs.append(feature)
    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits, all_outs

  def forward(self, inputs, out_all=False):
    if out_all: return self.forward_for_outs(inputs)
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits