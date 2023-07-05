import torch
from torch import nn
import torch.nn.functional as F

def build_bridges(cfg):
    in_channels = [256,256,256,256,256]
    out_channels = [256,256,256,256,256]

    bridges = {"s2t": [], "t2s": []}
    for stage in range(len(in_channels)):
        bridge_s2t = nn.Sequential(*[
            nn.Conv2d(in_channels[stage], out_channels[stage], kernel_size=3, stride=1, padding=1,
                            bias=False),
            nn.BatchNorm2d(out_channels[stage]),
        ])
        bridges['s2t'].append(bridge_s2t)
        bridge_t2s = nn.Sequential(*[
            nn.Conv2d(out_channels[stage], in_channels[stage], kernel_size=3, stride=1, padding=1,
                            bias=False),
            nn.BatchNorm2d(in_channels[stage]),
        ])
        bridges['t2s'].append(bridge_t2s)
    for key in bridges:
        bridges[key] = nn.ModuleList(bridges[key])
    bridges = nn.ModuleDict(bridges)
    for m in bridges.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return bridges
