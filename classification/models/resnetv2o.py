'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        # out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        # out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    stage_input = 'beforerelu'
    name = 'ResNet'
    def __init__(self, block, num_blocks, num_classes=100, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        f4 = out
        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # f5 = out
        # out = self.linear(out)
        return [f0, f1, f2, f3, f4]

    def stage_forward(self, stage, x):
        assert stage in [0, 1, 2, 3, 4], f"stage_forward called with invalid stage: {stage}"
        if stage == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            f_pre= x
        elif stage == 1:
            x, f_pre = self.layer1(x)
        elif stage == 2:
            x, f_pre = self.layer2(x)
        elif stage == 3:
            x, f_pre = self.layer3(x)
        elif stage == 4:
            x, f_pre = self.layer4(x)
        else:
            raise ValueError

        return x

    def get_stage_module(self, stage):
        if stage == 0:
            stage_module = nn.ModuleList([self.conv1, self.bn1])
        elif stage == 1:
            stage_module = self.layer1
        elif stage == 2:
            stage_module = self.layer2
        elif stage == 3:
            stage_module = self.layer3
        elif stage == 4:
            stage_module = self.layer4
        else:
            raise ValueError

        return stage_module

    def forward_classifier(self, x):
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x



def ResNet18o(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34o(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50o(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101o(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152o(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)