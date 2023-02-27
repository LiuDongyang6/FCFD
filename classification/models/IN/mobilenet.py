import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    stage_input = 'afterrelu'
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        return self.extract_feature(x)

    def extract_feature(self, x):
        feat0 = self.model[0](x)
        feat1 = self.model[1:4](feat0)
        feat2 = self.model[4:6](feat1)
        feat3 = self.model[6:12](feat2)
        feat4 = self.model[12:14](feat3)

        return [feat0, feat1, feat2, feat3, feat4]

    def stage_forward(self, stage, x):
        assert stage in [0, 1, 2, 3, 4], f"stage_forward called with invalid stage: {stage}"
        if stage == 0:
            x = self.model[0](x)
        elif stage == 1:
            x = self.model[1:4](x)
        elif stage == 2:
            x = self.model[4:6](x)
        elif stage == 3:
            x = self.model[6:12](x)
        elif stage == 4:
            x = self.model[12:14](x)
        else:
            raise ValueError

        return x

    def get_stage_module(self, stage):
        # used to support ghost forward
        assert stage in [0, 1, 2, 3, 4], f"stage_forward called with invalid stage: {stage}"
        if stage == 0:
            stage_module = self.model[0]
        elif stage == 1:
            stage_module = self.model[1:4]
        elif stage == 2:
            stage_module = self.model[4:6]
        elif stage == 3:
            stage_module = self.model[6:12]
        elif stage == 4:
            stage_module = self.model[12:14]
        else:
            raise ValueError

        return stage_module

    def forward_classifier(self, x):
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

