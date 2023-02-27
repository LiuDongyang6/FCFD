import torch
import torch.nn as nn
import random
from utils import freeze, unfreeze
import torch.nn.functional as F

def _first_n(path, until_stage):
    assert isinstance(path, (str, list))
    if until_stage < 0:
        raise ValueError
    if isinstance(path, str):
        path = path.split('.')
    if until_stage == 0:
        return path[0]
    else:
        return '.'.join(path[:until_stage+1])

def _is_pure(path_name):
    if path_name == '':
        return True
    l = path_name.split('.')
    first = l[0]
    if all([_ == first for _ in l]):
        return True
    else:
        return False

class ClsHead(nn.Module):
    def __init__(self, f_dim=64, num_classes=100):
        super(ClsHead, self).__init__()
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(f_dim, num_classes)

    def forward(self, features):
        x = self.relu(features)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class MetaModel(nn.Module):
    def __init__(self, models, extra_type, input_size=(1, 3, 32, 32), num_classes=100):
        super(MetaModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        self.channels = []
        self.shapes = []
        with torch.no_grad():
            for i in range(self.n_models):
                demo_input = torch.zeros(input_size, dtype=torch.float, device=list(self.models[i].parameters())[0].device)
                demo_output = self.models[i](demo_input)
                self.channels.append([_.shape[1] for _ in demo_output])
                self.shapes.append([_.shape[2] for _ in demo_output])

        self.n_stages = len(self.channels[0])
        self.output_stages = self.n_stages - 1
        self.extra_path_list = self._create_candidate_path(extra_type)
        print(f"extra_type: {extra_type}")
        print(f"extra_path_list: {self.extra_path_list}")
        self.default_paths = [[m_id] * self.n_stages for m_id in range(self.n_models)]

        # build bridges
        self.bridges = \
            [
                nn.ModuleList([
                    nn.ModuleList([
                        nn.Module() for model_source in range(self.n_models)
                    ])
                    for model_target in range(self.n_models)
                ])
                for stage in range(self.n_stages)
            ]
        for stage in range(1, self.n_stages):
            for m in range(self.n_models):
                for m_s in range(self.n_models):
                    if m_s == m:
                        continue
                    self.bridges[stage][m][m_s] = self.build_bridge(m_s, m , stage)
        self.bridges = nn.ModuleList(self.bridges)

        for m in self.bridges.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, extra_path_num=0, val=False):
        if val:
            extra_path_num = 0
        extra_paths = random.sample(self.extra_path_list, extra_path_num)

        full_paths = self.default_paths + extra_paths
        for i, p in enumerate(full_paths):
            full_paths[i] = [str(_) for _ in p]

        inters = [[{} for m in range(self.n_models)] for stage in range(self.n_stages)]
        for stage_idx in range(self.n_stages):
            for path_idx, path in enumerate(full_paths):
                m = int(path[stage_idx])
                if stage_idx == 0:
                    stage_input = images
                else:
                    stage_input = inters[stage_idx - 1][m][_first_n(path, stage_idx - 1)]

                out_feat_name = _first_n(path, stage_idx)
                out_is_pure = _is_pure(out_feat_name)
                if out_feat_name not in inters[stage_idx][m]:
                    with GhostForward(self.models[m].get_stage_module(stage_idx), freeze_model=False,
                                      own_bn_trace=True, bn_mark=out_feat_name, on=not out_is_pure):
                        output = self.models[m].stage_forward(stage_idx, stage_input)

                    inters[stage_idx][m][out_feat_name] = output
                    if out_is_pure and stage_idx!=0:
                        for other_m in range(self.n_models):
                            if other_m != m:
                                need_pass_bri = False
                                if other_m == 0:
                                    need_pass_bri = True
                                else:
                                    for p in full_paths[self.n_models:]:
                                        if _first_n(p, stage_idx) == out_feat_name and int(p[stage_idx+1]) == other_m:
                                            need_pass_bri = True
                                            break
                                if need_pass_bri:
                                    bridge = self.bridges[stage_idx][other_m][m]
                                    inters[stage_idx][other_m][out_feat_name] = bridge(output)

        outputs = []

        for path_idx, path in enumerate(full_paths):
            feat_name = '.'.join(path)
            m = path[-1]
            cls_input = inters[self.n_stages-1][int(m)][feat_name]
            cls_out = self.models[int(m)].forward_classifier(cls_input)
            outputs.append(cls_out)

        if not val:
            return outputs, inters[1:]
        else:
            return outputs, ['.'.join(_) for _ in full_paths]

    def _create_candidate_path(self, sample_type):
        if sample_type=="bi":
            if self.n_stages == 4:
                candidates = [[0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
            else: #5
                candidates = [[0,0,1,1,1], [0,0,0,1,1], [0,0,0,0,1], [1,1,0,0,0], [1,1,1,0,0], [1,1,1,1,0] ]
        elif sample_type=="s2t":
            if self.n_stages == 4:
                candidates = [[1, 1, 0, 0], [1, 1, 1, 0]]
            else: #5
                candidates = [[1,1,0,0,0], [1,1,1,0,0], [1,1,1,1,0] ]
        elif sample_type=="t2s":
            if self.n_stages == 4:
                candidates = [[0, 0, 1, 1], [0, 0, 0, 1]]
            else: #5
                candidates = [[0,0,1,1,1], [0,0,0,1,1], [0,0,0,0,1] ]
        elif sample_type=="nomix":
            candidates = []
        else:
            raise ValueError("unknown sample type")

        return candidates

    def freeze_teacher(self):
        print("freeze teacher!")
        freeze(self.models[0])

    def load_teacher(self, state_dict):
        teacher_current = self.models[0].state_dict()
        for key, val in state_dict.items():
            teacher_current[key].copy_(val)

    def build_bridge(self, m_s, m, stage):
        bridge = [
            torch.nn.Conv2d(self.channels[m_s][stage], self.channels[m][stage], kernel_size=3, stride=1, padding=1,
                            bias=False),
            nn.BatchNorm2d(self.channels[m][stage])]
        if self.shapes[m_s][stage] != self.shapes[m][stage]:
            if self.shapes[m_s][stage] == self.shapes[m][stage] * 2:
                bridge[0] = nn.Conv2d(self.channels[m_s][stage], self.channels[m][stage], kernel_size=3, stride=2,
                                      padding=1)
            elif self.shapes[m_s][stage] * 2 == self.shapes[m][stage]:
                bridge[0] = nn.ConvTranspose2d(self.channels[m_s][stage], self.channels[m][stage], kernel_size=4,
                                               stride=2, padding=1)
            else:
                raise NotImplementedError(
                    f"stu shape: {self.shapes[m_s][stage]}, tea shape: {self.shapes[m][stage]}")

        if self.models[m].stage_input == "afterrelu":
            bridge.append(nn.LeakyReLU(negative_slope=0.1))

        return nn.Sequential(*bridge)

class GhostForward:
    def __init__(self, model: nn.Module, freeze_model=True, own_bn_trace=True, bn_mark='ghost', on=True):
        assert bn_mark != "main", "We are ghosts, not bandits"
        self.on = on
        self.model = model
        self.freeze_model = freeze_model
        self.own_bn_trace = own_bn_trace
        self.bn_mark = bn_mark
        self.frozen_by_me = []
    def __enter__(self):
        if not self.on:
            return

        if self.freeze_model:
            for p in self.model.parameters():
                if p.requires_grad:
                    self.frozen_by_me.append(p)
                    p.requires_grad = False
        if self.own_bn_trace:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if not hasattr(m, 'ghost_dict'):
                        m.ghost_dict = {}
                    m.ghost_dict['main'] = {}
                    m.ghost_dict['main']['running_mean'] = m.running_mean
                    m.ghost_dict['main']['running_var'] = m.running_var
                    m.ghost_dict['main']['num_batches_tracked'] = m.num_batches_tracked
                    if self.bn_mark in m.ghost_dict:
                        ori_dict = m.ghost_dict[self.bn_mark]
                        m.running_mean = ori_dict['running_mean']
                        m.running_var = ori_dict['running_var']
                        m.num_batches_tracked = ori_dict['num_batches_tracked']
                    else:
                        m.running_mean = torch.zeros_like(m.running_mean)
                        m.running_var = torch.zeros_like(m.running_var)
                        m.num_batches_tracked = torch.zeros_like(m.num_batches_tracked)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.on:
            return

        if self.freeze_model:
            for p in self.frozen_by_me:
                p.requires_grad = True
        if self.own_bn_trace:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.ghost_dict[self.bn_mark] = {}
                    m.ghost_dict[self.bn_mark]['running_mean'] = m.running_mean
                    m.ghost_dict[self.bn_mark]['running_var'] = m.running_var
                    m.ghost_dict[self.bn_mark]['num_batches_tracked'] = m.num_batches_tracked

                    m.running_mean = m.ghost_dict['main']['running_mean']
                    m.running_var = m.ghost_dict['main']['running_var']
                    m.num_batches_tracked = m.ghost_dict['main']['num_batches_tracked']

