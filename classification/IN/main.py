import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
import random
import numpy as np
import time
import shutil
import argparse
from easydict import EasyDict as edict
import yaml
import datetime

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from timm.utils import NativeScaler
from timm.scheduler import CosineLRScheduler

from utils import prepare_dirs, save_config
from data_loader import get_IN_train_loader, get_IN_test_loader
from utils import accuracy, AverageMeter, LazyAverageMeterDict, print_and_write
import models


parser = argparse.ArgumentParser(description='FCFD ImageNet experiment')
USE_FP16 = False


def str2bool(v):
    return v.lower() in ('true', '1')

# data params
data_arg = parser.add_argument_group('Data Params')
data_arg.add_argument('--num_classes', type=int, default=1000,
                      help='Number of classes to classify')
data_arg.add_argument('--batch_size', type=int, default=256,
                      help='# of images in each batch of data')
data_arg.add_argument('--data_dir', type=str, default=os.path.join(os.environ['HOME'],'data/imagenet/images'),
                      help='Directory in which data is stored')

# training params
train_arg = parser.add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum value')
train_arg.add_argument('--epochs', type=int, default=100,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.1,
                       help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=1e-4,
                       help='value of weight dacay for regularization')
train_arg.add_argument('--nesterov', type=str2bool, default=True,
                       help='Whether to use Nesterov momentum')
train_arg.add_argument('--gamma', type=float, default=0.1,
                       help='value of learning rate decay')
train_arg.add_argument('--scheduler', type=str, default='step', choices=['step', 'cos'],
                       help='value of learning rate decay')
train_arg.add_argument('--resume', default=None, type=str,
                      help='If not None, resume from this checkpoint')
train_arg.add_argument('--FP16', action="store_true",
                      help='if use FP16 for training')

# FCFD params
fcfd_arg = parser.add_argument_group('FCFD Params')
fcfd_arg.add_argument('--extra_paths', type=int, default=2,
                      help="number of extra paths for training")
fcfd_arg.add_argument('--extra_type', type=str, default="bi", choices=["bi", "s2t", 't2s', 'nomix'],
                      help="extra path choice")
fcfd_arg.add_argument('--student', type=str, required=True,
                      help='student model')
fcfd_arg.add_argument('--teacher', type=str, default=None,
                      help='teacher model')
fcfd_arg.add_argument('--teacher_ckpt', type=str, default=None,
                      help='pretrained teacher path')

# other params
misc_arg = parser.add_argument_group('Misc.')
misc_arg.add_argument('--random_seed', type=int, default=0,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--output_dir', type=str, default='./output/',
                      help='output directory')
misc_arg.add_argument('--save_name', type=str, default='model',
                      help='Name of the model to save as')
misc_arg.add_argument('--config', type=str, default=None,
                      help='config yaml file')

def get_config():
    args, unparsed = parser.parse_known_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = edict(yaml.load(f, Loader=yaml.FullLoader))
    else:
        config = edict()

    for k, v in args.__dict__.items():
        config[k] = v

    if config.FP16:
        global USE_FP16
        USE_FP16=True
    return config, unparsed


def main():
    config, unparsed = get_config()

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    kwargs = {'num_workers': 8, 'pin_memory': True}
    # instantiate data loaders
    test_data_loader = get_IN_test_loader(
        config.data_dir, config.batch_size, **kwargs
    )
    
    if config.is_train:
        # ensure directories are setup
        prepare_dirs(config)
        save_config(config)

        train_data_loader = get_IN_train_loader(
            config.data_dir, config.batch_size,
            **kwargs
        )
        data_loader = (train_data_loader, test_data_loader)
    else:
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes

        # training params
        self.model_prefix = config.save_name

        self.model_factory = models

        self.model_num = 2
        self.best_valid_top1 = [0.] * self.model_num
        self.best_valid_top5 = [0.] * self.model_num
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.output_stages=None

        self.loss_ce = nn.CrossEntropyLoss()

    def train(self):
        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )
        print(f"USE FP16: {USE_FP16}")
        self.scaler = NativeScaler()

        model_list = []
        for i in range(self.model_num):
            # build models
            if i == 0:
                m = self.model_factory.__dict__[self.config.teacher](num_classes=1000, pretrained=True)
                print(f"creating model {i} (teacher): {self.config.teacher}")
            else:
                m = self.model_factory.__dict__[self.config.student](num_classes=1000)
                print(f"creating model {i} (student): {self.config.student}")
            model_list.append(m)
            m.cuda()
            summary(m, input_size=[(3, 224, 224)], batch_size=1, device="cuda")

        config = self.config

        self.model = self.model_factory.MetaModel(model_list,
                                                  input_size=(1, 3, 224, 224),
                                                  extra_type=self.config.extra_type,
                                                  num_classes=1000)
        self.output_stages = self.model.output_stages
        self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.init_lr, momentum=config.momentum,
                                   weight_decay=config.weight_decay, nesterov=config.nesterov)

        if self.config.scheduler == "step":
            if self.config.epochs == 100:
                mile = [30, 60, 90]
            else:
                raise ValueError
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=mile,
                                                       gamma=self.config.gamma, last_epoch=-1)
        elif self.config.scheduler == "cos":
            self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.config.epochs, warmup_t=0, warmup_lr_init=1e-5,
                                               lr_min=0.0001)
        else:
            raise ValueError

        start_epoch = 0
        if self.config.resume is not None:
            checkpoint = torch.load(self.config.resume, device='cuda')
            start_epoch = checkpoint['epoch'] + 1
            self.best_valid_top1 = checkpoint['best_valid_top1']
            self.best_valid_top5 = checkpoint['best_valid_top5']
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optim_state'])
            if USE_FP16 and 'scaler_state' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state'])

        # freeze teacher
        self.model.freeze_teacher()
        print(f"teacher freezed!!! ")

        for epoch in range(start_epoch, self.config.epochs):

            self.scheduler.step(epoch)

            print('\nEpoch: {}/{}'.format(epoch + 1, self.config.epochs))

            # train for 1 epoch
            self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_top1, valid_top5 = self.validate(epoch)

            for i in range(self.model_num):
                is_best = valid_top1[i].avg > self.best_valid_top1[i]
                if is_best and i == 1:
                    shutil.copyfile(
                        os.path.join(self.config.logs_dir, f"latest_result.log"),
                        os.path.join(self.config.logs_dir, f"stubest.log"))

                self.best_valid_top1[i] = max(valid_top1[i].avg, self.best_valid_top1[i])
                self.best_valid_top5[i] = max(valid_top5[i].avg, self.best_valid_top5[i])
                print(f'best_top1: {self.best_valid_top1[i]} best_top5: {self.best_valid_top5[i]}')

                if not is_best and i > 0:
                    continue
                self.save_checkpoint({'epoch': epoch,
                                      'model_state': self.model.state_dict(),
                                      'optim_state': self.optimizer.state_dict(),
                                      'scaler_state': self.scaler.state_dict() if USE_FP16 else None,
                                      'best_valid_top1': self.best_valid_top1,
                                      'best_valid_top5': self.best_valid_top5
                                      }, is_best, i
                                     )

    def train_one_epoch(self, epoch):
        default_T = self.config.T
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        n_total_paths = self.model_num + self.config.extra_paths

        meter_losses = []
        meter_accs = []
        for i in range(n_total_paths):
            meter_losses.append(LazyAverageMeterDict())
            meter_accs.append(AverageMeter())

        meter_l2 = [AverageMeter() for _ in range(self.output_stages)]

        epoch_start_time = None
        tic = time.time()
        for iter_count, (images, labels) in enumerate(self.train_loader):
            if iter_count == 0:
                epoch_start_time = time.time()
            images, labels = images.cuda(), labels.cuda()
            data_time.update(time.time() - tic)

            # forward pass
            with torch.cuda.amp.autocast(USE_FP16):
                outputs, inters_all_stages = self.model(images, extra_path_num=self.config.extra_paths)

            total_loss = 0

            for stage in range(len(inters_all_stages)):
                is_final = stage==(len(inters_all_stages)-1)

                if iter_count == 0:
                    print(f"stage {stage}")

                # appearance l2
                inters = inters_all_stages[stage][0]
                inter_target = list(inters.values())[0].detach()
                loss_l2 = 0
                for val in list(inters.values())[1:]:
                    loss_l2 += F.mse_loss(val, inter_target)
                if is_final:
                    inter_target = F.adaptive_avg_pool2d(F.relu(inter_target), 1).view(images.shape[0], -1)
                    for val in list(inters.values())[1:]:
                        val = F.adaptive_avg_pool2d(F.relu(val), 1).view(images.shape[0], -1)
                        loss_l2 += F.mse_loss(val, inter_target)
                meter_l2[stage].update(loss_l2.item())
                if epoch < 5:
                    loss_l2 = min(1, (epoch+1) / 5) * loss_l2
                total_loss += loss_l2 * self.config.weight_l2

            for i, output in enumerate(outputs):
                if i == 0:
                    loss_weights = self.config.loss_teacher
                elif i == 1:
                    loss_weights = self.config.loss_student
                else:
                    loss_weights = self.config.loss_mixture

                loss_i_dict = {}
                loss_i = 0

                loss_i_dict['loss_label'] = 0
                loss_i_dict['kl_stu_weighted'] = 0
                kl_stu_weights = {}
                t_kl_stu = {}
                with torch.cuda.amp.autocast(USE_FP16):
                    label_weight = self._get_loss_weight("label", epoch, loss_weights)
                    if label_weight != 0:
                        loss_i_dict['loss_label'] = self.loss_ce(output, labels)
                    for j in range(len(outputs)):
                        if i != j:
                            T = loss_weights['kl_student'][min(j, self.model_num)].get("T", default_T)
                            t_kl_stu[j] = T
                            weight = self._get_loss_weight(min(j, self.model_num), epoch, loss_weights['kl_student'])
                            if weight != 0:
                                kl_loss_ij = self._loss_kl(outputs[i], outputs[j] ,T)
                                loss_i_dict['kl_stu_weighted'] += kl_loss_ij * weight
                            kl_stu_weights[j] = weight

                loss_i += loss_i_dict['kl_stu_weighted']
                loss_i += loss_i_dict['loss_label'] * label_weight

                to_item = lambda x: x.item() if isinstance(x, Tensor) else x
                meter_losses[i].update('total', to_item(loss_i), images.size()[0])
                for key in loss_i_dict.keys():
                    meter_losses[i].update(key, to_item(loss_i_dict[key]), images.size()[0])

                if iter_count == 0:
                    print(f"model {i}: label_w {label_weight:.3f} "
                          f"teacher_w {' '.join([f'{key}:{val}' for key, val in kl_stu_weights.items()])} "
                          f"T_kl_student: {' '.join([f'{key}:{val}' for key, val in t_kl_stu.items()])} "
                          f"lr: {self.optimizer.param_groups[0]['lr']} ")

                total_loss += loss_i
                # measure accuracy and record loss
                prec = accuracy(output.data, labels.data, topk=(1,))[0]
                meter_accs[i].update(prec.item(), images.size()[0])


            self.optimizer.zero_grad(set_to_none=True)
            self.scaler(total_loss, self.optimizer)

            batch_time.update(time.time() - tic)
            tic = time.time()

            if epoch == 0 and iter_count % 100 == 0:
                eta_seconds = batch_time.avg * (len(self.train_loader) - iter_count)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f'step {iter_count}/ {len(self.train_loader)} \t'
                      f'batch time {batch_time.avg:.4f} \t'
                      f'data time {data_time.avg:.4f} \t'
                      f'ETA {eta_string} \t'
                      f"lr {self.optimizer.param_groups[0]['lr']}")

        epoch_end_time = time.time()
        MB = 1024.0 * 1024.0
        print(f"epoch_training time: {epoch_end_time - epoch_start_time:.2f}, max memory: {torch.cuda.max_memory_allocated() / MB:.2f}")


        for stage in range(self.output_stages):
            print(f"******** stage {stage} ********")
            print(f"l2: {meter_l2[stage].avg} weight: {self.config.weight_l2}")


        for i in range(n_total_paths):
            print(f"model{i}: " + " ".join([f"{key}:{val.avg:.3f}" for key, val in meter_losses[i].items()]))
            print(f"model_{i}: train acc: {meter_accs[i].avg} ")

        return

    @torch.cuda.amp.autocast(enabled=USE_FP16)
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        meter_top1_dict = LazyAverageMeterDict()
        meter_top5_dict = LazyAverageMeterDict()

        for _, (images, labels) in enumerate(self.valid_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs, full_paths = self.model(images, val=True)

            for i in range(len(outputs)):
                prec1, prec5 = accuracy(outputs[i].data, labels.data, topk=(1, 5))
                meter_top1_dict.update(full_paths[i], prec1.item(), images.size()[0])
                meter_top5_dict.update(full_paths[i], prec5.item(), images.size()[0])

        with open(os.path.join(self.config.logs_dir, f"latest_result.log"), 'w') as f:
            for path_name in meter_top1_dict.keys():
                print_and_write(f, f"model {path_name}: top1: {meter_top1_dict[path_name].avg:.3f} top5: {meter_top5_dict[path_name].avg:.3f}")

        return list(meter_top1_dict.inner.values()), list(meter_top5_dict.inner.values())

    @torch.no_grad()
    def test(self):
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model = self.model_factory.__dict__[self.config.student](num_classes=1000)
        self.model.eval()
        self.model.cuda()
        # load checkpoint
        state_meta = torch.load(self.config.resume, map_location='cuda')['model_state']
        state_to_load = {}
        for key, val in state_meta.items():
            assert isinstance(key, str)
            if key.startswith("models.1."):
                state_to_load[key[9:]] = val
        self.model.load_state_dict(state_to_load)
        del state_meta

        self.model.eval()
        for i, (images, labels) in enumerate(self.test_loader):
            images, labels = images.cuda(), labels.cuda()

            # forward pass
            feats = self.model(images)
            outputs = self.model.forward_classifier(feats[-1])

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            top1.update(prec1.item(), images.size()[0])
            top5.update(prec5.item(), images.size()[0])

        print(
            '[*] top1_acc: {:.3f}%, top5_acc: {:.3f}%'.format(
                top1.avg, top5.avg)
        )

    def save_checkpoint(self, state, is_best, best_model_idx=None):
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_prefix + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.config.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if state['epoch'] == self.config.epochs - 1:
            filename = self.model_prefix + '_final_ckpt.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.config.ckpt_dir, filename))

        if int(state['epoch']) in self.config.get("save_epoch", []):
            filename = self.model_prefix + f'_epoch_{int(state["epoch"])}.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.config.ckpt_dir, filename)
            )

        if is_best:
            assert best_model_idx is not None
            filename = self.model_prefix + str(best_model_idx) + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.config.ckpt_dir, filename)
            )

    @staticmethod
    def _get_loss_weight(loss_name, epoch, loss_config):
        weight_dict = loss_config[loss_name]
        cur_val = None
        for x, y in zip(weight_dict['x'], weight_dict['y']):
            if x <= epoch:
                cur_val = y
            else:
                break
        return cur_val

    def _loss_kl(self, src, target, T=1):
        return F.kl_div(F.log_softmax(src / T, dim=1), F.softmax(target.detach() / T, dim=1),
                        reduction='batchmean') * T * T


if __name__ == '__main__':
    main()
