import argparse
import ssl
from abc import ABC
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torchvision.transforms as T
import random
import warnings
from tqdm import tqdm
from typing import List, Tuple
from os import mkdir, remove
from os.path import exists
import matplotlib.pyplot as plt
import torch.nn.functional as F
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet, SplitOmniglot
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience, NCScenario
from collections import Counter
from collections import defaultdict

from CustomDataset import CustomDataset
from temp_global import Global_T, CosineDecay, LinearDecay, DistillKL

canshu = {"cosine_decay": True,
          "decay_max": 0,
          "decay_min": -1,
          "decay_loops": 10,
          "t_start": 1,
          "t_end": 20,
          }
if canshu["cosine_decay"]:
    gradient_decay = CosineDecay(max_value=canshu["decay_max"], min_value=canshu["decay_min"],
                                 num_loops=canshu["decay_loops"])
else:
    gradient_decay = LinearDecay(max_value=canshu["decay_max"], min_value=canshu["decay_min"],
                                 num_loops=canshu["decay_loops"])


def adjust_learning_rate(epoch, learning_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray([30, 50, 100]))
    if steps > 0:
        new_lr = learning_rate * (0.1 ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


paper_name = 'MEC'

parser = argparse.ArgumentParser(f'./{paper_name}.py',
                                 description='Class-Incremental Learning via Knowledge Amalgamation')

parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use', required=False,
                    choices=['usps', 'mnist', 'cifar10', 'cifar100', 'tiny10', 'tiny20', 'omniglot'])
parser.add_argument('--seed', type=int, default=1024, metavar='N', help='Set a seed to compare runs')
parser.add_argument('--cuda', default=True, action='store_true', help='enable  CUDA')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device identifier')

# Teacher models configuration
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2 ** 6, help='Batch size for base model training',
                    choices=[2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8])
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per task (base learning)',
                    choices=[1, 5, 10, 50, 100, 150, 200, 400, 500, 1000])
parser.add_argument('--force_base_retraining', type=bool, default=False, help='Force base model retraining')

# Amalgamation configuration
parser.add_argument('--cfl_lr', type=float, default=None, help='Common feature amalgamation learning rate')
parser.add_argument('--amalgamation_strategy', type=str, default='all_together', help='Amalgamation Strategy',
                    choices=['all_together', 'one_at_a_time'])
parser.add_argument('--amalgamation_epochs', type=int, default=100, help='Amalgamation epochs',
                    choices=[1, 10, 30, 100, 160, 200, 500, 1000])

# Memory strategy configuration
parser.add_argument('--memory_strategy', type=str, default='grow', help='Memory Strategy',
                    choices=['fixed', 'grow'])
parser.add_argument('--memory_budget', type=int, default=5000, help='Memory Budget',
                    choices=[0, 10, 100, 200, 300, 400, 500, 1000, 2000, 3000,5000])
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha')


# Helpers
def enum(**enums):
    return type('Enum', (), enums)


FIELD = enum(EXT_IDX='last_searched_external_idx',
             INT_IDX='last_searched_internal_idx',
             N_ELEM='n_elem',
             CLASSES_LIST='classes_list')


class AverageTracker():
    FIELD = enum(VALUE='value',
                 COUNT='count')

    def __init__(self):
        self.book = dict()

    def reset(self, key: str = None) -> None:
        item = self.book.get(key, {})
        if key is None:
            self.book.clear()
        else:
            item[self.FIELD.VALUE] = 0.
            item[self.FIELD.COUNT] = 0
            self.book[key] = item

    def update(self, key: str, val: torch.Tensor) -> None:
        item = self.book.get(key, None)
        if item is None:
            self.reset(key)
            self.update(key, val)
        else:
            item[self.FIELD.VALUE] += val
            item[self.FIELD.COUNT] += 1

    def get(self, key: str) -> float:
        item = self.book.get(key, None)
        assert item is not None
        return item[self.FIELD.VALUE] / float(item[self.FIELD.COUNT]) if float(item[self.FIELD.COUNT]) > 0. else 0.

    def count(self, key: str) -> float:
        item = self.book.get(key, None)
        assert item is not None
        return item[self.FIELD.COUNT]


# Code
class CommonFeatureLearningLoss(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CommonFeatureLearningLoss, self).__init__()
        self.beta = beta

    # 特征约束，参数层面
    # 学生和教师之间的特征，位于公共空间
    # 公共空间与原来之间的特征
    # aligned_t, extend_t, ft_all
    def forward(self, aligned_t: torch.Tensor, extend_t: torch.Tensor, ft_all: torch.Tensor,
                sf: torch.Tensor) -> torch.Tensor:
        kl_loss = 0.0
        mse_loss = 0.0

        # import pdb
        # pdb.set_trace()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # kl_loss = torch.sum(torch.nn.functional.mse_loss(aligned_t, torch.squeeze(sf)))
            kl_loss += (torch.nn.functional.kl_div(torch.log_softmax(aligned_t, dim=1), torch.softmax(sf, dim=1)))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mse_loss = torch.sum(torch.nn.functional.mse_loss(extend_t, ft_all, ))

        # for i in range(len(ft_)):
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings('ignore')
        #         mse_loss += (torch.nn.functional.mse_loss(ft_[i], ft[i]))
        # mse_loss += (torch.nn.functional.kl_div(torch.log_softmax(ft_[i], dim=1), torch.softmax(ft[i], dim=1)))
        # mse_loss += (torch.nn.functional.kl_div(torch.log_softmax(ft_[i], dim=1), torch.softmax(ft[i], dim=1)))
        # print("kl_loss",kl_loss)
        # print("mse_loss",mse_loss)
        # return kl_loss
        return kl_loss + self.beta * mse_loss


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=inplanes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(in_channels=planes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)
        self.downsample = None
        if stride > 1 or inplanes != planes:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels=inplanes,
                                                                  out_channels=planes,
                                                                  kernel_size=(1, 1),
                                                                  stride=stride,
                                                                  bias=False)
                                                  )

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class CommonFeatureBlocks(torch.nn.Module):
    def __init__(self, n_student_channels: int, n_teacher_channels: List[int], n_hidden_channel: int):
        super(CommonFeatureBlocks, self).__init__()

        ch_s = n_student_channels  # Readability
        ch_ts = n_teacher_channels  # Readability
        ch_h = n_hidden_channel  # Readability
        # c=int(ch_h/2)
        print(ch_s, ch_ts, ch_h)
        # self.key = torch.zeros(512,1).to(device)
        self.key = torch.rand(512, 1).to(device)
        # self.align_t = torch.nn.Sequential(nn.Linear(ch_h*len(ch_ts), int(0.5*ch_h*len(ch_ts))),
        #                         torch.nn.BatchNorm1d(int(0.5*ch_h*len(ch_ts))),
        #                         torch.nn.Linear(int(0.5*ch_h*len(ch_ts)), 2048),
        #                         torch.nn.BatchNorm1d(2048),
        #                         torch.nn.Linear(2048, ch_h),

        #                                   torch.nn.ReLU(inplace=True))
        self.align_t = torch.nn.Sequential(nn.Linear(ch_h * len(ch_ts), 2048),
                                           torch.nn.BatchNorm1d(2048),
                                           torch.nn.Linear(2048, 2048),
                                           torch.nn.BatchNorm1d(2048),
                                           torch.nn.Linear(2048, ch_h),
                                           torch.nn.ReLU(inplace=True))
        self.extend = torch.nn.Sequential(nn.Linear(ch_h, ch_h * len(ch_ts)),

                                          torch.nn.ReLU(inplace=True))
        # self.extend = torch.nn.Sequential(nn.Linear(ch_h, 2048),
        #                         torch.nn.BatchNorm1d(2048),
        #                         nn.Linear(2048, ch_h*len(ch_ts)),
        #                         torch.nn.ReLU(inplace=True))

        for layer in self.align_t:
            if isinstance(layer, nn.Linear):
                # 使用默认的初始化方法对全连接层进行初始化
                nn.init.xavier_uniform_(layer.weight)  # 使用Xavier初始化权重
                nn.init.zeros_(layer.bias)
        for layer in self.extend:
            if isinstance(layer, nn.Linear):
                # 使用默认的初始化方法对全连接层进行初始化
                nn.init.xavier_uniform_(layer.weight)  # 使用Xavier初始化权重
                nn.init.zeros_(layer.bias)

    def forward(self, fs: torch.Tensor, ft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ft_all = torch.cat(ft, dim=1)
        ft_all = torch.squeeze(ft_all)
        # +++

        # key = torch.zeros(512,1).to(device)
        score = []
        for i in range(0, len(ft_all[0]), 512):
            score.append(torch.matmul(ft_all[:, i:i + 512], self.key).to(device))
        score = torch.cat(score, dim=1)
        score = torch.softmax(score, dim=1)
        # print("score",len(score[0]))
        q = []
        for i in range(len(ft_all)):
            for j in range(0, len(ft_all[0]), 512):
                ft_all[i][j:j + 512] = ft_all[i][j:j + 512] + ft_all[i][j:j + 512] * score[i][int(j / 512)]
                # ft_all[i][j:j+512] = ft_all[i][j:j+512]*score[i][int(j/512)]
                # print(score[i][int(j/512)])
                # if score[i][int(j/512)] > 0.48:
                # print(score[i][int(j/512)])

            # max_index = np.argmax(score[i].cpu().numpy())
            # q.append(ft_all[i][int(max_index*512):max_index*512+512])
            # if score[i][int(j/512)] > 0.5:
        #             ft_all[i][j:j+512] = ft_all[i][j:j+512]*score[i][int(j/512)]        #
        # else:
        #             ft_all[i][j:j+512] = ft_all[i][j:j+512]*(score[i][int(j/512)]+0.8)
        # +++
        # q = torch.stack(q,dim=0)
        # aligned_t = q
        aligned_t = self.align_t(ft_all)
        extend_t = self.extend(aligned_t)
        # extend_t = self.extend(q)
        # import pdb
        # pdb.set_trace()
        return aligned_t, extend_t, ft_all


class MySimpleModel(torch.nn.Module):
    @property
    def features(self) -> torch.Tensor:
        assert self.handles is not None
        return self.resnet.avgpool.output

    @property
    def feature_dimension(self) -> int:
        return self.resnet.layer4[-1].conv2.out_channels

    @property
    def soft_output(self) -> torch.Tensor:
        return self.fc.output

    @property
    def n_output(self) -> int:
        return self.fc[-1].out_features

    def __init__(self, n_output: int):
        super(MySimpleModel, self).__init__()
        self.handles = {}
        print("n_output", n_output)
        self.resnet = torchvision.models.resnet34(pretrained=True)
        # +++
        # self.resnet = torchvision.models.resnet34(pretrained=False)
        # state_dict = torch.load(r"/home/zanghan/CFA-main/statenew/cfa_cifar100_3")
        # self.resnet.load_state_dict(torch.load(f'./statenew/cfa_cifar100_7'), False)
        # ++++
        self.resnet.fc_backup = self.resnet.fc
        self.resnet.fc = torch.nn.Sequential()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.resnet.fc_backup.in_features,
                            self.resnet.fc_backup.in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.resnet.fc_backup.in_features // 2,
                            n_output)
        )

    def register_hooks(self) -> None:
        def forward_hook(module: torch.nn.modules.container.Sequential, _: tuple, output: torch.Tensor):
            module.output = output

        self.handles['avg_conv_layer'] = self.resnet.avgpool.register_forward_hook(forward_hook)
        self.handles['conv_layer'] = self.resnet.layer4.register_forward_hook(forward_hook)
        self.handles['fc_layer'] = self.fc.register_forward_hook(forward_hook)

    def remove_hooks(self) -> None:
        assert self.handles is not None
        for k, v in self.handles.items():
            self.handles[k].remove()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        # print("softmax",torch.max(torch.softmax(x, dim=1),1)[0])
        # print("index",torch.softmax(x, dim=1).argmax(1))
        return torch.softmax(x, dim=1).argmax(1)


def save_load_best_model(model: MySimpleModel, experience: NCExperience, is_train=True, pbar=None) -> Tuple[
    MySimpleModel, float]:
    n_task = experience.current_experience
    path = f'./add'
    state_path = f'{path}/cfa_{args.dataset}_{n_task + 1}'

    if not exists(path):
        mkdir(path)
    if not exists(state_path):
        torch.save(model.state_dict(), state_path)
    assert exists(path)
    assert exists(state_path)

    with torch.no_grad():
        model.eval()
        corrects = 0
        total_task = 0

        if not is_train:
            model.load_state_dict(torch.load(state_path))

        data_loader = torch.utils.data.DataLoader(experience.dataset, batch_size=args.batch_size, shuffle=True)
        # experience.classes_in_this_experience.sort()
        for _, data in enumerate(data_loader):
            x = data[0].to(device)
            y = (data[1] - min(experience.classes_in_this_experience)).to(device)
            # print("x,y",x,y)
            # y = (experience.classes_in_this_experience[data[1]] - min(experience.classes_in_this_experience)).to(device)
            total_task += len(x)
            corrects += int(sum(model.predict(x) == y))
        accuracy = (corrects / total_task) if total_task > 0 else 0

        if is_train:
            description = f'Train accuracy for base task {n_task + 1}: {accuracy * 100:.2f}% ({corrects}/{total_task})'

            if pbar is None:
                print(description)
            else:
                pbar.set_description_str(description)
            torch.save(model.state_dict(), state_path)
            model.train()
        else:
            model.load_state_dict(torch.load(state_path))

    return model, accuracy


def amalgamate(teachers: List[MySimpleModel], data_array: List = [], labels: List = [],
               train: NCScenario = None, test: NCScenario = None, epochs: int = 100) -> Tuple[
    MySimpleModel, List[int], List[int], np.ndarray, np.ndarray]:
    def memory_keys(all_data: NCScenario, teachers: List[MySimpleModel], labels: List[int], idx: int,
                    previous_data_idxs: List[int] = None) -> List[int]:
        if args.memory_strategy == 'grow':
            print("len(labels[idx])", len(labels[idx]))
            n_elem = args.memory_budget // len(labels[idx])
            # n_elem = 20
            return get_mean_exemplar_keys(all_data, teachers[idx], labels[idx], n_elem, previous_data_idxs)
        elif args.memory_strategy == 'fixed':
            n_elem = args.memory_budget // (sum(len(v) for v in labels) -10)
            return get_mean_exemplar_keys(all_data, teachers[idx], labels[idx], n_elem, previous_data_idxs)

    def get_mean_exemplar_keys(all_data: NCScenario, teacher: MySimpleModel, labels: List[int], n_elem_per_class: int,
                               previous_data_idxs: List[int] = None) -> List[int]:
        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        current_batch_size = 0
        n_samples = 0
        label_mean = {}
        print("first")
        # ADD
        en_loader = torch.utils.data.DataLoader(all_data.dataset, batch_size=args.batch_size, shuffle=False)
        labels_count = Counter()
        with torch.no_grad():
            if previous_data_idxs is None:
                for _, data in tqdm(enumerate(en_loader)):
                    x = data[0].to(device)
                    y = (data[1]).to(device)
                    teacher(x)
                    key = y.cpu().numpy()
                    labels_count.update(key)
                    value = teacher.features
                    for k, v in zip(key, value):
                        v = v.unsqueeze(0)
                        if k in label_mean.keys():
                            label_mean[k] = label_mean[k] + v
                        else:
                            label_mean[k] = v
                for k in labels_count.keys():
                    label_mean[k] = label_mean[k] / labels_count[k]
            else:
                for _, idx in enumerate(previous_data_idxs):
                    x, y, _ = all_data[0].dataset[idx]
                    if current_batch_size < args.batch_size:
                        if y == label:
                            batch_sample[current_batch_size] = x
                            current_batch_size += 1
                    elif current_batch_size == 0:
                        break
                    elif current_batch_size > 0:
                        batch_sample = batch_sample[:current_batch_size].to(device)
                        teacher(batch_sample)
                        label_mean[label] += sum(teacher.features, 1).unsqueeze(0)
                        n_samples += current_batch_size
                        current_batch_size = 0

                # label_mean[label] /= n_samples
                # n_samples = 0

        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        batch_idx = np.empty(args.batch_size, dtype=int)
        current_batch_index = 0
        label_idx_distance = {}
        label_idx = {}
        print("secend")

        # 数据集 【0】所有
        en_loader = torch.utils.data.DataLoader(all_data.dataset, batch_size=args.batch_size, shuffle=False)
        for i in labels_count.keys():
            label_idx_distance[i] = {}
            # label_idx[i] = []
        with torch.no_grad():
            id_all_data = 0
            for _, data in tqdm(enumerate(en_loader)):
                x = data[0].to(device)
                y = (data[1]).to(device)
                teacher(x)
                key = y.cpu().numpy()
                value = teacher.features
                for k, v in zip(key, value):
                    label_idx_distance[k][id_all_data] = float(torch.dist(value, label_mean[k], 2))
                    id_all_data += 1
            for label in labels_count.keys():
                # label_idx[label] = [idx for idx, _ in sorted(label_idx_distance[label].items(), key=lambda x: x[1])][
                #                    :n_elem_per_class]
                # 对于最新类，我们使用全部数据，对于旧类，我们适应fixed 2000
                if label in list(labels_count.keys())[-10:]:
                    label_idx[label] = [idx for idx, _ in sorted(label_idx_distance[label].items(), key=lambda x: x[1])][
                                   :500]
                else:
                    label_idx[label] = [idx for idx, _ in sorted(label_idx_distance[label].items(), key=lambda x: x[1])][
                                   :n_elem_per_class]
        return np.concatenate([v for k, v in label_idx.items()], 0)

    def get_conf_keys(all_data: NCScenario, teacher: MySimpleModel, labels: List[int],
                      n_elem: int = args.memory_budget) -> List[int]:
        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        batch_idx = np.empty(args.batch_size, dtype=int)
        current_batch_size = 0
        conf = {}
        for label in labels:
            conf[label] = {}
            for idx, [x, y, _] in enumerate(all_data[0].dataset):
                if x is None:
                    break

                if current_batch_size < args.batch_size and x is not None:
                    if y == label:
                        batch_sample[current_batch_size] = x
                        batch_idx[current_batch_size] = idx
                        conf[label][idx] = 0
                        current_batch_size += 1
                elif x is None and current_batch_size == 0:
                    break
                elif current_batch_size > 0 or x is None:
                    batch_sample = batch_sample[:current_batch_size].to(device)
                    batch_idx = batch_idx[:current_batch_size]

                    soft_top_2 = torch.softmax(teacher(batch_sample), 1).topk(2)[0].tolist()
                    for i, j in enumerate(batch_idx):
                        conf[label][j] = soft_top_2[i][0] - soft_top_2[i][1]

                    batch_sample = torch.empty((args.batch_size, 3, 224, 224))
                    batch_idx = np.empty(args.batch_size, dtype=int)
                    current_batch_size = 0

        idxs = []
        for label in conf.keys():
            idxs = idxs + list(dict(sorted(conf[label].items(),
                                           key=lambda x: x[1],
                                           reverse=True)).keys())[:(n_elem // len(labels))]

        return idxs[:n_elem]

    student = MySimpleModel(sum([teacher.n_output for teacher in teachers])).to(device)
    # +++
    # if(len(teachers)>3):
    #     student = teachers[3].to(device)
    # else:
    #     student = MySimpleModel(sum([teacher.n_output for teacher in teachers])).to(device)
    # ++++
    cfl_blk = CommonFeatureBlocks(student.feature_dimension,
                                  [teacher.feature_dimension for teacher in teachers],
                                  #   [teachers[0].feature_dimension, teachers[1].feature_dimension],
                                  int(sum([teacher.feature_dimension for teacher in teachers]) / len(teachers))).to(
        device)

    cfl_lr = args.lr * 10 if args.cfl_lr is None else args.cfl_lr
    # cfl_lr = args.lr if args.cfl_lr is None else args.cfl_lr
    mlp_net = Global_T()
    mlp_net.train()
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    params_10x = [param for name, param in student.named_parameters() if 'fc' in name]
    params_1x = [param for name, param in student.named_parameters() if 'fc' not in name]
    # optimizer = torch.optim.Adam([{'params': params_1x,            'lr': args.lr},
    #                               {'params': params_10x,           'lr': args.lr * 10},
    #                               {'params': cfl_blk.parameters(), 'lr': cfl_lr},
    #                               {'params': mlp_net.parameters(), 'lr': cfl_lr*100}])
    ###++++
    optimizer = torch.optim.SGD([{'params': params_1x, 'lr': args.lr},
                                 {'params': params_10x, 'lr': args.lr * 10},
                                 {'params': cfl_blk.parameters(), 'lr': cfl_lr},
                                 {'params': mlp_net.parameters(), 'lr': cfl_lr }],
                                momentum=0.9,
                                weight_decay=5e-4)
    ###+++

    student.train()
    [teacher.register_hooks() for teacher in teachers]
    [teacher.eval() for teacher in teachers]
    student.register_hooks()
    average_tracker = AverageTracker()

    common_feature_learning_criterion = CommonFeatureLearningLoss().to(device)
    # print("len(teachers)", len(teachers))
    print('Adjusting replay memory - sorry the delay, this part of the code is not optimized')
    data_idx = []
    train_stream_split, _ = load_dataset(args.dataset)
    data_array_label = [None] * len(data_array)
#++++ address
    save_dir = './add'
    
    # for idx, data in enumerate(data_array):
    #     data_idx.append(memory_keys(train_stream_split[idx], teachers, labels, idx, data))
        # ++++
        # file_path = os.path.join(save_dir, args.dataset + str(idx) + '.npy')
        # if os.path.exists(file_path):
        #     MK_numpy = np.load(file_path)
        # else:
        #     MK_numpy = memory_keys(train_stream_split[idx], teachers, labels, idx, data)
        #     np.save(file_path, MK_numpy)
        # data_idx.append(MK_numpy)
        # ++++
    for idx, data in enumerate(data_array):
        if os.path.exists(str(idx) + '.npy'):
            MK_numpy = np.load(str(idx) + '.npy')
        else:
            MK_numpy = memory_keys(train_stream_split[idx], teachers, labels, idx, data)
            np.save(str(idx) + '.npy', MK_numpy)
        data_idx.append(MK_numpy)
        # data_array[idx] = torch.stack([train_stream_split[idx].dataset[idx_][0] for idx_ in data_idx[idx]])
        data_array[idx] = torch.stack([train_stream_split[idx].dataset[idx_][0] for idx_ in data_idx[idx]])
        data_array_label[idx] = torch.stack(
            [torch.tensor(train_stream_split[idx].dataset[idx_][1]) for idx_ in data_idx[idx]])
    # combined_list = list(zip(data_array, data_array_label))

    print('Replay memory adjusted')

    # all_data = torch.cat([data for data in data_array])

    
    all_data0 = torch.cat([data for data in data_array])
    
    all_data_random = torch.rand_like(all_data0)* 0.15
    all_data_random = all_data0+all_data_random
    
    all_data1 = torch.cat([data1 for data1 in data_array_label])
    
    paired_data = [(all_data0[i], all_data1[i]) for i in range(len(all_data0))]
    paired_data_random = [(all_data_random[i], all_data1[i]) for i in range(len(all_data_random))]
    paired_data = paired_data+paired_data_random

    student.eval()
    with torch.no_grad():
        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        b_accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for _, [x, y, _] in enumerate(test[0].dataset):
            if int(y) not in labels_:
                continue

            label = torch.tensor(labels_.index(y)).to(device)
            sample = x.view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                if label in task_labels:
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            b_accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0
    # CSdataset =  CustomDataset(all_data)
    CSdataset = CustomDataset(paired_data)
    en_loader = torch.utils.data.DataLoader(CSdataset, batch_size=args.batch_size, shuffle=True)
    student.train()
    # import pdb
    # pdb.set_trace()
    with tqdm(unit='Epoch', total=epochs) as pbar:
        while pbar.n < epochs:
            # adjust_learning_rate(pbar.n, args.lr, optimizer)
            number = 0
            # difnumber = 0
            average_tracker.reset()
            # batch_sample = torch.empty((args.batch_size, 3, 224, 224))
            # current_batch_index = 0
            # for i in range(len(teachers)):
            #     Diff[i]={}
            Diff = {i: [] for i in range(len(teachers))}
            for _, batch_sample1 in enumerate(en_loader):
                dif = {i: [] for i in range(len(teachers))}
                max_ind = []
                max_i = []
                batch_sample = batch_sample1[0].to(device)
                # sample_label = batch_sample[1].to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    [teacher(batch_sample) for teacher in teachers]
                    ##注释
                    # for i in range(len(teachers)):
                    #     for j in range(len(teachers[i].soft_output)):
                    #         max_value = torch.max(teachers[i].soft_output[j])
                    #         # squared_differences = torch.square(teachers[i].soft_output[j] - max_value)
                    #         squared_differences = torch.pow((teachers[i].soft_output[j] - max_value),2)
                    #         average_squared_difference = torch.mean(squared_differences)
                    #         Diff[i].append(average_squared_difference)
                    #         dif[i].append(average_squared_difference)
                    # # 取最大值
                    # for i in range(len(dif[0])):
                    #     max_in = 0
                    #     max_v = dif[0][i]
                    #     for j in range(len(dif)):
                    #         # print(dif[j][i])
                    #         if dif[j][i] > max_v:
                    #             max_v=dif[j][i]
                    #             max_in = j
                    #         # print(max_in)
                    #     # max_ind对应任务数（也就是行）
                    #     max_ind.append(max_in)
                    #     # max_i对应样本（也就是列）
                    #     max_i.append(i)
                    # # import pdb
                    # # pdb.set_trace()
                    # for i in range(len(batch_sample1[1])):
                    #     if max_ind[i] != (batch_sample1[1][i]//10):
                    #         # m1=torch.argmax(teachers[max_ind[i]].soft_output[max_i[i]])
                    #         # m2=torch.argmax(teachers[batch_sample1[1][i]//10].soft_output[max_i[i]])
                    #         # teachers[max_ind[i]].soft_output[max_i[i]][m1],teachers[batch_sample1[1][i]//10].soft_output[max_i[i]][m2] = teachers[batch_sample1[1][i]//10].soft_output[max_i[i]][m2],teachers[max_ind[i]].soft_output[max_i[i]][m1]
                    #         # print("max_ind[i]",max_ind[i])
                    #         # for j in range(len(dif)):
                    #         #     print("dif",dif[j][max_i[i]])
                    #         #     print("teachers.soft_output",teachers[j].soft_output[max_i[i]])
                    #         # print(batch_sample1[1][i])
                    #         difnumber+=1
                    ##注释
                    # import pdb
                    # pdb.set_trace()
                    teacher_soft = torch.cat(tuple([teacher.soft_output for teacher in teachers]), dim=1)
                    # print("teacher_soft",teacher_soft)
                    # print("teacher_soft index",torch.argmax(teacher_soft,dim = 1))
                    # print("teacher_soft label",batch_sample1[1].to(device))
                    ##注释
                    for i in range(len(batch_sample1[1])):
                    #     # torch.argmax(teacher_soft_test,dim = 1)[i]为predict
                        if torch.argmax(teacher_soft,dim = 1)[i] != batch_sample1[1][i]:
                    # teacher_soft[i][torch.argmax(teacher_soft,dim = 1)[i]],teacher_soft[i][batch_sample1[1][i]] = teacher_soft[i][batch_sample1[1][i]],teacher_soft[i][torch.argmax(teacher_soft,dim = 1)[i]]

                    # m1=torch.argmax(teachers[max_ind[i]].soft_output[max_i[i]])
                    #     m2=torch.argmax(teachers[batch_sample1[1][i]//10].soft_output[max_i[i]])
                    # teachers[max_ind[i]].soft_output[max_i[i]][m1],teachers[batch_sample1[1][i]//10].soft_output[max_i[i]][m2] = teachers[batch_sample1[1][i]//10].soft_output[max_i[i]][m2],teachers[max_ind[i]].soft_output[max_i[i]][m1]
                            number+=1
                    # for i in range(len(batch_sample1[1])):
                    #     if max_ind[i] != (torch.argmax(teacher_soft,dim = 1)[i]//10):
                    #         m = torch.argmax(teachers[max_ind[i]].soft_output[i])
                    #         teacher_soft[i][torch.argmax(teacher_soft,dim = 1)[i]],teacher_soft[i][m] =teacher_soft[i][m],teacher_soft[i][torch.argmax(teacher_soft,dim = 1)[i]]
                    ##注释

                #     teacher_soft = teacher_soft_test
                #     teacher_soft = torch.cat(tuple([teacher.soft_output for teacher in teachers]), dim=1)

                student(batch_sample)
                student_soft = student.soft_output

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    cos_value = -gradient_decay.get_value(pbar.n + 1)
                    
                    temp = mlp_net(teacher_soft, student_soft, cos_value)
                    temp = torch.sigmoid(temp)
                    # temp = canshu["t_start"] + canshu["t_end"] * torch.sigmoid(temp)
                    # temp = torch.log(torch.sigmoid(temp)+1e-8)
                    # print("000",temp)
                    # temp = 1/temp
                    # temp = (temp/(1+temp))/10
                    # temp = 0.01
                    # temp = torch.sigmoid(temp)
                    temp = temp.cuda()
                    # calculate loss
                    
                    T = temp
                    # print("111",temp)
                    # ++
                    
                    loss_cls = criterion_cls(student_soft, batch_sample1[1].to(device))
                    # ++
                    epsilon = 1e-9
                    # print(F.log_softmax(student_soft / T, dim=1),F.softmax(teacher_soft / T, dim=1))
                    cross_entropy_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_soft / T, dim=1)+epsilon,
                                                                             F.softmax(teacher_soft / T, dim=1)+epsilon) * T * T
                #     t_s = torch.softmax(teacher_soft, dim=1)
                #     s_s = torch.log_softmax(student_soft, dim=1)
                #     # import pdb
                #     # pdb.set_trace()
                #     # s_s = s_s[t_s>0.8]
                #     # t_s = t_s[t_s>0.8]
                #     cross_entropy_loss = torch.nn.functional.kl_div(s_s,t_s)

                # # aligned_t, extend_t, ft_all
                # aligned_t, extend_t, ft_all = cfl_blk(student.features, [teacher.features for teacher in teachers])
                # # 可以将空间对齐
                # # common_features_loss = 0.1*common_feature_learning_criterion(aligned_t, extend_t, ft_all, student.features)
                # common_features_loss =common_feature_learning_criterion(aligned_t, extend_t, ft_all,
                #                                                                student.features)
                # if pbar.n>8:

                # loss = args.alpha * cross_entropy_loss + (1 - args.alpha) * common_features_loss
                # else:
                loss = 5*cross_entropy_loss + 5*loss_cls
                # loss = 5*cross_entropy_loss
                # max_grad_norm = 1.0  # 设置梯度裁剪的阈值
                # torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                # for teacher in teachers:
                #     torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_grad_norm)
                loss.backward()
                optimizer.step()

                average_tracker.update('loss', loss.item())
                average_tracker.update('ce', cross_entropy_loss.item())
                average_tracker.update('cf', loss_cls.item())
                # average_tracker.update('cf', common_features_loss.item())

                # description = f'Amalgamating ' \
                #                 f'Loss={average_tracker.get("loss"):.2f} '\
                #                 f'(cross entropy={average_tracker.get("ce")}, '\
                #                 f'common features={average_tracker.get("cf"):.2f})'
                description = f'Amalgamating ' \
                              f'Loss={average_tracker.get("loss"):.2f} ' \
                              f'(cross_entropy_loss={average_tracker.get("ce")}, ' \
                              f'cross entropy={average_tracker.get("cf"):.2f})'
                # description = f'Amalgamating ' \
                #                 f'Loss={average_tracker.get("loss"):.2f} '\
                #                 f'cross entropy={average_tracker.get("ce")}'
                pbar.set_description_str(description)
                pbar.refresh()

            # for j in range(len(Diff[0])):
            #     print("start")
            #     for i in range(len(Diff)):
            #         print("The difference:",Diff[i][j])
            #     print("end")
            # for i in range(len(Diff)):
            #     print(Diff[:,i])
            # print("dif_wrong_number",difnumber)
            # print("dif_wrong_number",'%.5f'%(difnumber/len(all_data0)))
            pbar.update()
        print("wrong_number",number)
        print("wrong_number",'%.5f'%(number/len(all_data0)))

    [teacher.remove_hooks() for teacher in teachers]
    student.remove_hooks()
    student.eval()
    with torch.no_grad():

        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for _, [x, y, _] in enumerate(test[0].dataset):
            if int(y) not in labels_:
                continue

            label = torch.tensor(y).to(device)
            sample = x.view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                # print("labels",labels)
                if label in task_labels:
                    # print("task_labels",task_labels)
                    # print("label  pred:",label, pred)
                    # print("abs",abs(pred-label))
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            print("corrects[idx]", idx, corrects[idx])
            accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0
    return student, [data for d in data_idx for data in d], accuracy, b_accuracy


def load_dataset(dataset: str, force_unique_task: bool = False) -> Tuple[NCScenario, NCScenario]:
    if dataset in ['mnist', 'usps', 'omniglot']:
        transforms = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor()])

        if dataset == 'mnist':
            args.n_tasks = 1 if force_unique_task else 5
            # import pdb
            # pdb.set_trace()
            # args.n_tasks = 2
            data = SplitMNIST(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 10),
                              train_transform=transforms, eval_transform=transforms)
        # args.n_tasks = 5
        # data10 = SplitMNIST(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 10),
        #                 train_transform=transforms, eval_transform=transforms)
        elif dataset == 'usps':
            args.n_tasks = 1 if force_unique_task else 5
            usps_train = torchvision.datasets.USPS(root='./data', train=True, download=True)
            usps_test = torchvision.datasets.USPS(root='./data', train=False, download=True)

            data = nc_benchmark(usps_train, usps_test, n_experiences=args.n_tasks, seed=args.seed, task_labels=True,
                                fixed_class_order=range(0, 10), train_transform=transforms, eval_transform=transforms)
        elif dataset == 'omniglot':
            args.n_tasks = 1 if force_unique_task else 241
            data = SplitOmniglot(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 964),
                                 train_transform=transforms, eval_transform=transforms)

    elif dataset in ['cifar10', 'cifar100', 'tinyImageNet10', 'tiny10', 'tinyImageNet20', 'tiny20']:
        transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        if dataset == 'cifar10':
            args.n_tasks = 1 if force_unique_task else 2
            data = SplitCIFAR10(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 10),
                                train_transform=transforms, eval_transform=transforms)
        elif dataset == 'cifar100':
            args.n_tasks = 1 if force_unique_task else 10
            data = SplitCIFAR100(n_experiences=args.n_tasks, first_exp_with_half_classes=False, seed=args.seed,
                                 fixed_class_order=range(0, 100),
                                 train_transform=transforms, eval_transform=transforms)

        elif dataset in ['tinyImageNet10', 'tiny10']:
            args.n_tasks = 1 if force_unique_task else 10
            data = SplitTinyImageNet(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 200),
                                     train_transform=transforms, eval_transform=transforms)
        elif dataset in ['tinyImageNet20', 'tiny20']:  # 200
            args.n_tasks = 1 if force_unique_task else 20
            data = SplitTinyImageNet(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 200),
                                     train_transform=transforms, eval_transform=transforms)
    return data.train_stream, data.test_stream


def main(args):
    path = f'./add'
    state_path = f'{path}/{paper_name}_{args.dataset}'
    is_training_base_model = False
    # Prepare data
    train_stream, test_stream = load_dataset(args.dataset)

    # train_stream = train_stream[0:2] # TODO
    # test_stream = test_stream[0:2] # TODO

    # Training base model
    if args.force_base_retraining is not None and args.force_base_retraining:
        for i in range(args.n_tasks):

            if exists(f'{state_path}_{i + 1}'):
                remove(f'{state_path}_{i + 1}')
    if exists(path):
        for i in range(args.n_tasks):
            if not exists(f'{state_path}_{i + 1}'):
                is_training_base_model = True
                break
    else:
        is_training_base_model = True

    if is_training_base_model:
        print('Training base model')
        for experience in train_stream:
            model = MySimpleModel(len(experience.classes_in_this_experience)).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # experience.classes_in_this_experience.sort()
            with tqdm(unit='Epoch', total=args.epochs) as pbar:
                train_loader = torch.utils.data.DataLoader(experience.dataset, batch_size=args.batch_size, shuffle=True)
                while pbar.n < args.epochs:
                    model.train()
                    for _, data in tqdm(enumerate(train_loader)):
                        # 数据中的样本
                        x = data[0].to(device)
                        # print("1111",data[1],int(data[1]).type)
                        # import pdb
                        # pdb.set_trace()
                        # 数据中的标签
                        y = (data[1] - min(experience.classes_in_this_experience)).to(device)
                        # y = (experience.classes_in_this_experience[data[1]] - min(experience.classes_in_this_experience)).to(device)
                        optimizer.zero_grad()
                        output = model(x)
                        criterion(output, y).backward()
                        optimizer.step()
                    save_load_best_model(model, experience, pbar=pbar)
                    pbar.update()

    base_models = []
    experiences = []
    # basemodel = []
    print('Base model performance')

    for experience in test_stream:
        # experience.classes_in_this_experience.sort()

        model = MySimpleModel(len(experience.classes_in_this_experience)).to(device)
        model, accuracy = save_load_best_model(model, experience, False)
        print(
            f'Test accuracy for base task {experience.current_experience + 1} {experience.classes_in_this_experience}: {accuracy * 100:.2f}')
        if experience.current_experience == 0:
            accuracy_0 = accuracy
        # basemodel.append(model)
        base_models.append(model.cpu())
        experiences.append(experience)
    for i in range(len(experiences)):
        experiences[i].classes_in_this_experience.sort()
        print("experiences", i, experiences[i].classes_in_this_experience)
    # ++++
    print("have a try")
    feature = {}
    softlabel = {}
    # basemode = {}
    # for i in range (10):
    #     feature[i] = {}
    #     softlabel[i] = {}
    #     # base_models[i].register_hooks()
    #     # basemode[i] = MySimpleModel(10).to(device)
    #     path = f'./statetest'
    #     state_path = f'{path}/cfa_{args.dataset}_{i+1}'
    #     model.load_state_dict(torch.load(state_path))
    # for i in range(10):
    #     # teachers=[base_models[idx].to(device) for idx in range(n_task)]
    #         data_loader = torch.utils.data.DataLoader(train_stream[i].dataset, batch_size=args.batch_size, shuffle=True)
    #         # experience.classes_in_this_experience.sort()
    #         for _, data in enumerate(data_loader):
    #             x = data[0].to(device)
    #             y = (data[1] - min(experience.classes_in_this_experience)).to(device)
    #             y_label = data[1].cpu().numpy()
    #             for j in range(10):
    #                 # model = basemodel[j]
    #                 model= base_models[j].to(device)
    #                 model.register_hooks()
    #                 model(x)
    #                 f = model.features
    #                 f=f.cpu()
    #                 f=f.detach()
    #                 f=f.numpy()
    #                 # f=f.detach().numpy()
    #                 l = model.logit(x)
    #                 l=l.cpu()
    #                 l= l.detach()
    #                 l= l.numpy()
    #                 # l=l.detach().numpy()
    #                 for k in range(len(y_label)):
    #                     if int(y_label[k]) not in softlabel[j].keys():
    #                         softlabel[j][y_label[k]]=[]
    #                         softlabel[j][y_label[k]].append(l[k].squeeze())
    #                     else:
    #                         softlabel[j][y_label[k]].append(l[k].squeeze())

    #                     if int(y_label[k]) not in feature[j].keys():
    #                         feature[j][y_label[k]]=[]
    #                         feature[j][y_label[k]].append(f[k].squeeze())
    #                     else:
    #                         feature[j][y_label[k]].append(f[k].squeeze())
    #             # np.save('feature'+str(idx)+'.npy', f)
    #             model.remove_hooks()
    # for i in range(10):
    #     for k in feature[i].keys():
    #         v = np.array(feature[i][k])
    #         np.save('feature/feature_'+str(i)+'_'+str(k)+'.npy',v)
    #     for q in softlabel[i].keys():
    #         m = np.array(softlabel[i][q])
    #         np.save('softlabel/softlabel_'+str(i)+'_'+str(q)+'.npy',m)
    # base_models[i].remove_hooks()
    # ++++
    # import pdb
    # pdb.set_trace()
    args.n_tasks_original = args.n_tasks
    n_tasks = args.n_tasks
    accuracies = np.zeros((n_tasks, n_tasks))
    b_accuracies = np.zeros((n_tasks, n_tasks))
    accuracies[0, 0] = accuracy_0
    train_stream, test_stream = load_dataset(args.dataset, True)

    if args.amalgamation_strategy == 'one_at_a_time':
        amalgamated_model, data, accuracy, b_accuracy = amalgamate(
            teachers=[base_models[0].to(device), base_models[1].to(device)],
            data_array=[None, None],
            labels=[experiences[0].classes_seen_so_far, experiences[1].classes_in_this_experience],
            train=train_stream,
            test=test_stream,
            epochs=args.amalgamation_epochs)
        accuracies[0, 1] = accuracy[0]
        accuracies[1, 1] = accuracy[1]
        b_accuracies[0, 1] = b_accuracy[0]
        b_accuracies[1, 1] = b_accuracy[1]
        if n_tasks > 2:
            for i in range(1, n_tasks):
                amalgamated_model, data, accuracy, b_accuracy = amalgamate(
                    teachers=[amalgamated_model.to(device), base_models[i].to(device)],
                    data_array=[data, None],
                    labels=[experiences[i - 1].classes_seen_so_far, experiences[i].classes_in_this_experience],
                    train=train_stream,
                    test=test_stream,
                    epochs=args.amalgamation_epochs)
                accuracies[i - 1, i] = accuracy[0]
                accuracies[i, i] = accuracy[1]
                b_accuracies[i - 1, i] = b_accuracy[0]
                b_accuracies[i, i] = b_accuracy[1]
    elif args.amalgamation_strategy == 'all_together':
        for n_task in range(2, n_tasks + 1, 1):
            _, _, accuracy, b_accuracy = amalgamate(teachers=[base_models[idx].to(device) for idx in range(n_task)],
                                                    data_array=[None] * n_task,
                                                    labels=[experiences[idx].classes_in_this_experience for idx in
                                                            range(n_task)],
                                                    train=train_stream,
                                                    test=test_stream,
                                                    epochs=args.amalgamation_epochs)
            for i in range(len(accuracy)):
                accuracies[i, n_task - 1] = accuracy[i]
                b_accuracies[i, n_task - 1] = b_accuracy[i]

    print(f'accuracies \n {accuracies}')
    print(f'b_accuracies (random initialization) \n {b_accuracies}')

    acc = np.nanmean(np.where(accuracies != 0, accuracies, np.nan), 0)[-1]
    print(f'ACC: {acc * 100:.2f}%')

    bwt = 0
    for i in range(n_tasks - 1):
        j = {'one_at_a_time': i + 1, 'all_together': -1}
        bwt += accuracies[i, j[args.amalgamation_strategy]] - accuracies[i, i]
    bwt = bwt / (n_tasks - 1)
    print(f'BWT: {bwt * 100:.2f}%')

    fwt = 0
    for i in range(1, n_tasks):
        j = {'one_at_a_time': i, 'all_together': -1}
        fwt += accuracies[i - 1, j[args.amalgamation_strategy]] - b_accuracies[i, i]
    fwt = fwt / (n_tasks - 1)
    print(f'FWT: {fwt * 100:.2f}%')


if __name__ == '__main__':
    args = parser.parse_args()

    # Configure random seed and devices
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_device}' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    print(f'Device: {device}')

    main(args)
