# coding: utf-8
# author: ZhangYuan
# function: slimming
import argparse
from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
import os
from utils.tiny_prune_utils import *
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Slimming pruning")
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--model_def', type=str, default='cfg/yolov3-tiny-mobilenet-small.cfg')
parser.add_argument('--data_config', type=str, default='data/uavcut.data')
parser.add_argument('--model', type=str, default='weights/yolov3_tiny_uavcut_mobilenetsmall.pt')  # γ L1稀疏后的模型
parser.add_argument('--prune_percent', type=float, default=0.4)
opt = parser.parse_args()
print(opt)

# 剪枝率
percent = opt.prune_percent

device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

model = Darknet(opt.model_def).to(device)

# 记录anchor并转换
for item in model.module_defs:
    if item['type'] == 'yolo':
        anchors = item['anchors']
        break
int_anchors = []
for i in anchors:
    for j in i:
        int_anchors.append(int(j))
str_anchors = ','.join(str(i) for i in int_anchors)

# 预训练模型
if opt.model:
    if opt.model.endswith(".pt"):
        model.load_state_dict(torch.load(opt.model, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.model)

data_config = parse_data_cfg(opt.data_config)
print(data_config)

valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

eval_model = lambda model: test(model=model, cfg=opt.model_def, data=opt.data_config)

obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

# 原始模型推理
with torch.no_grad():
    origin_model_metric = eval_model(model)
# 计算模型参数个数
origin_nparameters = obtain_num_parameters(model)
print(origin_nparameters)

CBL_idx, Conv_idx, prune_idx = parse_module_defs(model.module_defs)

# 获取全局bn_weights
bn_weights = gather_bn_weights(model.module_list, prune_idx)

# 查看bn_weights分布
sns.displot(bn_weights)
# plt.show()

# torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
sorted_bn = torch.sort(bn_weights)[0]

# 计算安全剪枝率(每个BN层的gamma的最大值的最小值即为阈值上限)
highest_thres = [model.module_list[idx][1].weight.data.abs().max().item() for idx in prune_idx]
highest_thre = min(highest_thres)

# 找到highest_thre对应的下标对应的百分比
percent_limit = (sorted_bn == highest_thre).nonzero().item() / len(bn_weights)

print(f'Threshold should be less than {highest_thre:.4f}.')
print(f'The corresponding prune ratio is {percent_limit:.3f}.')


# 该函数有很重要的意义：
# ①先用深拷贝将原始模型拷贝下来，得到model_copy
# ②将model_copy中，BN层中低于阈值的α参数赋值为0
# ③在BN层中，输出y=α*x+β，由于α参数的值被赋值为0，因此输入仅加了一个偏置β
# ④很神奇的是，network slimming中是将α参数和β参数都置0，该处只将α参数置0，但效果却很好：其实在另外一篇论文中，已经提到，可以先将β参数的效果移到
# 下一层卷积层，再去剪掉本层的α参数

# 该函数用最简单的方法，让我们看到了，如何快速看到剪枝后的效果


def prune_and_eval(model, sorted_bn, percent=.0):
    model_copy = deepcopy(model)  # 完全拷贝了父对象及其子对象
    thre_index = int(len(sorted_bn) * percent)
    # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    thre = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

    remain_num = 0
    for idx in prune_idx:
        bn_module = model_copy.module_list[idx][1]

        mask = obtain_bn_mask(bn_module, thre, opt.device)

        remain_num += int(mask.sum())
        bn_module.weight.data.mul_(mask)
        # bn_module.bias.data.mul_(mask)

    with torch.no_grad():
        mAP = eval_model(model_copy)[1].mean()

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1 - remain_num / len(sorted_bn):.3f}')
    print(f'mAP of the pruned model is {mAP:.4f}')

    return thre


threshold = prune_and_eval(model, sorted_bn, percent)


# ****************************************************************
# 虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型

def obtain_filters_mask(model, thre, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:

            mask = obtain_bn_mask(bn_module, thre, opt.device).cpu().numpy()
            remain = int(mask.sum())
            pruned = pruned + len(mask) - remain

            if remain == 0:
                print("Channels would be all pruned!")
                raise Exception

        # print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
        #      f'remaining channel: {remain:>4d}')
        else:
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.copy())

    # 因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask


num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)

# CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

# 此时mask对应的γ和β都已屏蔽为0

with torch.no_grad():
    mAP = eval_model(pruned_model)[1].mean()
print('after prune_model_keep_size map is {}'.format(mAP))

# 获得原始模型的module_defs，并修改该defs中的卷积核数量
compact_module_defs = deepcopy(model.module_defs)
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num)

compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
compact_nparameters = obtain_num_parameters(compact_model)


def get_input_mask2(module_defs, idx, CBLidx2mask):
    if idx == 0:
        # 如果是一层卷积层，它的上一通道mask为3（因为图像为三通道）
        return np.ones(3)
    if idx <= 12:
        if module_defs[idx - 2]['type'] == 'convolutional':
            return CBLidx2mask[idx - 2]

    else:
        if module_defs[idx - 1]['type'] == 'convolutional':
            return CBLidx2mask[idx - 1]
        elif module_defs[idx - 1]['type'] == 'shortcut':
            return CBLidx2mask[idx - 2]
        elif module_defs[idx - 1]['type'] == 'route':
            route_in_idxs = []
            for layer_i in module_defs[idx - 1]['layers'].split(","):
                if int(layer_i) < 0:
                    route_in_idxs.append(idx - 1 + int(layer_i))
                else:
                    route_in_idxs.append(int(layer_i))
            if len(route_in_idxs) == 1:
                return CBLidx2mask[route_in_idxs[0]]

            elif len(route_in_idxs) == 2:
                return np.concatenate([CBLidx2mask[route_in_idxs[0] - 1], CBLidx2mask[route_in_idxs[1]]])

            else:
                print("Something wrong with route module!")
                raise Exception


def init_weights_from_loose_model2(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    # compact_model新模型，loose_model旧模型
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        # np.argwhere返回非零元素的索引

        # 未剪的输出通道id
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        # print(compact_bn.bias.data)
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask2(loose_model.module_defs, idx, CBLidx2mask)

        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()

        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
        # 访问多维tensor时，需要分维访问，索引数组需要有相同的形状
        # compact_conv.weight.data = loose_conv.weight.data[out_channel_idx, in_channel_idx, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


# 完成BN以及卷积的复制
init_weights_from_loose_model2(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

random_input = torch.rand((1, 3, 416, 416)).to(device)


# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
with torch.no_grad():
    compact_model_metric = eval_model(compact_model)

# %%
# 比较剪枝前后参数数量的变化、指标性能的变化
metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[1].mean():.6f}', f'{compact_model_metric[1].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"]
]
print(AsciiTable(metric_table).table)


# 生成剪枝后的cfg文件并保存模型
# 'cfg/uavcut.cfg'
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')
# 由于原始的compact_module_defs将anchor从字符串变为了数组，因此这里将anchors重新变为字符串
for item in compact_module_defs:
    if item['type'] == 'yolo':
        item['anchors'] = str_anchors

pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file saved: {pruned_cfg_file}')

compact_model_name = 'weights/yolov3_tiny_uavcut_pruning_' + str(percent) + 'percent.pt'
chkpt = {'epoch': -1,
         'best_fitness': None,
         'training_results': None,
         'model': compact_model.state_dict(),
         'optimizer': None}

torch.save(chkpt, compact_model_name)

# save_weights(compact_model, path=compact_model_name)
print(f'Compact model saved: {compact_model_name}')
