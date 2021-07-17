import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


# 解析需要剪枝的层（须有BN）
def parse_module_defs(module_defs):
    # 卷积 + BN + LR
    CBL_idx = []
    # 卷积
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()
    # 不可以剪枝的卷积层 为了提高精度 上采样前一层不剪
    ignore_idx.add(18)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def gather_bn_weights(module_list, prune_idx):
    # ModuleList(
    # (0): Sequential(
    #     (Conv2d): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (BatchNorm2d): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (activation): LeakyReLU(negative_slope=0.1, inplace=True)
    # )

    # module_list[idx][1] -> BN Layer
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()  # 注意需要深拷贝
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def obtain_quantiles(bn_weights, num_quantile=5):
    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total // num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile + 1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):
    if idx == 0:
        return np.ones(3)

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
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
        else:
            print("Something wrong with route module!")
            raise Exception


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    # print(pruned_model)
    # print(CBL_idx)

    # 改进 补偿剪枝带来的影响
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)  # inplace

        # 留下BN层的bias
        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        if idx < 12:
            next_idx_list = [idx + 2]
        else:
            next_idx_list = [idx + 1]

        # 分支
        if idx == 13:
            next_idx_list.append(18)

        for next_idx in next_idx_list:
            next_conv = pruned_model.module_list[next_idx][0]
            # print(next_conv.weight.data.shape)
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))  # 对一个卷积核内部进行求和
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)

            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                # 这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre, device):
    thre = thre.to(torch.device(device))
    mask = bn_module.weight.data.abs().ge(thre).float()  # true --> 1.0

    return mask
