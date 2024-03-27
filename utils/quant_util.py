import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple

from .quantization_utils.quant_utils import *


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight

def kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=False):
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        new_weight_data = layer.weight.data.clone()
        new_weight_data.zero_()
        this_cl_list = centroid_label_dict[i]
        num_centroids = this_cl_list[0][0].numel()
        if num_centroids > 2**6 and free_high_bit:
            # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
            continue
        for j in range(num_centroids):
            mask_cl = (this_cl_list[0][1] == j).float()
            new_weight_data += (layer.weight.data * mask_cl).sum() / mask_cl.sum() * mask_cl
        layer.weight.data = new_weight_data


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class Quant(nn.Module):
    def __init__(self, range_left=-6, range_right=6, dim=128):
        super(Quant, self).__init__()
        self.range_left = torch.tensor([range_left], requires_grad=True).cuda()
        self.range_right = torch.tensor([range_right], requires_grad=True).cuda()
        self.act_function = AsymmetricQuantFunction.apply

    def forward(self, inputs, a_bit):
        # quant_act = self.act_function(inputs, a_bit, self.range_left, self.range_right)
        scale, zero_point = asymmetric_linear_quantization_params(
                a_bit, self.range_left, self.range_right
        )
        new_quant_x = torch.round(scale * inputs.transpose(1,-1) - zero_point)
        n = 2**(a_bit - 1)
        new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)

        quant_act = (new_quant_x_1 + zero_point) / scale
        quant_act = quant_act.transpose(1,-1)
        return quant_act
    

seq = 0
class QModule(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, w_bit=8, a_bit=8, half_wave=False, sequence=None, args=None):
        super(QModule, self).__init__()

        self._a_bit = a_bit
        self._w_bit = w_bit
        self._b_bit = 32
        
        self._half_wave = half_wave
        # self.sequence = list(reversed(sequence))
        self.len_seq = 100
        self.index_seq = 0
        self.args = args
        # print(self.sequence)

        self.init_range_min = torch.Tensor(-4. * torch.ones(self.len_seq))
        self.init_range_max = torch.Tensor(6. * torch.ones(self.len_seq))
        self.group_num = 8
        # self.activation_range_min = nn.Parameter(torch.Tensor(torch.zeros(self.len_seq, in_channels)))
        # self.activation_range_max = nn.Parameter(torch.Tensor(torch.zeros(self.len_seq, in_channels)))
        self.activation_range_min = torch.Tensor(torch.zeros(self.len_seq, in_channels))
        self.activation_range_max = torch.Tensor(torch.zeros(self.len_seq, in_channels))
        self.groups_range = torch.Tensor(torch.zeros([self.len_seq, self.group_num, 2])).cuda()

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True 
        self._calibrate = False
        self._first_calibrate = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_function = AsymmetricQuantFunction.apply
        self.act_function = AsymmetricQuantFunction.apply

        self.activation_range_min1 = torch.Tensor(torch.zeros(self.len_seq, self.in_channels)).cuda()
        self.activation_range_max1 = torch.Tensor(torch.zeros(self.len_seq, self.in_channels)).cuda()
        self.weight_range_min = torch.Tensor(torch.zeros(self.out_channels)).cuda()
        self.weight_range_max = torch.Tensor(torch.zeros(self.out_channels)).cuda()

        self.alpha_activ = nn.Parameter(torch.Tensor(self.len_seq, self.group_num, in_channels), requires_grad=True)
        self.alpha_activ.data.fill_(0.01)
        self.sw = torch.Tensor(self.group_num, in_channels).cuda()
        self.mix_activ_mark1 = nn.ModuleList()

    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def a_bit(self):
        return self._a_bit 

    @a_bit.setter
    def a_bit(self, a_bit):
        self._a_bit = a_bit

    @property
    def b_bit(self):
        return self._b_bit

    @property
    def half_wave(self):
        return self._half_wave

    @property
    def quantized(self):
        return self._quantized

    @property
    def tanh_weight(self):
        return self._tanh_weight

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable_activation_range=True):
        self._trainable_activation_range = trainable_activation_range
        self.activation_range.requires_grad_(trainable_activation_range)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def first_calibrate(self, calibrate=True):
        self._first_calibrate = calibrate

    def set_tanh(self, tanh=True):
        self._tanh_weight = tanh

    def calibrate_quantization(self, inputs, init_min, init_max):
        inputs_calibrate = inputs.clone().transpose(0, 1)
        inputs_calibrate = inputs_calibrate.reshape(self.in_channels, -1)

        self.activation_range_min1[self.index_seq] = torch.min(inputs_calibrate, 1)[0]
        self.activation_range_max1[self.index_seq] = torch.max(inputs_calibrate, 1)[0]

        self.activation_range_min1[self.index_seq] = torch.where(self.activation_range_min1[self.index_seq] > init_min
                                                    , init_min*torch.ones(self.in_channels).cuda(), self.activation_range_min1[self.index_seq])
        self.activation_range_max1[self.index_seq] = torch.where(self.activation_range_max1[self.index_seq] < init_max
                                                    , init_max*torch.ones(self.in_channels).cuda(), self.activation_range_max1[self.index_seq])

        self.activation_range_min1[self.index_seq], group_range_min = GroupWise_Quantizaion(self.activation_range_min1[self.index_seq].detach().clone(), 
                                                                        dim=self.in_channels, group_n=self.group_num, maxmin='min')
        self.activation_range_max1[self.index_seq], group_range_max = GroupWise_Quantizaion(self.activation_range_max1[self.index_seq].detach().clone(), 
                                                                        dim=self.in_channels, group_n=self.group_num, maxmin='max')

        self.activation_range_min = torch.tensor(self.activation_range_min1, device=inputs.device)
        self.activation_range_max = torch.tensor(self.activation_range_max1, device=inputs.device)

        self.groups_range[self.index_seq] = torch.stack([group_range_min, group_range_max], dim=1)
        
        # differentiable search
        self.mix_activ_mark1 = nn.ModuleList()
        for group in torch.stack([group_range_min, group_range_max], dim=1):
            self.mix_activ_mark1.append(Quant(range_left=group[0], range_right=group[1], dim=self.in_channels))
        self.sw = F.softmax(self.alpha_activ[self.index_seq], dim=0)
        inputs_1 = inputs.transpose(1, -1)
        for i, branch in enumerate(self.mix_activ_mark1):
            x = branch(inputs_1, a_bit=self._a_bit)
            if i == 0:
                outs_x = (x * self.sw[i]).unsqueeze(0).clone()
            else:
                outs_x = torch.cat((outs_x, (x * self.sw[i]).unsqueeze(0)), 0)
        activ = torch.sum(outs_x, dim=0).transpose(1, -1)
        return activ

    def _quantize_activation(self, inputs):
        # print(self._a_bit, self._w_bit)
        if self.index_seq >= 100:
            self.index_seq = 0
        global seq
        seq += 1
        if seq >= (44+1):  # 一个Qconv，一次前传22个conv
            seq = 1

        if self._calibrate:
            # first search
            if self._first_calibrate:
                best_score = 1e+10
                best_max = self.init_range_max[self.index_seq]
                best_min = self.init_range_min[self.index_seq]
                for aa in range(9):
                    new_max = self.init_range_max[self.index_seq] * (1.0 - (aa * 0.1))
                    new_min = self.init_range_min[self.index_seq] * (1.0 - (aa * 0.1))
                    activ_tmp = self.calibrate_quantization(inputs, new_min, new_max)
                    score = lp_loss(activ_tmp, inputs, p=0.5, reduction='all')
                    if score < best_score:
                        best_max = new_max
                        best_min = new_min
                        best_score = score
                if best_score < 0.2:
                    self.init_range_max[self.index_seq] = best_max
                    self.init_range_min[self.index_seq] = best_min
            activ = self.calibrate_quantization(inputs, self.init_range_min[self.index_seq], self.init_range_max[self.index_seq])
            self.index_seq += 1
            return activ

        activation_range_diff_min, activation_range_diff_max = 0, 0
        self.sw = F.softmax(self.alpha_activ[self.index_seq], dim=0)
        # differentiable
        for i in range(self.group_num):
            group_left_tmp = self.groups_range[self.index_seq][i][0]
            group_right_tmp = self.groups_range[self.index_seq][i][1]
            activation_range_diff_min += group_left_tmp * self.sw[i]
            activation_range_diff_max += group_right_tmp * self.sw[i]

        # quantization
        scale, zero_point = asymmetric_linear_quantization_params(
                self._a_bit, activation_range_diff_min, activation_range_diff_max
        )
        new_quant_x = torch.round(scale * inputs.transpose(1,-1) - zero_point)
        n = 2**(self._a_bit - 1)
        new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)

        quant_act = (new_quant_x_1 + zero_point) / scale
        quant_act = quant_act.transpose(1,-1)

        self.index_seq += 1
        return quant_act

    def _quantize_weight(self, weight):
        if self._calibrate:
            x_transform = weight.data.contiguous().view(self.out_channels, -1)
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            tmp_min = torch.stack((self.weight_range_min, w_min), dim=0)
            tmp_max = torch.stack((self.weight_range_max, w_max), dim=0)

            self.weight_range_min += -self.weight_range_min + torch.min(tmp_min, 0)[0]
            self.weight_range_max += -self.weight_range_max + torch.max(tmp_max, 0)[0]
            return weight
        scaling_factor = self.weight_range_max / (pow(2., self._w_bit - 1) - 1.)
        w = 0.5 * ((-weight.transpose(0,-1) + self.weight_range_min).abs() - (weight.transpose(0,-1) - self.weight_range_max).abs() + self.weight_range_min + self.weight_range_max)
        w.div_(scaling_factor).round_().mul_(scaling_factor)
        w = w.transpose(0,-1)
        return w

    def _quantize_bias(self, bias):
        if bias is not None and self._quantized and self._b_bit > 0:
            if self._calibrate:
                return bias
            ori_b = bias
            threshold = ori_b.data.max().item() + 0.00001
            scaling_factor = threshold / (pow(2., self._b_bit - 1) - 1.)
            b = torch.clamp(ori_b.data, -threshold, threshold)
            b.div_(scaling_factor).round_().mul_(scaling_factor)
            # STE
            if self._fix_weight:
                return b.detach()
            else:
                # b = ori_b + b.detach() - ori_b.detach()
                return STE.apply(ori_b, b)
        else:
            return bias

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        # print(inputs.shape, weight.shape, self.out_channels)  # torch.Size([64, 512, 8, 8]) torch.Size([256, 512, 3, 3]) 256
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return 'w_bit={}, a_bit={}, half_wave={}, tanh_weight={}'.format(
            self.w_bit if self.w_bit > 0 else -1, self.a_bit if self.a_bit > 0 else -1,
            self.half_wave, self._tanh_weight
        )


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QConv2d(QModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 w_bit=8, a_bit=8, half_wave=False, sequence=None, args=None):
        super(QConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, 
                                      w_bit=w_bit, a_bit=a_bit, half_wave=half_wave, sequence=sequence, args=args)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={}, a_bit={}'.format(self.w_bit, self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s.format(**self.__dict__)


def GroupWise_Quantizaion(x, dim=128, group_n=8, maxmin='max'):
    C = dim
    range_max = x.max()
    range_min = x.min()
    range_group = [range_min]
    range_div = range_max - range_min
    # 分组量化组别分界值
    for m in range(group_n):
        range_group.append(range_min + range_div * (m+1)/group_n)
    # print(range_group)
    # 分组标识矩阵
    mark = torch.zeros(C).cuda()
    for m in range(group_n):
        mark = torch.where(((x>=range_group[m])&(x<=range_group[m+1])), (m+1)*torch.ones(C).cuda(), mark)
    # print(mark)
    # 分组量化阈值
    group_mean = []
    for m in range(group_n):
        group_mean1 = torch.masked_select(x, mark == (m+1))
        # if group_mean1 == torch.Size([]):
        if min(group_mean1.shape) == 0:
                group_mean.append(range_group[m+1])
        else:
            if maxmin in 'max':
                group_mean.append(torch.max(group_mean1))
            elif maxmin in 'min':
                group_mean.append(torch.min(group_mean1))
    # print(group_mean)
    group_mean = torch.tensor(group_mean, requires_grad=True, device=x.device)
    # 输出分组量化结果矩阵
    x_q = torch.zeros(C).cuda()
    for m in range(group_n):
        x_q += torch.where(mark == (m+1), group_mean[m], torch.zeros(C).cuda())
    # print(x_q)
    return x_q, group_mean


def find_scale_by_percentile_min(x, percentile=0.9999):
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * (1 - percentile))
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]

def find_scale_by_percentile_max(x, percentile=0.9999):
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * percentile)
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]