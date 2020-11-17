import os
import cv2
import math
import numpy as np
import torch
import torch.utils.data
from glob import glob
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair


def abs_max(fa, fb):
    a = torch.abs(fa.data)
    b = torch.abs(fb.data)
    ab_max = torch.max(a, b)
    a_judge = torch.tensor(a == ab_max).type(torch.FloatTensor).cuda()
    b_judge = torch.tensor(b == ab_max).type(torch.FloatTensor).cuda()
    fa_judge = fa * a_judge
    fb_judge = fb * b_judge
    fab_max_abs = fa_judge + fb_judge

    return fab_max_abs


def pos_mat(inH, inW, outH, outW, scale, add_scale=True):  # input_matrix_wpn
    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    # outH, outW = int(scale * inH), int(scale * inW)

    #### mask records which pixel is invalid, 1 valid or 0 invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))

    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)
    if add_scale:
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        scale_mat = torch.cat([scale_mat] * (inH * inW * (scale_int ** 2)), 0)

    ####projection  coordinate  and caculate the offset
    h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord 
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord) 

    offset_w_coord = w_project_coord - int_w_project_coord 
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1
    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            # print(w_offset.shape)
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)  # ((scale_int*inH), (scale_int*inW), 2)=(outH, outW, 2)
    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2) 
    if add_scale: 
        pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2) 

    return pos_mat.cuda(), mask_mat.cuda()


def read_test_data(path_ab):
    img_a = torch.tensor(cv2.imread(path_ab[0])[np.newaxis]).permute(0, 3, 1, 2).type(torch.FloatTensor) / 255.0
    img_b = torch.tensor(cv2.imread(path_ab[1])[np.newaxis]).permute(0, 3, 1, 2).type(torch.FloatTensor) / 255.0

    return img_a, img_b


# def load_ckp(args, model):
#     all_ckp = glob(args.ckp_path + args.modelname + '*.pth')
#     all_ckp_epoch = list(map(lambda x:int(x[len(args.ckp_path + args.modelname):-4]), all_ckp))
#     ok = False

#     if all_ckp_epoch != []:
#         start_epoch = max(all_ckp_epoch)
#         path = args.ckp_path + args.modelname + '{}.pth'.format(start_epoch)
#         if os.path.exists(path):
#             model = model.cuda()
#             checkpoint = torch.load(path)
#             model.load_state_dict(checkpoint)
#             ok = True
#             print('[*] Sucessfully loaded checkpoint {0} !'.format(args.modelname + '{}.pth'.format(start_epoch)))
#             return model, ok

#         else:
#             print('[*] No model file, please train first! ')
#             return model, ok
#     else:
#         print('[*] No model file, please train first! ')
#         return model, ok


from collections import OrderedDict
def load_ckp(args, model):
    all_ckp = glob(args.ckp_path + args.modelname + '*.pth')
    all_ckp_epoch = list(map(lambda x:int(x[len(args.ckp_path + args.modelname):-4]), all_ckp))
    ok = False
    
    if all_ckp_epoch != []:
        start_epoch = max(all_ckp_epoch)
        path = args.ckp_path + args.modelname + '{}.pth'.format(start_epoch)
        if os.path.exists(path):
            model = model.cuda()
            checkpoint = torch.load(path)
            # newck1 = OrderedDict([(k.replace('BPMetaNet', 'RCMs'), v) if 'BPMetaNet' in k else (k, v) for k, v in checkpoint.items()])

            model.load_state_dict(checkpoint)
            ok = True
            print('[*] Sucessfully loaded checkpoint {0} !'.format(args.modelname + '{}.pth'.format(start_epoch)))
            return model, ok

        else:
            return model, ok
    else:
        return model, ok

        
# # ---------------------------------- SAME convolution in torch ---------------------------------------
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)

class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv_S(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv_S, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

# # ---------------------------------- SAME convolution in torch ---------------------------------------

def scale_image(img, color_max=1, fg_mask = None):
    if fg_mask is None:
        fg_mask = torch.ones(img.shape).bool().cuda()
    fg_vals = torch.masked_select(img, fg_mask)
    minv = fg_vals.min()
    maxv = fg_vals.max()
    img = (img - minv)
    img = img / (maxv - minv)
    img = img * color_max
    img[img > color_max] = color_max
    img[img < 0] = 0
    return img

def prepare(args):
    if not os.path.exists(args.testres_path):
        os.mkdir(args.testres_path)