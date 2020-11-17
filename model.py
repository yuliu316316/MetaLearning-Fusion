import torch
import math
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn
from utils import *


class DDense_Unit(nn.Module):
    def __init__(self, inC, outC, k, dk, d=1):
        super(DDense_Unit, self).__init__()
        self.Conv = nn.Sequential(*[Conv_S(inC, outC, k), nn.LeakyReLU(0.2),
                                    Conv_S(outC, outC, dk, dilation=d), nn.PReLU()])

    def forward(self, x, cat_input = True):
        f = self.Conv(x)
        if cat_input:
            return torch.cat([x, f], dim=1)
        return f

class DDense_block(nn.Module):
    def __init__(self, c, k):
        super(DDense_block, self).__init__()
        self.conv0 = nn.Sequential(Conv_S(c*2, c, k), nn.LeakyReLU(0.2))

        self.DConv_r1 = DDense_Unit(c, c, 1, k, 1)
        self.DConv_r3 = nn.Sequential(DDense_Unit(c, c, k, k, 3),
                                      DDense_Unit(c*2, c, k, k, 3),
                                      DDense_Unit(c*3, c, k, k, 3),
                                      DDense_Unit(c*4, c, k, k, 3))
        self.DConv_r5 = DDense_Unit(c, c, 5, k, 5)

        self.conv_k1 = nn.Sequential(Conv_S(c*7, c*2, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        f0 = self.conv0(x)
        f_r1 = self.DConv_r1(f0, cat_input=False)
        f_r3 = self.DConv_r3(f0)
        f_r5 = self.DConv_r5(f0, cat_input=False)
        f_cat = torch.cat([f_r1, f_r3, f_r5], dim=1)
        f_residual = self.conv_k1(f_cat)

        return x + f_residual

class FeatureEtraction(nn.Module):
    def __init__(self, args):
        super(FeatureEtraction,self).__init__()
        self.num_FEBs = args.num_FEBs
        c = args.fe_C

        self.conv0 = nn.Sequential(Conv_S(3, c*2, 3), nn.LeakyReLU(0.2))
        self.FENet = nn.ModuleList()
        for i in range(self.num_FEBs):
            self.FENet.append(DDense_block(args.fe_C, args.fe_k))
        self.conv_before_meta = nn.Sequential(Conv_S(c*2, args.meta_C, 1), nn.LeakyReLU(0.2))

    def forward(self, img):
        f_0 = self.conv0(img)
        f = f_0
        for i in range(self.num_FEBs):
            f = self.FENet[i](f)

        f = f + f_0
        f = self.conv_before_meta(f)
        torch.cuda.empty_cache()

        return f

class Position_Attention(nn.Module):
    def __init__(self):
        super(Position_Attention, self).__init__()
        self.PA_module =nn.Sequential(OrderedDict([
            ('GAP', nn.AdaptiveAvgPool3d((1, None, None))),
            ('MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),

            ('conv1', nn.Sequential(Conv_S(1, 8, 3), nn.LeakyReLU(0.2))),
            ('AvgPool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),

            ('conv2', nn.Sequential(Conv_S(8, 16, 3), nn.LeakyReLU(0.2))),
            ('subpixel_conv1', nn.PixelShuffle(2)),

            ('conv_k1', nn.Sequential(Conv_S(12, 4, 1), nn.LeakyReLU(0.2))), 
            ('subpixel_conv2', nn.PixelShuffle(2)),

            ('sigmoid', nn.Sigmoid())
        ]))
    #
    def forward(self, f0):
        f = self.PA_module.GAP(f0)
        f = self.PA_module.MaxPool(f)
        f_conv1 = self.PA_module.conv1(f)
        f = self.PA_module.AvgPool(f_conv1)
        f = self.PA_module.conv2(f)
        f = self.PA_module.subpixel_conv1(f)
        f = F.interpolate(f, f_conv1.shape[2:])
        f = torch.cat((f_conv1, f), dim = 1)
        f = self.PA_module.conv_k1(f)
        f = self.PA_module.subpixel_conv2(f)
        f = F.interpolate(f, f0.shape[2:])
        w = self.PA_module.sigmoid(f)

        return w


class Pos2Weight(nn.Module):
    def __init__(self, inC, outC, kernel_size=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.outC = outC
        self.kernel_size=kernel_size
        self.meta_block=nn.Sequential(
            nn.Linear(3, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.kernel_size*self.kernel_size*self.inC*self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)

        return output

class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        self.P2W = Pos2Weight(inC=args.meta_C, outC=args.meta_C)

    def repeat_x(self, x, r_int):
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)
        x = torch.cat([x] * r_int, 3)
        x = torch.cat([x] * r_int, 5).permute(0, 3, 5, 1, 2, 4)

        return torch.reshape(x, (-1, C, H, W)) 

    def forward(self, x, r, pos_mat, mask, HRsize):
        # torch.cuda.empty_cache() 
        scale_int = math.ceil(r)
        outC = x.size(1)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_x = self.repeat_x(x, scale_int) 
        cols = nn.functional.unfold(up_x, kernel_size=3, padding=1)
        cols = torch.reshape(cols, (cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2), 1)).permute(0, 1, 3, 4, 2)
        local_weight = torch.reshape(local_weight, (x.size(2), scale_int, x.size(3), scale_int, -1, outC)).permute(1,3,0,2,4,5)
        local_weight = torch.reshape(local_weight, (scale_int ** 2, x.size(2) * x.size(3), -1, outC)) 
        out = torch.matmul(cols, local_weight)
        out = out.permute(0, 1, 4, 2, 3)
        out = torch.reshape(out, (x.size(0), scale_int, scale_int, outC, x.size(2), x.size(3))).permute(0, 3, 4, 1, 5, 2)
        out = torch.reshape(out, (x.size(0), outC, scale_int * x.size(2), scale_int * x.size(3)))

        re_sr = torch.masked_select(out, mask)
        re_sr = torch.reshape(re_sr, (x.size(0), outC, HRsize[0], HRsize[1]))
        torch.cuda.empty_cache()

        return re_sr


class RCM(nn.Module):
    def __init__(self, args):
        super(RCM, self).__init__()
        self.meta_f_down = MetaLearner(args)
        self.conv_after_meta_down = nn.Sequential(Conv_S(args.meta_C, args.meta_C, args.meta_k), nn.LeakyReLU(0.2))
        self.meta_error_up = MetaLearner(args)

    def forward(self, fa, fb, fa_up, fb_up, f_fused, rs, pos_mat_a_up, mask_a_up, pos_mat_b_up, mask_b_up, pos_mat_a_down, mask_a_down, pos_mat_b_down, mask_b_down):

        fa_down = self.meta_f_down(fa_up, rs[2], pos_mat_a_down, mask_a_down, fa.shape[2:])
        error_a = torch.sub(fa, fa_down)
        error_a_up = self.meta_error_up(error_a, rs[0], pos_mat_a_up, mask_a_up, fa_up.shape[2:])

        fb_down = self.meta_f_down(fb_up, rs[3], pos_mat_b_down, mask_b_down, fb.shape[2:])
        error_b = torch.sub(fb, fb_down)
        error_b_up = self.meta_error_up(error_b, rs[1], pos_mat_b_up, mask_b_up, fb_up.shape[2:])

        error_max = abs_max(error_a_up, error_b_up)

        fa_up = fa_up + error_a_up
        fb_up = fb_up + error_b_up
        f_fused = f_fused + error_max

        return fa_up, fb_up, f_fused


class MLFusion(nn.Module):
    def __init__(self, args):
        super(MLFusion, self).__init__()
        self.args = args
        self.f_net = FeatureEtraction(args)
        self.meta_up_0 = MetaLearner(args)
        self.PA = Position_Attention()
        self.attention_AB_cat_CONV = nn.Sequential(Conv_S(args.meta_C*3, args.meta_C, kernel_size=1), nn.LeakyReLU(0.2))
        self.attention_max_fuse_CONV = nn.Sequential(Conv_S(args.meta_C*2, args.meta_C, kernel_size=1), nn.LeakyReLU(0.2))
        
        self.RCMs = nn.ModuleList()
        for i in range(self.args.num_RCMs):
            self.RCMs.append(RCM(args))

        self.conv_final_ab=nn.Sequential(Conv_S(args.meta_C, 3, kernel_size=args.fe_k))
        self.conv_final_fuse = nn.Sequential(Conv_S(args.meta_C, 3, kernel_size=args.fe_k))

        print('[*] Finished model initialization !')

    def attention_C(self, features):
        f_channel_ab = []

        for f in features:
            N, C, H, W = f.shape 
            f_channel = []

            for i in range(N):
                fi = f[i]
                fi = fi.permute(1, 2, 0).view(H * W, C)
                G = torch.matmul(fi.transpose(1, 0), fi)
                w = torch.sum(G, 0)
                mi = torch.min(w)
                d = torch.max(w) - mi
                w = torch.tensor(list((x - mi)/d for x in w)).to(w.device)
                softmax = nn.Softmax(dim=0)
                w = softmax(w)

                f_c = fi * w
                f_c = f_c.view(1, H, W, C).permute(0, 3, 1, 2)
                f_channel.append(f_c)

            f_channel_ab.append(torch.cat(f_channel, dim=0))

        return f_channel_ab


    def attention_P(self, features):
        f_position_ab = []

        for f in features:

            f_position = []
            for i in range(f.shape[0]):
                fi = f[i]
                fi = fi.unsqueeze(0)
                w = self.PA(fi)
                w = w.squeeze()
                fi = fi * w
                f_position.append(fi)

            f_position_ab.append(torch.cat(f_position, dim=0))

        return f_position_ab


    def FM(self, fa, fb):
        fa_C, fb_C = self.attention_C([fa, fb])
        [fa_P, fb_P] = self.attention_P([fa, fb])
        fa_Attention = torch.cat((fa, fa_C, fa_P), dim=1)
        fb_Attention = torch.cat((fb, fb_C, fb_P), dim=1)
        fa_CP = self.attention_AB_cat_CONV(fa_Attention)
        fb_CP = self.attention_AB_cat_CONV(fb_Attention)

        f_attention_fused = abs_max(fa_CP, fb_CP)

        f_max_fused = abs_max(fa, fb)

        f_cat = torch.cat((f_attention_fused, f_max_fused), dim=1)

        fuse_f = self.attention_max_fuse_CONV(f_cat)
        
        return fuse_f


    def forward(self, ia, ib, r_a_up, r_b_up, HRsize):
        _, _, inH_a, inW_a = ia.shape
        _, _, inH_b, inW_b = ib.shape

        rs = [r_a_up, r_b_up, 1.0/r_a_up, 1.0/r_b_up]

        pos_mat_a_up, mask_a_up = pos_mat(inH_a, inW_a, HRsize[0], HRsize[1], rs[0], add_scale=True)
        pos_mat_b_up, mask_b_up = pos_mat(inH_b, inW_b, HRsize[0], HRsize[1], rs[1], add_scale=True)
        pos_mat_a_down, mask_a_down = pos_mat(HRsize[0], HRsize[1], inH_a, inW_a, rs[2], add_scale=True)
        pos_mat_b_down, mask_b_down = pos_mat(HRsize[0], HRsize[1], inH_b, inW_b, rs[3], add_scale=True)

        # ------ 1.Features Extraction Network (in FEM) ------
        fa = self.f_net(ia)
        fb = self.f_net(ib)

        # ------ 2.Initial Meta-Upscale (in FEM) ------
        fa_up = self.meta_up_0(fa, rs[0], pos_mat_a_up, mask_a_up, HRsize)
        fb_up = self.meta_up_0(fb, rs[1], pos_mat_b_up, mask_b_up, HRsize)

        # ------ 3.Fusion Module ------
        f_fused = self.FM(fa_up, fb_up)

        # ------ 4.Residual Compensation ------ 
        for i in range(self.args.num_RCMs):
            fa_up, fb_up, f_fused = self.RCMs[i](
                fa, fb, fa_up, fb_up, f_fused, rs, 
                pos_mat_a_up, mask_a_up, pos_mat_b_up, mask_b_up, 
                pos_mat_a_down, mask_a_down, pos_mat_b_down, mask_b_down)

        # ------ 5.Reconstruct to images ------ 
        sr_a = self.conv_final_ab(fa_up)
        sr_b = self.conv_final_ab(fb_up)
        sr_fuse = self.conv_final_fuse(f_fused)

        sr_a, sr_b, sr_fuse = scale_image(sr_a), scale_image(sr_b), scale_image(sr_fuse)

        return sr_a, sr_b, sr_fuse