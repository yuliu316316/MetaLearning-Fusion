from utils import *
from torchvision.transforms import Compose, ToPILImage, CenterCrop, ToTensor
from torchvision.utils import make_grid, save_image
from time import time

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.L1 = nn.L1Loss()
        # Resize the HRs and fused image for reduce the computation cost
        self.H, self.W = 10, 10
        self.trans = Compose([ToPILImage(), CenterCrop((100, 100)), Resize((self.H, self.W)), ToTensor()])

    def L_contrast(self, HR_a, HR_b, sr_):
        Iavg = (HR_a + HR_b) / 2

        A, B, SR, IAVG = HR_a.clone().cpu(), HR_b.clone().cpu(), sr_.clone().cpu(), Iavg.clone().cpu()
        A, B, SR, IAVG = SizeTransform(A, self.trans), SizeTransform(B, self.trans), SizeTransform(SR, self.trans), SizeTransform(IAVG, self.trans) 

        D = torch.norm(SR - IAVG, p='fro').pow(2)  # F范数的平方

        def denominator(xp, yp, xq, yq):
            d_position = ((xp - xq)**2 + (yp - yq)**2)**0.5
            d_intensity = torch.abs(A[:, :, xp, yp] - A[:, :, xq, yq]) + torch.abs(B[:, :, xp, yp] - B[:, :, xq, yq])
            d_intensity = 1 - torch.tanh(d_intensity / 2)

            return d_position * d_intensity

        C = 0
        xy = list(map(lambda i: divmod(i, self.W), [x for x in range(self.H * self.W)]))
        
        for i in range(self.H * self.W):
            for j in range(self.H * self.W):
                if i == j:
                    continue
                molecule = torch.abs(SR[:, :, xy[i][0], xy[i][1]] - SR[:, :, xy[j][0], xy[j][1]])
                denominate = denominator(xy[i][0], xy[i][1], xy[j][0], xy[j][1]) + 0.0001 # (xp, yp, xq, yq)
                temp = torch.sum(molecule / denominate)
                C = C + temp.data

        # print('L_contrast over, cost time =', time()-start)
        return torch.abs(D - C).to(D.device)


    def L_pixel(self, HR_a, HR_b, sr_a, sr_b, sr_fused):
        L_pixel_a = self.L1(sr_a, HR_a)  # (input, target)
        L_pixel_b = self.L1(sr_b, HR_b)
        L_pixel_a_fuse = self.L1(sr_fused, HR_a)
        L_pixel_b_fuse = self.L1(sr_fused, HR_b)

        return (L_pixel_a + L_pixel_b), (L_pixel_a_fuse + L_pixel_b_fuse)


    def forward(self, HR_a, HR_b, sr_a, sr_b, sr_fused, loss_lambda=0, finetune_AAF=False):
        hr_a, hr_b = HR_a.to(sr_a.device), HR_b.to(sr_a.device) # 复制到同一个device
        l_pixel_sr, l_pixel_fuse = self.L_pixel(hr_a, hr_b, sr_a, sr_b, sr_fused)

        if finetune_AAF:
            l_contrast = self.L_contrast(hr_a, hr_b, sr_fused) * loss_lambda # * 2 # 最后才计算，因为中间会把HR_ab给缩小
            return l_pixel_sr, l_pixel_fuse, l_contrast

        return l_pixel_sr, l_pixel_fuse