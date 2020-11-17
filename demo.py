import os
import argparse
from demo import *
from model import *
from torch.autograd import Variable
from torchvision.utils import make_grid
parser = argparse.ArgumentParser(description=None)

parser.add_argument('--meta_C', type=int, default=8, help='Channel number of feature maps in MUM/MDM (Meta-Upscale/Downscale Module ).')
parser.add_argument('--meta_k', type=int, default=3, help='Size of kernel produced by MUM/MDM.')
parser.add_argument('--num_RCMs', type=int, default=2, help='Number of RCMs (Residual Compensation Modules).')
parser.add_argument('--fe_C', type=int, default=32, help='Channel number of the feature maps in FEM (Feature Extraction Module).')
parser.add_argument('--fe_k', type=int, default=3, help='Size of kernel in FEM.')
parser.add_argument('--num_FEBs', type=int, default=6, help='Number of the FEBs (Feature Extraction Blocks).')

parser.add_argument('--data_path', type=str, default='sourceimages/', help='Path of data to save.')
parser.add_argument('--ckp_path', type=str, default='checkpoint/', help='Path of pretrain model to save.')
parser.add_argument('--testres_path', type=str, default='test_result/', help='Path of test results to save.')
parser.add_argument('--modelname', type=str, default='MLFusion', help='Name of the model.')

args = parser.parse_args()
print(args)

class Demo():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_single_image(self, ia, ib, k, name, outSize):
        outH, outW = outSize

        r_a = max(outH / ia.shape[2], outW / ia.shape[3])
        r_b = max(outH / ib.shape[2], outW / ib.shape[3])

        print('outSize =', [outH, outW], ' ia.shape =', ia.shape[2:], ' ib.shape =', ib.shape[2:])
                        
        ia, ib = Variable(ia.cuda()), Variable(ib.cuda())

        _, _, sr_fused = self.model(ia, ib, r_a, r_b, [outH, outW])

        sr_fused = make_grid(sr_fused).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        r_a, r_b = round(1/r_a, 1), round(1/r_b, 1)# r_a, r_b = round(r_a, 2), round(r_b, 2) 
        save_path = self.args.testres_path + '/img{}_{}_ra{}_rb{}.tif'.format(k+1, name, r_a, r_b)
        sr_fused = cv2.cvtColor(sr_fused, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(save_path, sr_fused)


    def test(self):
        with torch.no_grad():
            self.model.eval()
            labels_pathsA = glob(self.args.data_path + '/IR_1.0/*')
            labels_pathsB = glob(self.args.data_path + '/VIS_1.0/*')
            data_A05 = glob(self.args.data_path + '/IR_0.5/*')
            data_B05 = glob(self.args.data_path + '/VIS_0.5/*')
            data_A08 = glob(self.args.data_path + '/IR_0.8/*')
            data_B08 = glob(self.args.data_path + '/VIS_0.8/*')

            labels_pathsA.sort()
            labels_pathsB.sort()
            data_A05.sort()
            data_B05.sort()
            data_A08.sort()
            data_B08.sort()

            print('[!] Start Testing, test_every_r!')

            for k in range(0, len(labels_pathsA)):
                HR_a, HR_b = read_test_data([labels_pathsA[k], labels_pathsB[k]])
                ia05, ib05 = read_test_data([data_A05[k], data_B05[k]])
                ia08, ib08 = read_test_data([data_A08[k], data_B08[k]])

                A = [ia05, ia08, HR_a]
                B = [ib05, ib08, HR_b]

                name = labels_pathsA[k].split('/')[-1].split('.')[0]
                print('\n ---------- Testing No.{} image pair, name = {} ----------- '.format(k+1, name))

                outSize = HR_a.shape[2:] 

                for i in range(len(A)):
                    ia = A[i]
                    for j in range(len(B)):
                        ib = B[j]
                        self.test_single_image(ia, ib, k, name, outSize)

            print('[!] Testing has finished ! The result images saved in {0}.'.format(self.args.testres_path))


if __name__ == '__main__':
    MLF_model = MLFusion(args)
    MLF_model, ok = load_ckp(args, MLF_model)
    d = Demo(args, MLF_model)
    prepare(args)

    if ok == True:
        d.test()
    else: 
        print('[!] No checkpoint file, please train first ! ')
        exit(0)