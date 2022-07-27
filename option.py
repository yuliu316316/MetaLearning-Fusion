import argparse

parser = argparse.ArgumentParser(description=None)

# Trainning Specifications
parser.add_argument('--workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--fine_tune_epoch', type=int, default=3, help='Epoch number to fine tune the network with the contrast loss.')
parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training')
parser.add_argument('--num_RCMs', type=int, default=2, help='Number of RCMs (Residual Compensation Modules).')
parser.add_argument('--num_FEBs', type=int, default=6, help='Number of the FEBs (Feature Extraction Blocks).')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=0.005, help='The weight of the contrast loss function.')
parser.add_argument('--decay_type', type=str, default='multi_step', help='The type of lr to decay')
parser.add_argument('--lr_decay', type=str, default='300', help='Learning rate decay per N epochs')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='Learning rate decay factor for step decay')
parser.add_argument('--optim', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'))
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM/RMSprop epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD optimizer momentum')
parser.add_argument('--meta_C', type=int, default=8, help='Channel number of feature maps in MUM/MDM (Meta-Upscale/Downscale Module')
parser.add_argument('--fe_C', type=int, default=32, help='Channel number of the feature maps in FEM (Feature Extraction Module).')

# The size of image and kernal
parser.add_argument('--meta_k', type=int, default=3, help='Size of kernel produced by MUM/MDM.')
parser.add_argument('--fe_k', type=int, default=3, help='Size of kernel in FEM.')

# The pathes of some files to save.
parser.add_argument('--data_path', type=str, default='../../PycharmProjects/MetaFuseSR_single_cuda/dataset/', help='The path of dataset for train and verify.')
parser.add_argument('--testdata_path', type=str, default='sourceimages/', help='The path of dataset for test.')
parser.add_argument('--testres_path', type=str, default='test_result/', help='the dir of test results to save')
parser.add_argument('--log_path', type=str, default='logs/', help='the dir of log to save')
parser.add_argument('--ckp_path', type=str, default='checkpoint/', help='the dir of model to save')

# Other setting
parser.add_argument('--mode', type=str, default='train', choices=('train', 'test'))
parser.add_argument('--finetune_AAF', action='store_true', default=False, help='fine tune the AAF module.')
parser.add_argument('--modelname', type=str, default='MLFusion') #MetaSRFuse1Res_bnmeta_nobias MetaSRFuse2Res_test2 MetaSRFuse1Res6feb2meta_rgbtFLIR MetaSRFuse1Res_5feb_IRVIS_test MetaSRFuse1Res_5feb_allbn_nooutrelu_IRVIS MetaSRFuse2Res_5feb_IRVIS MetaSRFuse2Res_DIV2K-multifocus MetaSRFuse2Res_1e-3_IRVIS_PA+conv MetaSRFuse2Res_IRVISall_fixbn_gray MetaSRFuse2Res_IRVISall_fixbn_gray MetaSRFuse2Res_moreCONV_S_1e-3_IRVIS # bestMetaSRFuse2Res_moreCONV_S_1e-3 MetaSRFuse1Res_5feb_nobn_nooutrelu_IRVIS_lrnodecay_0.1lrdecay

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

print(args)