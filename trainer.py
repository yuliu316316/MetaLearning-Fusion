from loss import Loss
import time
from PIL import Image
from model import *
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if args.mode == 'train':
            self.trainset_loader = load_dataset(self.args, 'train_small') 
            self.valset_loader = load_dataset(self.args, 'varf')    

    def train(self, start_epoch):
        self.writer = SummaryWriter(log_dir=self.args.log_path, comment='train')
        self.Criterion = Loss(self.args)
        self.optimizer = make_optimizer(self.args, self.model.parameters())
        self.scheduler = make_scheduler(self.args, self.optimizer)

        counter = start_epoch * len(self.trainset_loader)
        end_epoch = (start_epoch + self.args.fine_tune_epoch) if self.args.finetune_AAF else (start_epoch + self.args.epochs)
        for epoch in range(start_epoch, end_epoch):
            self.scheduler.step(epoch)

            for idx, (img_AB, _) in enumerate(self.trainset_loader):
                time_sart = time.time()

                img_AB = rondomResize_HR(img_AB) # Rondomly crop patches with size of 128*128 from the source images.

                HR_a = img_AB[:self.args.batch_size]
                HR_b = img_AB[self.args.batch_size:]

                N, C, outH, outW = HR_a.size()
                r_a, r_b = np.around(np.random.uniform(1.0, 3.01, 2), decimals=1) # Rondomly generate the upscale factor of source images
                ia, ib = HR_a.clone(), HR_b.clone() 
                ia, ib = make_input(ia, r_a), make_input(ib, r_b)
                ia, ib = Variable(ia.cuda()), Variable(ib.cuda())
                HR_a, HR_b = Variable(HR_a.cuda()), Variable(HR_b.cuda())

                print('\n[Epoch:{0}/{1}][batch:{2}/{3}] ia.shape={4}, ra={5}, ib.shape={6}, rb={7}, HRsize={8}'
                      .format(epoch, end_epoch, idx, len(self.trainset_loader), ia.shape, r_a, ib.shape, r_b, [outH, outW]))
                self.optimizer.zero_grad()

                # Generate the super-resolution and fusion results
                sr_a, sr_b, sr_fused = self.model(ia, ib, r_a, r_b, [outH, outW])

                if not self.args.finetune_AAF:
                    L_pixel_sr, L_pixel_fuse = self.Criterion(HR_a, HR_b, sr_a, sr_b, sr_fused)
                    loss = L_pixel_sr + L_pixel_fuse
                    loss.backward()
                    print('Loss = {0:.4f}, time={1:.4f}'.format(loss.data, time.time() - time_sart))

                # Fine tune the network with the contrast loss function
                else:
                    L_pixel_sr, L_pixel_fuse, L_contrast = self.Criterion(HR_a, HR_b, sr_a, sr_b, sr_fused, self.args.loss_lambda, finetune_AAF=True)
                    loss = L_pixel_sr + L_pixel_fuse + L_contrast# + L_resolution
                    loss.backward()
                    self.writer.add_scalar('loss/Contrast', L_contrast.data, counter)
                    self.writer.add_scalar('loss/Total_loss', loss.data, counter)
                    print('Loss={0:.4f} l_pixel_sr={1:.4f} l_pixel_fuse={2:.4f} L_contrast={3:.4f} time={4:.4f}'.format(loss.data, L_pixel_sr.data, L_pixel_fuse.data, L_contrast.data, time.time() - time_sart))

                # Record the variety of loss to the log 
                self.writer.add_scalar('loss/Pixel_sr', L_pixel_sr.data, counter)
                self.writer.add_scalar('loss/Pixel_fuse', L_pixel_fuse.data, counter)
                self.writer.add_scalar('loss/Total_loss', loss.data, counter)
                self.optimizer.step()

                 # Save model and training result
                if (counter + 1) % 1500 == 0:
                    torch.save(self.model.state_dict(), self.args.ckp_path + self.args.modelname + '{}.pth'.format(epoch+1))
                    grid = make_grid(torch.cat([HR_a, sr_a, HR_b, sr_b, sr_fused], dim=0), nrow=self.args.batch_size).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    grid_gray = cv2.cvtColor(grid, cv2.COLOR_RGB2GRAY)
                    grid_gray3 = np.concatenate([grid_gray[:, :, np.newaxis]] * 3, axis=2)
                    cv2.imwrite('{}_grad.png'.format(counter+1), grid_gray3)
                    print('[*] Saved imgs ...')
                    self.verify(counter + 1)
                    self.model.train() 

                counter += 1
                torch.cuda.empty_cache()

            if (epoch+1) % 1 == 0:
                torch.save(self.model.state_dict(), self.args.ckp_path + self.args.modelname + '{}.pth'.format(epoch+1)) 
                print('[*] Saved Model ...')

        self.writer.close()


    def verify(self, counter): 
        print('[!] Start verifying !')
        time_sart = time.time()
        self.model.eval()    
        if self.args.finetune_AAF:
            var_loss_contrast = []
        var_loss_sr = []
        var_loss_fuse = []
        var_loss_total = []
        R = np.arange(1.5, 3.01, 0.5)

        with torch.no_grad():
            for idx, (img_AB, _) in enumerate(self.valset_loader):
                print('\n[*] Start varify the No.{} image....'.format(idx))
                img_AB = rondomResize_HR(img_AB)
                img_AB = img_AB.unsqueeze(dim=1)

                HR_a = img_AB[0]
                HR_b = img_AB[1]

                HR_img = torch.cat([HR_a, HR_b, torch.zeros((2,) + HR_a.shape[1:])], dim=0)
                HR_img = make_grid(HR_img, nrow=4).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                save_grids = [HR_img]

                N, C, outH, outW = HR_a.size()
                for r in R:
                    ia, ib = HR_a.clone().cpu(), HR_b.clone().cpu()  
                    ia, ib = make_input(ia, r), make_input(ib, r)
                    ia, ib = ia.cuda(), ib.cuda()
                    HR_a, HR_b = HR_a.cuda(), HR_b.cuda()

                    sr_a, sr_b, sr_fused = self.model(ia, ib, r, r, [outH, outW])

                    img = torch.cat([sr_a, sr_b, sr_fused, torch.zeros(sr_a.shape).cuda()], dim=0)
                    grid = make_grid(img, nrow=4).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    text = 'ra={} rb={}\na_in=[{},{}]\nb_in=[{},{}]\nout=[{},{}]\n'.format(r, r, ia.shape[2], ia.shape[3], ib.shape[2], ib.shape[3], outH, outW)
                    x0, y0, dy = 5, 20, 30 
                    for j, substr in enumerate(text.split('\n')):
                        y = y0 + j * dy
                        grid = cv2.putText(grid, substr, (x0 + outW * 3, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # FONT_HERSHEY_SIMPLEX

                    save_grids.append(grid)

                    L_pixel_sr, L_pixel_fuse = self.Criterion(HR_a, HR_b, sr_a, sr_b, sr_fused)
                    loss = L_pixel_sr + L_pixel_fuse
                    if self.args.finetune_AAF:
                        L_pixel_sr, L_pixel_fuse, L_contrast = self.Criterion(HR_a, HR_b, sr_a, sr_b, sr_fused, self.args.loss_lambda, finetune_AAF=True)
                        loss = L_pixel_sr + L_pixel_fuse + L_contrast
                        var_loss_contrast.append(L_contrast)

                    var_loss_sr.append(L_pixel_sr.data)
                    var_loss_fuse.append(L_pixel_fuse.data)
                    var_loss_total.append(loss.data)

                    torch.cuda.empty_cache()

                if idx % 40 == 0:
                    save_path = '{}_test{}.png'.format(counter, idx)
                    grid_gray = cv2.cvtColor(np.concatenate(save_grids, axis=0), cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(save_path, grid_gray)

            if self.args.finetune_AAF:
                var_loss_contrast = sum(var_loss_contrast) / len(var_loss_contrast)
                self.writer.add_scalar('var_loss/Contrast', var_loss_contrast, counter)

            var_loss_sr = sum(var_loss_sr) / len(var_loss_sr)
            var_loss_fuse = sum(var_loss_fuse) / len(var_loss_fuse)
            var_loss_total = sum(var_loss_total) / len(var_loss_total)
            self.writer.add_scalar('var_loss/Pixel_sr', var_loss_sr, counter)
            self.writer.add_scalar('var_loss/Pixel_fuse', var_loss_fuse, counter)
            self.writer.add_scalar('var_loss/Total_loss', loss.data, counter)

            print( 'VarAvgLoss=L_pixel={0:.4f} time={1:.4f}'.format(var_loss_total, time.time() - time_sart))



    def test_single_image(self, ia, ib, k, name, outSize):
        outH, outW = outSize

        r_a = max(outH / ia.shape[2], outW / ia.shape[3])
        r_b = max(outH / ib.shape[2], outW / ib.shape[3])

        print('outSize =', [outH, outW], ' ia.shape =', ia.shape[2:], ' ib.shape =', ib.shape[2:])
                        
        ia, ib = Variable(ia.cuda()), Variable(ib.cuda())

        sr_a, sr_b, sr_fused = self.model(ia, ib, r_a, r_b, [outH, outW])

        sr_fused = make_grid(sr_fused).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        r_a, r_b = round(1/r_a, 1), round(1/r_b, 1)# r_a, r_b = round(r_a, 2), round(r_b, 2) 
        save_path = self.args.testres_path + '/img{}_{}_ra{}_rb{}.tif'.format(k+1, name, r_a, r_b)
        sr_fused = cv2.cvtColor(sr_fused, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(save_path, sr_fused)


    def test(self):
        with torch.no_grad():
            self.model.eval()
            labels_pathsA = glob(self.args.testdata_path + '/IR_1.0/*')
            labels_pathsB = glob(self.args.testdata_path + '/VIS_1.0/*')
            data_A05 = glob(self.args.testdata_path + '/IR_0.5/*')
            data_B05 = glob(self.args.testdata_path + '/VIS_0.5/*')
            data_A08 = glob(self.args.testdata_path + '/IR_0.8/*')
            data_B08 = glob(self.args.testdata_path + '/VIS_0.8/*')

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
