from __future__ import print_function
import os
import datetime
import argparse
import itertools
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import sys
# sys.path.append('/media/ntu/volume1/home/s122md306_05/G2R-ShadowNet')
print(sys.path)
from replace_D_by_ShaDetect.model_guided import Generator_S2F,Generator_F2S,Discriminator
from replace_D_by_ShaDetect.datasets_newISTD import ImageDataset
from replace_D_by_ShaDetect.utils import ReplayBuffer, LambdaLR, weights_init_normal
from custom_utils.plt_utils import draw_loss
import matplotlib.pyplot as plt
from BDRAR_util.model import BDRAR
import torchvision.transforms as transforms
plt.ioff()

#os.environ["CUDA_VISIBLE_DEVICES"]="7,3,1,2,0,5,6,4"
torch.manual_seed(628)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in use')
parser.add_argument('--output_dir', type=str, default="output", help='persist training state directory')
parser.add_argument('--pretrained_BDRAR', type=str, default="../BDRAR/ckpt/BDRAR/3000.pth", help='Pretrained shadow detector dir')
opt = parser.parse_args()


# ISTD
opt.dataroot = '../dataset/ISTD_Dataset'

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
opt.log_path = os.path.join(opt.output_dir, str(datetime.datetime.now()) + '.txt')

if torch.cuda.is_available():
    opt.cuda = True
    torch.cuda.set_device(opt.gpu_id)
    print(f"using GPU: {torch.cuda.current_device()}")
print(opt)

to_pil = transforms.ToPILImage()
gray_to_pil = transforms.Grayscale(num_output_channels=3)
def save_img(img_cuda,name,is_gray=False):
    if is_gray:
        img_cuda = gray_to_pil(img_cuda)
    img = 0.5*(img_cuda.detach().data+1.0)
    img = (to_pil(img.data.squeeze(0).cpu()))
    img.save(f"{opt.output_dir}/{name}")

###### Definition of variables ######
# Networks
netG = Generator_F2S(3,3)  # shadow to shadow_free
shadow_detector = BDRAR()

no_mask = torch.zeros((opt.size,opt.size))

if opt.cuda:
    netG.cuda()
    shadow_detector.cuda()
shadow_detector.load_state_dict(torch.load(opt.pretrained_BDRAR))
shadow_detector.eval()
for param in shadow_detector.parameters():
    param.requires_grad=False

netG.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()  # lsgan
# criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
def mask_guided_L1(a,b,mask):
    return criterion_identity(a*mask,b*mask)
# Optimizers & LR schedulers

optimizer_G = torch.optim.Adam(itertools.chain(netG.parameters()),
                               lr=opt.lr, betas=(0.9, 0.999))



lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor

input_A1 = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_B1 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_A2 = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_B2 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_u12 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_i12 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_u21 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_i21 = Tensor(opt.batchSize, 1, opt.size, opt.size)

target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_B_buffer = ReplayBuffer()

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot),
						batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
D_losses_temp = 0
G_losses = []
D_losses = []

open(opt.log_path, 'w').write(str(opt) + '\n\n')

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(tqdm(dataloader,mininterval=60)):
        # Set model input
        img1 = Variable(input_A1.copy_(batch["A1"]))
        mask1 = Variable(input_B1.copy_(batch["B1"]))
        img2 = Variable(input_A2.copy_(batch["A2"]))
        mask2 = Variable(input_B2.copy_(batch["B2"]))
        u12_mask = Variable(input_u12.copy_(batch["u12_mask"]))
        i12_mask = Variable(input_i12.copy_(batch["i12_mask"]))
        u21_mask = Variable(input_u21.copy_(batch["u21_mask"]))
        i21_mask = Variable(input_i21.copy_(batch["i21_mask"]))
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_1 = netG(img1,mask1)
        loss_identity_11 = criterion_identity(same_1, img1) 
        same_2 = netG(img2,mask2)
        loss_identity_22 = criterion_identity(same_2,img2)

        # Generative loss
        gen_u12 = netG(img1,u12_mask)
        u12_img_loss = mask_guided_L1(img1,gen_u12,(u12_mask+2-mask1)/2)
        pred_shad_u12 = shadow_detector(gen_u12)*2-1
        u12_shad_loss = criterion_identity(u12_mask,pred_shad_u12)

        gen_i12 = netG(img1,i12_mask)
        i12_img_loss = mask_guided_L1(img1,gen_i12,(mask1-i12_mask+2)/2)
        pred_shad_i12 = shadow_detector(gen_i12)*2-1
        i12_shad_loss = criterion_identity(i12_mask,pred_shad_i12)

        gen_u21 = netG(img2,u21_mask)
        u21_img_loss = mask_guided_L1(img2,gen_u21,(u21_mask+2-mask2)/2)
        pred_shad_u21 = shadow_detector(gen_u21)*2-1
        u21_shad_loss = criterion_identity(u21_mask,pred_shad_u21)

        gen_i21 = netG(img2,i21_mask)
        i21_img_loss = mask_guided_L1(img2,gen_i21,(mask2-i21_mask+2)/2)
        pred_shad_i21 = shadow_detector(gen_i21)*2-1
        i21_shad_loss = criterion_identity(i21_mask,pred_shad_i21)
        
        loss_weights = [10, 10, 5, 20, 5, 20, 5, 20, 5, 20]
        loss_arr = [loss_identity_11,loss_identity_22,u12_img_loss,u12_shad_loss,i12_img_loss,i12_shad_loss,u21_img_loss,u21_shad_loss,i21_img_loss,i21_shad_loss]
        # Total loss
        loss_G = 0
        for w,v in zip(loss_weights,loss_arr):
            loss_G += w*v
        loss_G.backward()

        with torch.no_grad():
            loss_D = sum([u12_shad_loss,i12_shad_loss,u21_shad_loss,i21_shad_loss])


        #G_losses.append(loss_G.item())
        G_losses_temp += loss_G.item()
        D_losses_temp += loss_D.item()
        optimizer_G.step()
        ###################################

        

        curr_iter += 1

        if (i+1) % opt.iter_loss == 0:
            log = f"iter {curr_iter}, loss_arr = {[x.item() for x in loss_arr]}"
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_losses.append(D_losses_temp/ opt.iter_loss)
            G_losses_temp = 0
            D_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f]' \
                      % (opt.iter_loss, G_losses[G_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

        if i==1 and (epoch + 1) % opt.snapshot_epochs == 0:
            save_img(img1,f"e{epoch}_img1.png")
            save_img(img2,f"e{epoch}_img2.png")
            save_img(mask1,f"e{epoch}_mask1.png",is_gray=True)
            save_img(mask2,f"e{epoch}_mask2.png",is_gray=True)

            save_img(gen_u12,f"e{epoch}_gen_u12.png")
            save_img(u12_mask,f"e{epoch}_mask_u12.png",is_gray=True)
            save_img(pred_shad_u12,f"e{epoch}_mask_pred_u12.png",is_gray=True)

            save_img(gen_i12,f"e{epoch}_gen_i12.png")
            save_img(i12_mask,f"e{epoch}_mask_i12.png",is_gray=True)
            save_img(pred_shad_i12,f"e{epoch}_mask_pred_i12.png",is_gray=True)

            save_img(gen_u21,f"e{epoch}_gen_u21.png")
            save_img(u21_mask,f"e{epoch}_mask_u21.png",is_gray=True)
            save_img(pred_shad_u21,f"e{epoch}_mask_pred_u21.png",is_gray=True)

            save_img(gen_i21,f"e{epoch}_gen_i21.png")
            save_img(i21_mask,f"e{epoch}_mask_i21.png",is_gray=True)
            save_img(pred_shad_i21,f"e{epoch}_mask_pred_i21.png",is_gray=True)

    if (epoch + 1) % opt.snapshot_epochs == 0:
        draw_loss([G_losses],["Generator Loss"],opt.iter_loss,opt.output_dir, "Generator_loss")

        draw_loss([D_losses],["D_B_losses"],opt.iter_loss,opt.output_dir,"Discriminator_loss")


    # Update learning rates
    lr_scheduler_G.step()


    if (epoch + 1) % opt.snapshot_epochs == 0:
        #f"{opt.output_dir}/netG_A2B_{epoch+1}.pth"
        torch.save(netG.state_dict(), f"{opt.output_dir}/netG_{epoch + 1}.pth")

    print('Epoch:{}'.format(epoch))