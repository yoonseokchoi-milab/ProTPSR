import os
import sys
import pdb
import math
import time
import random
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from utils import get_config, get_scheduler, get_loader, Timer, write_2images
from trainer_h5_b_ContentNet import ContentNet_Trainer
from customdataset_h5_ProTPSR import HCP_Dataset
from monai.transforms import Resize
import monai

import torch
import numpy as np
from torch import nn, optim
from torch.backends import cudnn
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from eval_h5_adj_comb_non_rd_vec_at_test import eval_net
# from models.Stacked_DSUNet_single_FE_residual_HFP import Stacked_DSUNet_single_FE

import h5py


import matplotlib.pyplot as plt
# from torchvision import read_image, write_png
import torchvision
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric

from tensorboardX import SummaryWriter

# basicConfig 함수는 script 최상단에 위치해야함
# log level과 log format등의 설정은 프로그램이 실행되기 전에
# 한 번만 수행되어야 하기때문
logging.basicConfig( level=logging.INFO, format='%(levelname)s: %(message)s' )


#-----------------------------------------------------------------------------------------------------------------------

def cleanup():
    dist.destroy_process_group()

def build_tensorboard_test(config):
    """Build a tensorboard logger."""
    from logger_test import Logger
    writer = Logger( config )
    return writer

def log_command(args):
    # 현재 날짜와 시간
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 실행된 명령어
    command = '/home/milab/anaconda3/envs/yoonseok/bin/python /SSD4_8TB/CYS/02.Super-resolution_Missing_data_imputation/04.ProTPSR/test_h5_b_ContentNet.py ' + ' '.join(sys.argv[1:])

    # 로그 파일에 기록
    with open('test_ContentNet.txt', 'a') as log_file:
        log_file.write(f'{current_time} : {command}\n\n')

# def save(ckpt_dir, model, optimizer, epochs):

#     # Save generators, discriminators, and optimizers
#     stacked_dsunet_name = os.path.join(ckpt_dir, 'stacked_dsunet_%04d.pt' % (epochs + 1))                    
#     opt_name = os.path.join(ckpt_dir, 'optimizer.pt')
    
#     # Stacked_dusunet
#     torch.save({'model': model.state_dict(),
#                 'lr': optimizer.param_groups[0]['lr']}, stacked_dsunet_name)
    
#     # Optimizers
#     torch.save({'optimizer': optimizer.state_dict()}, opt_name)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if models is None:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name
#-----------------------------------------------------------------------------------------------------------------------

def get_model_ckpt_name(checkpoint_dir, key:str, epochs):
    if os.path.exists(checkpoint_dir) is False:
        return None
    assert os.path.isfile(os.path.join(checkpoint_dir, key + '_0' + str(epochs) + '.pt')), '해당 ckpt 파일은 존재하지 않습니다.'

    ckpt_name = os.path.join(checkpoint_dir, key + '_0' + str(epochs) + '.pt')
    return ckpt_name

def denorm(img):
    denorm_img = img*0.5 + 0.5
    return denorm_img

def minmaxnorm(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img
    
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------

def main(config):

    # For fast training.
    cudnn.benchmark = True

    # # Define usable gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.dataset in ['IXI']:
        test_dataset  = HCP_Dataset( config=config, mode='test' )
        nb_total_test_imgs  = len(test_dataset)
    
    logging.info(f'''Starting training:
        Epochs            : {config.epochs}
        Batch size        : {config.batch_size}
        Learning rate of G: {config.gen_lr}
        Learning rate of D: {config.dis_lr}
        Test size         : {nb_total_test_imgs}
        Device            : {config.gpu}
    ''')
    main_worker( config )


def main_worker( config ):
    """
    main worker안에
    Distributed Data Parallel (DDP)
    set up부터 다 되어있다.
    
    """

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Make dir for test output
    if config.train_dataset:
        base_dir = 'TPSR_physics/outputs/ContentNet/' + config.tb_comment + "/train"
    else:
        base_dir = 'TPSR_physics/outputs/ContentNet/' + config.tb_comment + "/test"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Define a TestDataset
    if config.dataset in ['IXI']:
        if config.train_dataset:
            dataset = HCP_Dataset( config=config, mode='train' )
        else:
            dataset = HCP_Dataset( config=config, mode='test' )
    

    # Set up the tensorboard 
    if config.use_tensorboard:
        writer = build_tensorboard_test(config)

    # Setup devices for this process
    device = torch.device("cuda:" + config.device)

    # Model
    trainer = ContentNet_Trainer( config ).to( device )
        
    # Dataloader    
    test_loader = DataLoader( dataset     = dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = False,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )
    
    # epochs = 200
    
    #------------------- Load the weights of model from checkpoint if it exists -------------------------------------#
    # ContentNet 
    dir_path_contentNet = config.ckpt_dir_contentNet + config.tb_comment

    print(f'contentNet_ckpt 경로: {dir_path_contentNet}\n\n')

    # 혹시 model 선언만 되고 weight load가 안된건가?
    # if os.path.isdir(dir_path_contentNet):
        # trainer.load에서는 test에 필요한 gen_a 와 gen_b만 load 됨
    trainer.load(checkpoint_dir=dir_path_contentNet, key="gen", epochs=str(config.epochs))
    
        
    # # Stacked_DSUNet
    # dir_path_stacked_dsunet = config.ckpt_dir_stacked_dsunet + config.tb_comment
    
    # if os.path.isdir(dir_path_stacked_dsunet):
        # Load model
    # ckpt_model_name = get_model_ckpt_name(checkpoint_dir=dir_path_stacked_dsunet, key="stacked_dsunet", epochs=config.epochs)
    # assert os.path.isfile(ckpt_model_name), f"No checkpoint found at {ckpt_model_name}"
    # ckpt = torch.load(ckpt_model_name)
    # stacked_dsunet.load_state_dict(ckpt['model'])
    
    # Content encoder, Style encoder, and Decoder
    # content_encoder = trainer.gen_a.encode
    # style_encoder   = trainer.gen_b.encode
    # decoder         = trainer.gen_b.decode

    
    # stacked_dsunet.eval()

    n_test = len(test_loader)   
    
    # data 저장을 위한 list 선언

    # data_A
    data_A_list = []

    # data_B
    # data_B_list    = []
    data_B_HR_list = []

    # data_A_recon
    data_B_recon_list = []

    # c_a
    # c_a_list = []
    c_b_ds_1st_list = []
    c_b_ds_2nd_list = []
    # c_a_ds_3rd_list = []
    c_b_final_list = []

    # canny edge
    canny_edge_list = []

    # Stacked_dsunet output
    # refine_sr_img_list = []

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=2)
    psnr_metric = PSNRMetric(max_val=1)

    ContentNet_running_ssim_value = 0
    ContentNet_running_psnr_value = 0
    ContentNet_mean_ssim_value = 0
    ContentNet_mean_psnr_value = 0
    n_samples = 0

    dict_ContentNet_mean_metric_tb = {}
    
    with tqdm(total=n_test, 
              desc='Test', 
              unit='imgs', 
              ncols=180,
              ascii=' ==') as pbar:
        with torch.no_grad():
            # assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
            # """
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch
            # before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.        
            # """
            # # test_sampler.set_epoch( epochs )
                        
            for step, img in enumerate(test_loader, len(test_loader)):
                
                img["data_A"]       = img["data_A"].to(device)
                img["data_B_4fold"] = img["data_B_4fold"].to( device ) # data consistency를 위한 4fold t2
                img["data_B_2fold"] = img["data_B_2fold"].to( device ) # data consistency를 위한 2fold t2, 4fold -> 2fold로 갈때의 label
                img["data_B_41"]    = img["data_B_41"].to( device ) # input: 4fold에서 resize로 HR 만든 t2 
                img["data_B_21"]    = img["data_B_21"].to( device ) # input: 2fold에서 resize로 HR 만든 t2
                img["data_B_HR"]    = img["data_B_HR"].to( device ) # ground truth t2
                img["coeffs_hf_comp_A_1st"] = img["coeffs_hf_comp_A_1st"].to( device )
                img["coeffs_hf_comp_A_2nd"] = img["coeffs_hf_comp_A_2nd"].to( device )
                # img["data_A_edge"] = img["data_A_edge"].to( device )
                

                a_hr    = img["data_A"]
                b_4fold = img["data_B_4fold"] # (B, 1, 32,  256) | (B, 1, 320, 80)
                b_2fold = img["data_B_2fold"] # (B, 1, 64,  256) | (B, 1, 320, 160)
                b_41    = img["data_B_41"]    # (B, 1, 128, 256) | (B, 1, 320, 320)
                b_21    = img["data_B_21"]    # (B, 1, 128, 256) | (B, 1, 320, 320)                
                b_hr    = img["data_B_HR"]    # (B, 1, 128, 256) | (B, 1, 320, 320)
                coeffs_hf_comp_A_1st = img["coeffs_hf_comp_A_1st"] # (B, 3, 64, 128)   | (B, 3, 160, 160) 
                coeffs_hf_comp_A_2nd = img["coeffs_hf_comp_A_2nd"] # (B, 3, 32, 64)    | (B, 3, 80, 80)  
                data_A_edge = img["data_A_edge"].float() # (B, 1, 128, 256)  | (B, 1, 320, 320)
                data_B_HR_edge = img["data_B_HR_edge"].float()

                
                # Process for making Hybrid Feature Priors
                # Using non random vector
                trainer.eval()
                c_b, s_b_prime = trainer.gen_b.encode( b_hr )
                x_b_recon = trainer.gen_b.decode( c_b[-1], s_b_prime )
                # trainer.train()
                

                # SSIM 및 PSNR 계산
                ssim_value = ssim_metric(y_pred=x_b_recon, y= img["data_B_HR"][:])
                psnr_value = psnr_metric(y_pred=x_b_recon, y= img["data_B_HR"][:])
                
                # pdb.set_trace()

                ContentNet_running_ssim_value += torch.sum(ssim_value).item() # 배치별로 값이 하나씩 나와서 일단 sum해주고 itme으로 값만 가지고 온다.
                ContentNet_running_psnr_value += torch.sum(psnr_value).item() # 배치별로 값이 하나씩 나와서 일단 sum해주고 itme으로 값만 가지고 온다.
                n_samples += img["data_B_HR"].size(0)
                
                # 저장하고자 하는 데이터 list에 모아놓기

                # data_A
                data_A_list.append(denorm(img["data_A"][:].cpu().numpy()))

                # data_B
                # data_B_list.append(denorm(img["data_B"][:].cpu().numpy()))
                data_B_HR_list.append(denorm(img["data_B_HR"][:].cpu().numpy()))

                # data_AB
                # data_AB_list.append(minmaxnorm(x_ab).cpu().numpy())
                data_B_recon_list.append(denorm(x_b_recon).cpu().numpy())

                # c_b
                c_b_ds_1st_list.append(c_b[0].cpu().numpy())
                c_b_ds_2nd_list.append(c_b[1].cpu().numpy())
                # c_a_ds_3rd_list.append(c_a[2].cpu().numpy())
                c_b_final_list.append(c_b[-1].cpu().numpy())

                # canny edge
                canny_edge_list.append(data_B_HR_edge.cpu().numpy())

                # Stacked_dsunet output
                # refine_sr_img_list.append(minmaxnorm(refine_sr_img).cpu().numpy())
                # refine_sr_img_list.append(denorm(x_a_recon).cpu().numpy())

                pbar.update(img['data_B_HR'].shape[0])


                # =================================================================================== #
                #                                 3. Miscellaneous                                    #
                # =================================================================================== #

                # For ContentNet
                HR_t2_center = torch.Tensor(img["data_B_HR"][:config.tb_display_size].cpu())

                c_b_ds_1st = torch.Tensor(c_b[0][:config.tb_display_size, 0:1].cpu())  # config.tb_display_size x 128 x 64 x 128 중에 첫 번째 ch의 feature map만 가지고 옴
                c_b_ds_2nd = torch.Tensor(c_b[1][:config.tb_display_size, 0:1].cpu())  # config.tb_display_size x 256 x 32 x 64  중에 첫 번째 ch의 feature map만 가지고 옴
                c_b_final  = torch.Tensor(c_b[-1][:config.tb_display_size, 0:1].cpu()) # config.tb_display_size x 256 x 32 x 64  중에 첫 번째 ch의 feature map만 가지고 옴
                # c_a_ds_3rd = torch.Tensor(c_a[2][:config.tb_display_size, :].cpu()) # config.tb_display_size x 512 x 16 x 32

                resize_transform = Resize(spatial_size=HR_t2_center.shape[2:]) # Resize(spatial_size=(128,256))

                # print(f"\n\nHR_t2_center.shape[2:]: {HR_t2_center.shape[2:]}")
                # print(f"\nc_a_ds_1st.shape: {c_a_ds_1st.shape}")

                # pdb.set_trace()

            
                resize_c_b_ds_1st = torch.Tensor(torch.stack([resize_transform(c_b_ds_1st[i][:]) for i in range(c_b_ds_1st.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_b_ds_2nd = torch.Tensor(torch.stack([resize_transform(c_b_ds_2nd[i][:]) for i in range(c_b_ds_2nd.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_b_final  = torch.Tensor(torch.stack([resize_transform(c_b_final[i][:]) for i in range(c_b_final.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                # c_a_ds_LH_3rd = torch.Tensor(torch.stack([resize_transform(c_a_ds_3rd[i][0:1]) for i in range(c_a_ds_3rd.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                
                # print(f'HR_t2_center:{HR_t2_center.shape}')
                # print(f'c_a:{c_a.shape}')
                # print(f'x_ab:{x_ab.shape}')
                # print(f'LR_t2_center:{LR_t2_center.shape}')
                HR_t2_recon_center = torch.Tensor(x_b_recon[:config.tb_display_size].cpu())
                canny_edge = torch.Tensor(data_B_HR_edge[:config.tb_display_size])

                contentNet_img_tensor = torch.cat([resize_c_b_ds_1st,
                                                   resize_c_b_ds_2nd,
                                                   resize_c_b_final,
                                                   canny_edge,
                                                   HR_t2_recon_center,
                                                   HR_t2_center], dim=0)
                
                contentNet_img_grid = torchvision.utils.make_grid( tensor    = contentNet_img_tensor.data,
                                                            nrow      = config.tb_display_size, # 각 row 당 몇개의   이미지를 display 할건지
                                                            padding   = 0,
                                                            normalize = True )

            # 2024.07.17.(수)
            # eval 에는 매 epoch 마다 한 번씩 들어오게 되고
            # tensorboard plot의 step은 iteration이 다돌고 epoch이 바뀔 때마다
            # 한 번 update 하게끔 설정해놨음
            # 현재 설정으로는 eval_loader의 shuffle 기능을 True로 설정해놨기 때문에
            # writer.image_summary가 iteration for문 안에 있든, 바깥에 있든 계속 다른 이미지 plot 할 것.
            writer.image_summary( tag        = 'test/ContentNet--caDs1st--caDs2nd--caFinal--cannyEdge--xbRecon--xbCenter',
                                    img_tensor = contentNet_img_grid,
                                    step       = step + 1 )

            ContentNet_mean_ssim_value = ContentNet_running_ssim_value / n_samples
            ContentNet_mean_psnr_value = ContentNet_running_psnr_value / n_samples
            
            dict_ContentNet_mean_metric_tb['test/SSIM'] = ContentNet_mean_ssim_value
            dict_ContentNet_mean_metric_tb['test/PSNR'] = ContentNet_mean_psnr_value
            
            # Visualize metric graphs
            for tag, value in dict_ContentNet_mean_metric_tb.items():
                # print(f"Type of value: {type(value)}")
                print(f"{tag}: {value:.4f}")
                # if isinstance(value, monai.data.meta_tensor.MetaTensor):
                    # value = torch.Tensor(value.cpu())  # Convert MetaTensor to torch.Tensor
                writer.scalar_summary(tag, value, step+1)  # Use .item() to get a Python number from a tensor containing a single value
            
            writer.writer.close()

            # 데이터를 저장하기 전에 Reshape 해주는 것
            # (N/B, B, C, H, W) --> (B, C, H, W)
            data_A       = np.concatenate(data_A_list,        axis=0)
            # data_B_4to1       = np.concatenate(data_B_4to1_list,       axis=0)
            # data_B_2to1       = np.concatenate(data_B_2to1_list,       axis=0)
            data_B_HR    = np.concatenate(data_B_HR_list,     axis=0)
            data_B_recon = np.concatenate(data_B_recon_list,  axis=0)
            c_b_ds_1st   = np.concatenate(c_b_ds_1st_list,    axis=0)
            c_b_ds_2nd   = np.concatenate(c_b_ds_2nd_list,    axis=0)
            c_b_final    = np.concatenate(c_b_final_list,     axis=0)
            canny_edge   = np.concatenate(canny_edge_list,    axis=0)
            
            

            #-------------------------------------------------------------------------------------------------------#
            # Data save
            data_save_dir = os.path.join(base_dir, f"output_data_1.2mm_{config.epochs}.h5")
            with h5py.File(data_save_dir, 'w') as f:

                # data_A
                f.create_dataset("data_A",    data=data_A)

                # data_B
                # f.create_dataset("data_B",    data=data_B_list)
                f.create_dataset("data_B_HR", data=data_B_HR)

                # data_A_recon
                f.create_dataset("data_B_recon", data=data_B_recon)
                
                # c_a
                f.create_dataset("data_c_b_ds_1st",  data=c_b_ds_1st)
                f.create_dataset("data_c_b_ds_2nd",  data=c_b_ds_2nd)
                f.create_dataset("data_c_B",       data=c_b_final)

                # canny edge
                f.create_dataset("data_B_canny_edge", data=canny_edge)
        
        log_command(config)
        #-------------------------------------------------------------------------------------------------------#

    # cleanup()
#-----------------------------------------------------------------------------------------------------------

def str2bool(v):
    return v.lower() in ('true')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Through Plane Super Res')
    # current_date = datetime.date.today()

    # Training Configurations
    parser.add_argument('-gpu', '--gpu',         type=str,   default="7")
    parser.add_argument('-device', '--device',   type=str,   default="0")
    parser.add_argument('--dataset',             type=str,   default='IXI', choices=['IXI'])
    parser.add_argument('-w', '--workers',       type=int,   default=8,         help='number of data loader workers')
    parser.add_argument('-e', '--epochs',        type=int,   default=1,         help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size',    type=int,   default=10,        help='mini-batch size')
    parser.add_argument('-g_lr', '--gen_lr',     type=float, default=1e-4,      help='Learning rate of G')
    parser.add_argument('-d_lr', '--dis_lr',     type=float, default=1e-4,      help='Learning rate of D')
    parser.add_argument('-stacked_dsunet_lr', '--stacked_dsunet_lr', type=float, default=1e-4, help='Learning rate of Stacked_DSUNet')
    parser.add_argument('--beta1',               type=float, default=0.5,       help='beta1 for Adam optimizer')
    parser.add_argument('--beta2',               type=float, default=0.999,     help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay',        type=float, default=1e-4,      help='weight decay for Adam optimizer')
    parser.add_argument('--init',                type=str,   default='kaiming', help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr_policy',           type=str,   default='step',    help='learning rate scheduler')
    parser.add_argument('--step_size',           type=int,   default=20,        help='how often to decay learning rate when using step scheduler')
    parser.add_argument('--gamma',               type=float, default=0.1,       help='how much to decay learning rate when using StepLR scheduler')
    parser.add_argument('--patience',            type=int,   default=10,        help='how much to be patient for decaying learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--factor',              type=float, default=0.1,       help='how much to decay learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--gan_w',               type=int,   default=1,         help='weight of adversarial loss')
    parser.add_argument('--recon_x_w',           type=int,   default=10,        help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w',           type=int,   default=1,         help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w',           type=int,   default=1,         help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w',       type=int,   default=1,         help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--vgg_w',               type=int,   default=1,         help='weight of domain-invariant perceptual loss')
    parser.add_argument('--recon_ssim_w',        type=int,   default=1,         help='weight of ssim loss')
    parser.add_argument('--recon_L1_w',          type=int,   default=1,         help='weight of l1 loss')
    parser.add_argument('--recon_perceptual_w',  type=int,   default=1,         help='weight of  perceptual loss')
    parser.add_argument('--generator_steps',     type=int,   default=1,         help='number of steps for generator training')
    parser.add_argument('--discriminator_steps', type=int,   default=1,         help='number of steps for discriminator training')


    # Test configurations
    parser.add_argument('--test_epochs', type=int, default=10, help='test model from this epoch')

    # Data options
    parser.add_argument('--input_ch_a', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--input_ch_b', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--slice_thickness', type=str, default='2', choices=['1.2','2','3','4','5'], help='slice thickness of images')
    
    parser.add_argument('-nb_train_imgs', '--nb_train_imgs', type=int, default=20, help='number of images employed for train code checking')
    parser.add_argument('-nb_test_imgs', '--nb_test_imgs', type=int, default=20, help='number of images employed for test code checking')
    parser.add_argument('--train_dataset', type=str2bool, default=None,  help='Condition whether test the trained network on train dataset')
    # parser.add_argument('--plane', type=str, default='sag', choice=['sag','cor'], help='plane of images')

    # Model configurations
    # Generator
    parser.add_argument('--gen_dim', type=int, default=64, help='number of filters in the bottommost layer = the # of content code')
    parser.add_argument('--gen_mlp_dim', type=int, default=256, help='number of filters in MLP')
    parser.add_argument('--gen_style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--gen_activ', type=str, default='relu', help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--gen_n_downsample', type=int, default=2, help='number of downsampling layers in content encoder')
    parser.add_argument('--gen_n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--gen_pad_type', type=str, default='reflect', help='padding type [zero/reflect]')

    # Discriminator
    parser.add_argument('--dis_dim', type=int, default=64, help='number of filters in the bottommost layer')
    parser.add_argument('--dis_norm', type=str, default='none', help='normalization layer [none/bn/in/ln]')
    parser.add_argument('--dis_activ', type=str, default='lrelu', help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--dis_n_layer',type=int, default=4, help='number of layers in D')
    parser.add_argument('--dis_gan_type', type=str, default='lsgan', help='GAN loss [lsgan/nsgan]')
    parser.add_argument('--dis_num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--dis_pad_type', type=str, default='reflect', help='padding type [zero/reflect]')

    # Directories
    
    # ContentNet
    parser.add_argument('--log_dir',                type=str, default='TPSR_physics/logs/ContentNet/')
    parser.add_argument('--ckpt_dir_contentNet',         type=str, default='TPSR_physics/ckpts/ContentNet/', help='path to checkpoint directory')
    parser.add_argument('--train_sample_dir_contentNet', type=str, default='TPSR_physics/samples/train/ContentNet/')
    parser.add_argument('--val_sample_dir_contentNet',   type=str, default='TPSR_physics/samples/val/ContentNet/')
    parser.add_argument('--result_dir_contentNet',       type=str, default='TPSR_physics/results/ContentNet/')
    
    # parser.add_argument('--ixi_h5_1_2mm_dir', type=str, default='data/00.IXI_Dataset/h5/2d_tpsr/01.sag/01.slice_thickness_t1_1.2mm_t2_1.2mm_40_10/')
    # parser.add_argument('--ixi_h5_2mm_dir', type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/02.slice_thickness_t1_1.2mm_t2_2mm_40_10/')
    # parser.add_argument('--ixi_h5_3mm_dir', type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/03.slice_thickness_t1_1.2mm_t2_3mm_40_10/')
    # parser.add_argument('--ixi_h5_4mm_dir', type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/04.slice_thickness_t1_1.2mm_t2_4mm_40_10/')
    # parser.add_argument('--ixi_h5_5mm_dir', type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/05.slice_thickness_t1_1.2mm_t2_5mm_40_10/')   

    # Data dir
    parser.add_argument('--hcp_h5_0_7mm_dir', type=str, default='data/01.HCP_Dataset/h5/ProTPSR/00.slice_thickness_t1_0.7mm_t2_0.7mm_1.4mm_2.8mm_5.6mm_40_10/')
    
    # Miscellaneous
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--tb_display_size', type=int, default=3)
    parser.add_argument('--tb_comment', type=str, default='_test_TPSR_2023_10_25_wo_perceptual_loss_on_xab_t1_1.2mm_t2_2mm_b_16_e_1000/')   

    # Step size.
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--img_save_step', type=int, default=2)
    parser.add_argument('--img_display_step', type=int, default=2)
    parser.add_argument('--model_save_step', type=int, default=5)
    # parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    # print(config)
    main(config)