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
from eval_h5_b_ContentNet import eval_net
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision

from tensorboardX import SummaryWriter

# basicConfig 함수는 script 최상단에 위치해야함
# log level과 log format등의 설정은 프로그램이 실행되기 전에
# 한 번만 수행되어야 하기때문
logging.basicConfig( level=logging.INFO, format='%(levelname)s: %(message)s' )


#-----------------------------------------------------------------------------------------------------------------------

def cleanup():
    dist.destroy_process_group()

def build_tensorboard(config):
    """Build a tensorboard logger."""
    from logger import Logger
    writer = Logger( config )
    return writer

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



#-----------------------------------------------------------------------------------------------------------------------

def main(config):

    # For fast training.
    cudnn.benchmark = True

    # # Define usable gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.dataset in ['IXI']:
        train_dataset = HCP_Dataset( config=config, mode='train' )
        test_dataset  = HCP_Dataset( config=config, mode='test' )
        nb_total_train_imgs = len(train_dataset)
        nb_total_test_imgs  = len(test_dataset)
    
    logging.info(f'''Starting training:
        Epochs            : {config.epochs}
        Batch size        : {config.batch_size}
        Learning rate of G: {config.gen_lr}
        Learning rate of D: {config.dis_lr}
        Training size     : {nb_total_train_imgs}
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
    torch.manual_seed(0)  # trianing시에 Dataloader로 trainDataset의 index를 shuffle 할 때
                          # random으로 섞기 때문에 같은 performance를 유지하기 위해선 torch.manual_seed()가 필요하다.
    # Data loader.
    ixi_loader  = None

    if config.dataset in ['IXI']:
        train_dataset = HCP_Dataset( config=config, mode='train' )
        test_dataset  = HCP_Dataset( config=config, mode='test' )
        nb_total_train_imgs = len(train_dataset)
    
    # Set up the currenttime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    # Set up the tensorboard 
    if config.use_tensorboard:
        writer = build_tensorboard(config)

    
    # Load experiment setting
    max_epoch = config.epochs
    # display_size = config.display_size # How many images do you want to display each time

    # Setup devices for this process
    device = torch.device("cuda:" + config.device)

    # Model
    trainer = ContentNet_Trainer( config ).to( device )
    
    
    # Dataloader
    train_loader = DataLoader( dataset     = train_dataset, 
                               batch_size  = config.batch_size, 
                               shuffle     = True,
                               num_workers = config.workers,
                               pin_memory  = True, 
                            #    sampler     = train_sampler, 
                               drop_last   = True )
    
    eval_loader = DataLoader( dataset     = test_dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = False,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )

    test_loader = DataLoader( dataset     = test_dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = False,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )
    
    # Automatically resume from checkpoint if it exists    
    # ContentNet    
    dir_path_contentNet = config.ckpt_dir_contentNet + config.tb_comment
    if os.path.isdir(dir_path_contentNet):
        start_epochs = trainer.resume(dir_path_contentNet, config)
    else:
        start_epochs = 0
           
    
    
    print('start_epoch:', start_epochs + 1)
    
    # Start traing process ---------------------------------------------------------------------------
    for epochs in range( start_epochs, max_epoch ):
        # torch.autograd.set_detect_anomaly(True)
        """
        In distributed mode, calling the set_epoch() method at the beginning of each epoch
        before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        Otherwise, the same ordering will be always used.        
        """
        # train_sampler.set_epoch( epochs )    # Dataloader iterator 들어가기 전에 set_epoch( epoch )
                                           # 선언 안해주면 Dataset의 index shuffle 안해줌
        
        # ContentNet loss   
        dis_total_running_loss = 0
        gen_total_running_loss = 0
        dis_total_epoch_loss   = 0
        gen_total_epoch_loss   = 0        
        
        dict_dis_indiv_running_loss_tb = {}
        dict_gen_indiv_running_loss_tb = {}
        dict_dis_indiv_epoch_loss_tb = {}
        dict_gen_indiv_epoch_loss_tb = {}
        
        # DSU-Net loss
        # dsunet_ssim_running_loss       = 0
        # dsunet_l1_running_loss         = 0
        # dsunet_perceptual_running_loss = 0
        # dsunet_highfreq_running_loss   = 0
        # dsunet_ssim_epoch_loss       = 0
        # dsunet_l1_epoch_loss         = 0
        # dsunet_perceptual_epoch_loss = 0        
        # dsunet_total_running_loss = 0
        # dsunet_total_epoch_loss = 0
        
        # dict_dsunet_indiv_running_loss_tb = {}
        # dict_dsunet_indiv_epoch_loss_tb = {}
        
        dict_total_epoch_loss  = {}
        
        # For visualizing all the losses utilized in training phase
        dict_merged_epoch_loss_tb = {}
        
        # Generator와 Discriminator의 학습 비율 설정
        generator_steps = config.generator_steps
        discriminator_steps = config.discriminator_steps
        
        # Manually control on tqdm() updates by using a with statement
        with tqdm(total=nb_total_train_imgs,
                  desc=f'Epoch {epochs + 1}/{max_epoch}',
                  unit='imgs',
                  ncols=150,
                  ascii=' ==' ) as pbar:
            
            for step, img in enumerate( train_loader, start=epochs * nb_total_train_imgs):
                
                # Dictionary object는 .to 바로 쓸 수 없음!
                img["data_A"]       = img["data_A"].to(device)
                img["data_B_4fold"] = img["data_B_4fold"].to( device ) # data consistency를 위한 4fold t2
                img["data_B_2fold"] = img["data_B_2fold"].to( device ) # data consistency를 위한 2fold t2, 4fold -> 2fold로 갈때의 label
                img["data_B_41"]    = img["data_B_41"].to( device ) # input: 4fold에서 resize로 HR 만든 t2 
                img["data_B_21"]    = img["data_B_21"].to( device ) # input: 2fold에서 resize로 HR 만든 t2
                img["data_B_HR"]    = img["data_B_HR"].to( device ) # ground truth t2
                # img["coeffs_hf_comp_A_1st"] = img["coeffs_hf_comp_A_1st"].to( device )
                # img["coeffs_hf_comp_A_2nd"] = img["coeffs_hf_comp_A_2nd"].to( device )
                # img["coeffs_hf_comp_B_HR_1st"] = img["coeffs_hf_comp_B_HR_1st"].to( device )
                # img["coeffs_hf_comp_B_HR_2nd"] = img["coeffs_hf_comp_B_HR_2nd"].to( device )
                img["data_A_edge"]    = img["data_A_edge"].to( device )
                img["data_B_HR_edge"] = img["data_B_HR_edge"].to( device )

                a_hr    = img["data_A"]
                b_4fold = img["data_B_4fold"] # (B, 1, 32,  256) | (B, 1, 320, 80)
                b_2fold = img["data_B_2fold"] # (B, 1, 64,  256) | (B, 1, 320, 160)
                b_41    = img["data_B_41"]    # (B, 1, 128, 256) | (B, 1, 320, 320)
                b_21    = img["data_B_21"]    # (B, 1, 128, 256) | (B, 1, 320, 320)                
                b_hr    = img["data_B_HR"]    # (B, 1, 128, 256) | (B, 1, 320, 320)
                # coeffs_hf_comp_A_1st    = img["coeffs_hf_comp_A_1st"]    # (B, 3, 64, 128)   | (B, 3, 160, 160) 
                # coeffs_hf_comp_A_2nd    = img["coeffs_hf_comp_A_2nd"]    # (B, 3, 32, 64)    | (B, 3, 80, 80)  
                # coeffs_hf_comp_B_HR_1st = img["coeffs_hf_comp_B_HR_1st"] # (B, 3, 64, 128)   | (B, 3, 160, 160) 
                # coeffs_hf_comp_B_HR_2nd = img["coeffs_hf_comp_B_HR_2nd"] # (B, 3, 32, 64)    | (B, 3, 80, 80)  
                data_A_edge    = img["data_A_edge"] # (B, 1, 128, 256)  | (B, 1, 320, 320)
                data_B_HR_edge = img["data_B_HR_edge"] # (B, 1, 128, 256)  | (B, 1, 320, 320)

                # =================================================================================== #
                #                           1. Train the Discriminator                                #
                # =================================================================================== #
                
                for _ in range(discriminator_steps):

                    # Discriminator
                    loss_dis_total, dis_indiv_loss_tb, dis_opt, pred_real, pred_fake = trainer.dis_update( b_hr,
                                                                                                           config,
                                                                                                        #    coeffs_hf_comp_B_HR_1st,
                                                                                                        #    coeffs_hf_comp_B_HR_2nd,
                                                                                                           data_B_HR_edge )
                    dis_total_running_loss += loss_dis_total

                    # print("\n", dis_indiv_loss_tb.items())
                    # pdb.set_trace()
                    
                    # python 3.x 버전에서는 dict type인 경우 .items() method를 사용하여
                    # key-value 쌍을 얻는다.
                    # .items() method없이 그냥 사용하면
                    # {'D_loss/dis_adv_a': 3.000314235687256, 'D_loss/dis_adv_b': 2.9689536094665527} 가 나오고
                    # ValueError: too many values to unpack (expected 2) 생김
                    for key, val in dis_indiv_loss_tb.items(): 
                        if key not in dict_dis_indiv_running_loss_tb:
                            dict_dis_indiv_running_loss_tb[key] = val
                        else:
                            dict_dis_indiv_running_loss_tb[key] += val

                # =================================================================================== #
                #                           2. Train the Generator                                    #
                # =================================================================================== #

                for _ in range(generator_steps):

                    # Generator
                    loss_gen_total , gen_indiv_loss_tb, gen_opt, c_a, x_a_recon = trainer.gen_update(b_hr,
                                                                                                     config,
                                                                                                    #  coeffs_hf_comp_B_HR_1st,
                                                                                                    #  coeffs_hf_comp_B_HR_2nd,
                                                                                                     data_B_HR_edge )

                    gen_total_running_loss += loss_gen_total
                    
                    for key, val in gen_indiv_loss_tb.items():
                        if key not in dict_gen_indiv_running_loss_tb:
                            dict_gen_indiv_running_loss_tb[key] = val
                        else:
                            dict_gen_indiv_running_loss_tb[key] += val
                
                
                
                torch.cuda.synchronize() # 딥러닝을 돌리다보면 GPU는 아직 계산중인데
                                            # CPU는 GPU를 기다리지 않고 그 다음 코드를 실행하려고 할때가 있음
                                            # 즉, GPU와 CPU의 계산 타이밍이 어긋나는 것
                                            # torch.cuda.synchronize()는 그럴때 GPU와 CPU의 타이밍을 맞추기 위해
                                            # GPU 계산이 끝날때까지 CPU execution을 block한다.
            
                pbar.update(img['data_A'].shape[0]) # tqdm의 progress bar를 input data의 (batch_size) * (사용하는 gpu개수) 만큼 
                                                            # 업데이트 해준다.

            
            # Update learning rate
            trainer.dis_scheduler.step()
            trainer.gen_scheduler.step()         
            
            # Comput epoch loss for dis, gen, and dsunet
            dis_total_epoch_loss = dis_total_running_loss / (nb_total_train_imgs*discriminator_steps)
            gen_total_epoch_loss = gen_total_running_loss / (nb_total_train_imgs*generator_steps)
            
            dict_total_epoch_loss = { 'D_loss/total': dis_total_epoch_loss,
                                      'G_loss/total': gen_total_epoch_loss }

            for key, val in dict_dis_indiv_running_loss_tb.items():
                dict_dis_indiv_epoch_loss_tb[key] = val / nb_total_train_imgs
            
            for key, val in dict_gen_indiv_running_loss_tb.items():
                dict_gen_indiv_epoch_loss_tb[key] = val / nb_total_train_imgs
                

            # Dict type의 변수를 합치는 과정
            dict_merged_epoch_loss_tb.update( dict_total_epoch_loss )
            dict_merged_epoch_loss_tb.update( dict_dis_indiv_epoch_loss_tb )
            dict_merged_epoch_loss_tb.update( dict_gen_indiv_epoch_loss_tb )

            # Save checkpoint per 5 epochs
            if (epochs + 1) % config.model_save_step == 0:
                
                # Create directories if not exist.
                
                # ContentNet
                # if not os.path.exists(os.path.join(config.log_dir)):
                #     os.makedirs(os.path.join(config.log_dir), exist_ok=True)
                if not os.path.exists(os.path.join(config.ckpt_dir_contentNet, config.tb_comment)):
                    os.makedirs(os.path.join(config.ckpt_dir_contentNet, config.tb_comment), exist_ok=True)
                                    
                
                # ContentNet
                trainer.save( os.path.join(config.ckpt_dir_contentNet, config.tb_comment), epochs )
                


            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            if (epochs + 1) % config.img_display_step == 0: 
                
                if not os.path.exists(os.path.join(config.log_dir)):
                    os.makedirs(os.path.join(config.log_dir), exist_ok=True)

                # Visualization of result images during training
                with torch.no_grad():
                    # assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
                    
                    # # For ContentNet
                    # munit_img_outputs = trainer.sample(concat_A, concat_B)
                    # img_list = []  # Create an empty list to collect processed images

                    # for imgs in munit_img_outputs:

                    #     # print(imgs.shape)

                    #     if imgs.shape[1] == 3:
                    #         imgs = imgs[:,1:2]
                    #     elif imgs.shape[1] > 3: # c_a의 채널수는 64 --> visualization을 위해 첫 번째 채널만 선택
                    #         if imgs.shape[2:] != img["data_B_HR"].shape[2:]:
                    #             imgs = imgs[:, 1:2] # B x 1 x 32 x 64

                    #             resize_transform = Resize(spatial_size=img["data_B_HR"].shape[2:])
                    #             imgs = torch.stack([resize_transform(imgs[i]) for i in range(imgs.shape[0])])
                    #     img_list.append(torch.Tensor(imgs[:config.tb_display_size].cpu()))     
                        
                    # # Concatenate all images along the batch dimension
                    # munit_img_tensor = torch.cat(img_list, dim=0)

                    # Write to tensorboard                        
                    # munit_img_grid = torchvision.utils.make_grid( tensor    = munit_img_tensor.data,
                    #                                               nrow      = config.tb_display_size, # 각 row 당 몇개의 이미지를 display 할건지
                    #                                               padding   = 0,
                    #                                               normalize = True )
    
                    # writer.image_summary( tag        = 'train/ContentNet---x_a_center---c_a---x_a_recon---x_ab---x_b_center---x_b_recon',
                    #                       img_tensor = munit_img_grid,
                    #                       step       = epochs + 1 )
                    
                    
                        
                    # Visualize loss graphs
                    for tag, value in dict_merged_epoch_loss_tb.items():
                        # print(f"Type of value: {type(value)}")
                        print(f"{tag}: {value:.4f}")
                        # if isinstance(value, monai.data.meta_tensor.MetaTensor):
                            # value = torch.Tensor(value.cpu())  # Convert MetaTensor to torch.Tensor
                        writer.scalar_summary(tag, value.item(), epochs+1)  # Use .item() to get a Python number from a tensor containing a single value
                        

                    writer.scalar_summary('G_loss/lr', gen_opt.param_groups[0]['lr'], epochs + 1)
                    writer.scalar_summary('D_loss/lr', dis_opt.param_groups[0]['lr'], epochs + 1)
 
            # Validation phase
            if (epochs + 1) % 1 == 0:                
                eval_net( trainer, eval_loader, config, epochs, device, writer ) 
            
            if (epochs + 1) == 2:   
                log_command(config)

        # trainer.update_learning_rate(config)
    writer.writer.close()
    # cleanup()
#-----------------------------------------------------------------------------------------------------------

def str2bool(v):
    return v.lower() in ('true')

def log_command(args):
    # 현재 날짜와 시간
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 실행된 명령어
    command = '/home/milab/anaconda3/envs/yoonseok/bin/python /SSD4_8TB/CYS/02.Super-resolution_Missing_data_imputation/04.ProTPSR/train_h5_b_ContentNet.py ' + ' '.join(sys.argv[1:])

    # 로그 파일에 기록
    with open('train_ContentNet.txt', 'a') as log_file:
        log_file.write(f'{current_time} : {command}\n\n')

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
    parser.add_argument('--beta1',                   type=float, default=0.5,       help='beta1 for Adam optimizer')
    parser.add_argument('--beta2',                   type=float, default=0.999,     help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay',            type=float, default=1e-4,      help='weight decay for Adam optimizer')
    parser.add_argument('--init',                    type=str,   default='kaiming', help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr_policy',               type=str,   default='step',    help='learning rate scheduler')
    parser.add_argument('--step_size',               type=int,   default=20,        help='how often to decay learning rate when using step scheduler')
    parser.add_argument('--gamma',                   type=float, default=0.1,       help='how much to decay learning rate when using StepLR scheduler')
    parser.add_argument('--patience',                type=int,   default=10,        help='how much to be patient for decaying learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--factor',                  type=float, default=0.1,       help='how much to decay learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--gan_w',                   type=float,   default=1.0,         help='weight of adversarial loss')
    parser.add_argument('--recon_x_w',               type=float,   default=10.0,        help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w',               type=float,   default=1.0,         help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w',               type=float,   default=1.0,         help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w',           type=float,   default=1.0,         help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--vgg_w',                   type=float,   default=1.0,         help='weight of domain-invariant perceptual loss')
    parser.add_argument('--recon_l1_w',              type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_l1_w_c_b_dwt_1st',  type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_l1_w_c_b_dwt_2nd',  type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_w',            type=float,   default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_x_b',        type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_w_c_b',        type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_w_cyc_x_b',    type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_w_c_b_dwt_1st',type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_w_c_b_dwt_2nd',type=float,   default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_perceptual_w',      type=float,   default=1.0,         help='weight of  perceptual loss')
    parser.add_argument('--recon_perceptual_w_c_b_dwt_1st',  type=float,   default=1.0,         help='weight of  perceptual loss')
    parser.add_argument('--recon_perceptual_w_c_b_dwt_2nd',  type=float,   default=1.0,         help='weight of  perceptual loss')
    parser.add_argument('--generator_steps',     type=int,   default=1,         help='number of steps for generator training')
    parser.add_argument('--discriminator_steps', type=int,   default=1,         help='number of steps for discriminator training')


    # Test configurations
    parser.add_argument('--test_epochs', type=int, default=10, help='test model from this epoch')

    # Data options
    parser.add_argument('--input_ch_a', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--input_ch_b', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--slice_thickness', type=str, default='2', choices=['1.2','2','3','4','5'], help='slice thickness of images')
    
    parser.add_argument('--nb_train_imgs', type=int,  default=20,    help='number of images employed for train code checking')
    parser.add_argument('--nb_test_imgs',  type=int,  default=20,    help='number of images employed for test code checking')
    parser.add_argument('--train_dataset', type=bool, default=None,  help='Condition whether test the trained network on train dataset')

    # parser.add_argument('--plane', type=str, default='sag', choice=['sag','cor'], help='plane of images')

    # Model configurations
    # Generator
    parser.add_argument('--gen_dim', type=int, default=64, help='number of filters in the bottommost layer = the # of channels of content code')
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
    # parser.add_argument('--ixi_h5_2mm_dir',   type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/02.slice_thickness_t1_1.2mm_t2_2mm_40_10/')
    # parser.add_argument('--ixi_h5_3mm_dir',   type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/03.slice_thickness_t1_1.2mm_t2_3mm_40_10/')
    # parser.add_argument('--ixi_h5_4mm_dir',   type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/04.slice_thickness_t1_1.2mm_t2_4mm_40_10/')
    # parser.add_argument('--ixi_h5_5mm_dir',   type=str, default='data/00.IXI_Dataset/h5/2d_tpsr_adj/01.sag/00.adj_1/05.slice_thickness_t1_1.2mm_t2_5mm_40_10/')   
    
    # Data dir
    parser.add_argument('--hcp_h5_0_7mm_dir', type=str, default='data/01.HCP_Dataset/h5/ProTPSR/00.slice_thickness_t1_0.7mm_t2_0.7mm_1.4mm_2.8mm_5.6mm_40_10/')
    
    # Miscellaneous
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--tb_display_size', type=int, default=3)
    parser.add_argument('--tb_comment', type=str, default='_TPSR_2023_10_25_wo_perceptual_loss_on_xab_t1_1.2mm_t2_2mm_b_16_e_1000/')   

    # Step size.
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--img_save_step', type=int, default=2)
    parser.add_argument('--img_display_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=5)
    # parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    # print(config)
    main(config)