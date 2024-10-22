import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import argparse
import torch.distributed as dist
import os
from monai.inferers.inferer import SlidingWindowInferer
from utils import write_2images, __write_images

import torchvision
import matplotlib.pyplot as plt
from logger import Logger
import numpy as np

# from networks_ContentNet import GANLoss
# from perceptual import PerceptualLoss
# from monai.losses.ssim_loss import SSIMLoss
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric
from monai.transforms import Resize
import h5py
import pdb

def denorm(img):
    denorm_img = img*0.5 + 0.5
    return denorm_img

def minmaxnorm(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img

def eval_net(trainer, eval_loader, config, epochs, device, writer ):
    
    torch.manual_seed(0)

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1)
    psnr_metric = PSNRMetric(max_val=1)

    # Make dir for test output
    # base_dir = 'TPSR_physics/outputs/ContentNet/eval/' + config.tb_comment
    # if not os.path.exists(base_dir):
    #     os.makedirs(base_dir, exist_ok=True)
    
    trainer.eval()
    n_val = len(eval_loader)  # the number of batch
    running_loss = 0
    # writer = Logger( config )
    # sw_inferer = SlidingWindowInferer( roi_size=(80,80),
    #                                    sw_batch_size=1,
    #                                    overlap=0 )
    
    dict_val_imgs = {}
    val_img_outputs = []

    ContentNet_running_ssim_value = 0
    ContentNet_running_psnr_value = 0
    ContentNet_mean_ssim_value = 0
    ContentNet_mean_psnr_value = 0
    n_samples = 0

    dict_ContentNet_mean_metric_tb = {}

    with tqdm(total=n_val, 
              desc='Validation', 
              unit='imgs', 
              ncols=180,
              ascii=' ==') as pbar:
        
        with torch.no_grad():
            assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
            """
            In distributed mode, calling the set_epoch() method at the beginning of each epoch
            before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
            Otherwise, the same ordering will be always used.        
            """
            # test_sampler.set_epoch( epochs )
                        
            for step, img in enumerate(eval_loader, len(eval_loader)):

                img["data_A"]    = img["data_A"].to( device )
                           
                # Concat for ContentNet
                concat_A   = img["data_A"] # B x 3 x H x W
                canny_edge = img["data_A_edge"].float() # B x 1 x H x W
                

                # Using non random vector
                trainer.eval()
                c_a, s_a_prime = trainer.gen_a.encode( concat_A )
                x_a_recon = trainer.gen_a.decode( c_a[-1], s_a_prime )
                trainer.train()

                
                # SSIM 및 PSNR 계산
                ssim_value = ssim_metric(y_pred=x_a_recon, y= img["data_A"][:])
                psnr_value = psnr_metric(y_pred=x_a_recon, y= img["data_A"][:])
                
                # pdb.set_trace()

                ContentNet_running_ssim_value += torch.sum(ssim_value).item() # 배치별로 값이 하나씩 나와서 일단 sum해주고 itme으로 값만 가지고 온다.
                ContentNet_running_psnr_value += torch.sum(psnr_value).item() # 배치별로 값이 하나씩 나와서 일단 sum해주고 itme으로 값만 가지고 온다.
                n_samples += img["data_A"].size(0)

                
                pbar.update(img['data_A'].shape[0])
                
                

                # =================================================================================== #
                #                                 3. Miscellaneous                                    #
                # =================================================================================== #

                # For ContentNet
                HR_t1_center = torch.Tensor(img["data_A"][:config.tb_display_size].cpu())

                c_a_ds_1st = torch.Tensor(c_a[0][:config.tb_display_size, 0:1].cpu())  # config.tb_display_size x 128 x 64 x 128 중에 첫 번째 ch의 feature map만 가지고 옴
                c_a_ds_2nd = torch.Tensor(c_a[1][:config.tb_display_size, 0:1].cpu())  # config.tb_display_size x 256 x 32 x 64  중에 첫 번째 ch의 feature map만 가지고 옴
                c_a_final  = torch.Tensor(c_a[-1][:config.tb_display_size, 0:1].cpu()) # config.tb_display_size x 256 x 32 x 64  중에 첫 번째 ch의 feature map만 가지고 옴
                # c_a_ds_3rd = torch.Tensor(c_a[2][:config.tb_display_size, :].cpu()) # config.tb_display_size x 512 x 16 x 32

                resize_transform = Resize(spatial_size=HR_t1_center.shape[2:]) # Resize(spatial_size=(128,256))

                # print(f"\n\nHR_t1_center.shape[2:]: {HR_t1_center.shape[2:]}")
                # print(f"\nc_a_ds_1st.shape: {c_a_ds_1st.shape}")

                # pdb.set_trace()

            
                resize_c_a_ds_1st = torch.Tensor(torch.stack([resize_transform(c_a_ds_1st[i][:]) for i in range(c_a_ds_1st.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_a_ds_2nd = torch.Tensor(torch.stack([resize_transform(c_a_ds_2nd[i][:]) for i in range(c_a_ds_2nd.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_a_final  = torch.Tensor(torch.stack([resize_transform(c_a_final[i][:]) for i in range(c_a_final.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                # c_a_ds_LH_3rd = torch.Tensor(torch.stack([resize_transform(c_a_ds_3rd[i][0:1]) for i in range(c_a_ds_3rd.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                
                # print(f'HR_t1_center:{HR_t1_center.shape}')
                # print(f'c_a:{c_a.shape}')
                # print(f'x_ab:{x_ab.shape}')
                # print(f'LR_t2_center:{LR_t2_center.shape}')
                HR_t1_recon_center = torch.Tensor(x_a_recon[:config.tb_display_size].cpu())
                canny_edge = torch.Tensor(canny_edge[:config.tb_display_size])

                contentNet_img_tensor = torch.cat([resize_c_a_ds_1st,
                                                   resize_c_a_ds_2nd,
                                                   resize_c_a_final,
                                                   canny_edge,
                                                   HR_t1_recon_center,
                                                   HR_t1_center], dim=0)
                
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
            writer.image_summary( tag        = 'val/ContentNet--caDs1st--caDs2nd--caFinal--cannyEdge--xaRecon--xaCenter',
                                    img_tensor = contentNet_img_grid,
                                    step       = epochs + 1 )
                

                
            ContentNet_mean_ssim_value = ContentNet_running_ssim_value / n_samples
            ContentNet_mean_psnr_value = ContentNet_running_psnr_value / n_samples
            
            dict_ContentNet_mean_metric_tb['val/SSIM'] = ContentNet_mean_ssim_value
            dict_ContentNet_mean_metric_tb['val/PSNR'] = ContentNet_mean_psnr_value

            # Visualize loss graphs
            for tag, value in dict_ContentNet_mean_metric_tb.items():
                # print(f"Type of value: {type(value)}")
                print(f"{tag}: {value:.4f}")
                # if isinstance(value, monai.data.meta_tensor.MetaTensor):
                    # value = torch.Tensor(value.cpu())  # Convert MetaTensor to torch.Tensor
                writer.scalar_summary(tag, value, epochs+1)  # Use .item() to get a Python number from a tensor containing a single value
            
            
            
            # Save images
            # if (epochs + 1) % 20 == 0:                
            #     # write_2images( image_outputs     = stacked_dusnet_img_tensor,
            #     #                display_image_num = config.tb_display_size,
            #     #                image_directory   = config.val_sample_dir_stacked_dsunet + config.tb_comment,
            #     #                postfix           = 'val_%04d_' % (epochs + 1),
            #     #                munit             = False )
            
            # #-------------------------------------------------------------------------------------------------------#
            #     # Data save
            #     data_save_dir = os.path.join(base_dir, f"output_data_1.2mm_{config.slice_thickness}mm_{epochs+1}.h5")
            #     with h5py.File(data_save_dir, 'w') as f:

            #         # data_A
            #         f.create_dataset("combined_data_A",    data=data_A_list)

            #         # data_B
            #         f.create_dataset("combined_data_B",    data=data_B_list)
            #         f.create_dataset("combined_data_B_HR", data=data_B_HR_list)

            #         # data_A_to_B
            #         f.create_dataset("combined_data_AB", data=data_AB_list)

            #         # c_a
            #         f.create_dataset("data_c_A",  data=c_a_list)

            #         # Stacked_dsunet output
            #         f.create_dataset("data_B_SR", data=refine_sr_img_list)
                
            #-------------------------------------------------------------------------------------------------------#
