import os
import pdb
import h5py
import torch
import numpy as np
import random
from torch.utils import data
from PIL import Image
from monai.transforms import Transform, MapTransform, LoadImage, Compose, NormalizeIntensityd, Flipd, RandSpatialCrop, ResizeWithPadOrCropd, CropForeground, Resize, SpatialCrop, RandRotate
from monai.inferers.inferer import SlidingWindowInferer
from monai.config import IndexSelection, KeysCollection
from skimage.feature import canny
import pywt
    
class IXI_Dataset(data.Dataset):
    """Dataset class for the IXI dataset."""

    def __init__(self, config, mode):
        """Initialize and preprocess the IXI dataset."""

        # Train or test
        self.mode = mode

        self.config = config
        # Slice thickness of T2
        # self.slice_thickness = config.slice_thickness

        # Data directory
        self.ixi_h5_1_2mm_dir = config.ixi_h5_1_2mm_dir 

        if self.mode == "train":
            self.train_data_dir, self.nb_train_imgs = self.preprocess(config)
        elif self.mode == "test":
            self.test_data_dir, self.nb_test_imgs  = self.preprocess(config)

        self.data_dict = {}         

        # Transforms
        # self.resize = Resize(spatial_size=(128,256))

        
    def preprocess(self, config):
        # Set the # of imgs utilized for train and test
        
        if self.mode == 'train':                        
            data_dir = os.path.join(self.ixi_h5_1_2mm_dir, self.mode, f"output_data_1.2mm_2.4mm_4.8mm_50.h5")
            
            if config.nb_train_imgs is not None:
                assert isinstance(config.nb_train_imgs, int) and config.nb_train_imgs > 0, "config.nb_train_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_train_imgs <= len(f["data_B_HR"]), "config.nb_train_imgs should not exceed the total number of samples"
                nb_train_imgs = config.nb_train_imgs
            
            else: # config.nb_train_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 training 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_train_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_train_imgs
            
        elif self.mode == 'test':
            data_dir = os.path.join(self.ixi_h5_1_2mm_dir, self.mode, f"output_data_1.2mm_2.4mm_4.8mm_50.h5")
            
            if config.nb_test_imgs is not None:
                assert isinstance(config.nb_test_imgs, int) and config.nb_test_imgs > 0, "config.nb_test_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                # print(data_dir)
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_test_imgs <= len(f["data_B_HR"]), "config.nb_test_imgs should not exceed the total number of samples"
                nb_test_imgs = config.nb_test_imgs
            
            else: # config.nb_test_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 testing 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_test_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_test_imgs 
     
        
    def transforms(self, data):

        if self.mode == 'train':                
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"],subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
                # RandSpatialCrop(roi_size=(100,100),
                #                 random_size=False),
                # RandSpatialCrop(roi_size=(80,80),
                #                 random_size=False),
                # RandRotate( prob=0.1,
                #             range_x=[0.1, 0.1],
                #             range_y=[0.1, 0.1],
                #             range_z=[0.1, 0.1],
                #             mode='bilinear')        
            ])
            transformed_data = transforms( data )
            return transformed_data
        
        elif self.mode == 'test':
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"], subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
            ])
            transformed_data = transforms( data )
            return transformed_data
        
    def apply_dwt_canny(self, x_a):

        coeffs_hf_comp_hr_t1_1st = []
        coeffs_hf_comp_hr_t1_2nd = []
        # coeffs_hf_comp_hr_t1_3rd = []
        x_a_edge = []
        
        # Apply DWT 2 times
        # for i in range(x_a.shape[0]):
        # print(x_a.shape) # (3, 128, 256)
        coeffs_hr_t1_1st = pywt.dwt2(x_a[0], 'haar') # (3, 64, 128)
        coeffs_hr_t1_2nd = pywt.dwt2(coeffs_hr_t1_1st[0], 'haar') # (3, 32, 64)
        # coeffs_hr_t1_3rd = pywt.dwt2(coeffs_hr_t1_2nd[0], 'haar') # (3, 16, 32)
        
        # print(f"\ncoeffs_hr_t1_1st.shape: {coeffs_hr_t1_1st[1].shape}")
        # print(f"\ncoeffs_hr_t1_2nd.shape: {coeffs_hr_t1_2nd[1].shape}")


        coeffs_hf_comp_hr_t1_1st.append(coeffs_hr_t1_1st[1])
        coeffs_hf_comp_hr_t1_2nd.append(coeffs_hr_t1_2nd[1])
        # coeffs_hf_comp_hr_t1_3rd.append(coeffs_hr_t1_3rd[1])

        # print(f"\ncoeffs_hf_comp_hr_t1_1st.shape: {len(coeffs_hf_comp_hr_t1_1st)}")
        # print(f"\ncoeffs_hf_comp_hr_t1_2nd.shape: {len(coeffs_hf_comp_hr_t1_2nd)}")
        # print(f"\nx_a.shape: {x_a.shape}")
        # print(f"\nx_a[0].shape: {x_a[0].shape}")
        # pdb.set_trace()
        
        # Edge detection by using Canny filter
        x_a_edge_ = canny(x_a[0], sigma=1, low_threshold=0.1, high_threshold=0.2) # (128, 256)
        # np.expand_dims(x_a_edge, axis=0) # (1, 128, 256)
        x_a_edge.append(x_a_edge_)

        return coeffs_hf_comp_hr_t1_1st, coeffs_hf_comp_hr_t1_2nd, x_a_edge

    def apply_canny(self, x_a):

        x_a_edge = []
            
        # Edge detection by using Canny filter
        x_a_edge_ = canny(x_a[0], sigma=1, low_threshold=0.1, high_threshold=0.2) # (128, 256)
        # np.expand_dims(x_a_edge, axis=0) # (1, 128, 256)
        x_a_edge.append(x_a_edge_)

        return x_a_edge

    def __getitem__(self, index):

        """
        __getitem__ 에서는 batch 차원은 없다고 생각하고 data 크기 따지는 것
        Dataloader에서 뱉어낼 때 batch 만큼 차원이 앞에 생기는 것

        Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """

        if self.mode == 'train':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.train_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 

            # Ensure that data is in range [-1,1] --> 이미 subject 별로 min-max norm은 되어있음
            # Center
            # A    = (A - 0.5)/0.5
            # B    = (B - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            
            # print(f'A.shape:{A.shape}') # A.shape:(B,1,128,256)
            # print(f'B_SR_2to1.shape:{B_SR_2to1.shape}') # A.shape:(B,1,128,256)
            # pdb.set_trace()

            # Apply DWT and Canny filtering
            # coeffs_hf_comp_b_sr_2to1_1st, coeffs_hf_comp_b_sr_2to1_2nd, b_sr_2to1_edge = self.apply_dwt_canny(B_SR_2to1)
            if self.config.crf_domain == "t1":
                cdt_edge = self.apply_canny(A)
            elif self.config.crf_domain == "pd":
                cdt_edge = self.apply_canny(PD)
            elif self.config.crf_domain == "t2":
                cdt_edge = self.apply_canny(B_HR)
            elif self.config.crf_domain == "srt2":    
                cdt_edge = self.apply_canny(B_SR_2to1)

            # Apply resizing        
            resize_2fold = Resize(spatial_size=(64,256), mode="area")
            resize_4fold = Resize(spatial_size=(32,256), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                         "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
                        #  "coeffs_hf_comp_B_SR_2to1_1st": coeffs_hf_comp_b_sr_2to1_1st,
                        #  "coeffs_hf_comp_B_SR_2to1_2nd": coeffs_hf_comp_b_sr_2to1_2nd,
                         "data_cdt_edge" : cdt_edge}
        
            self.data_dict.update(data_dict)
           
            
            # Apply the Resize with pad or crop
            # processed_data_dict = self.transforms(self.data_dict)
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"     ])),                                   
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
                                    # "coeffs_hf_comp_B_SR_2to1_1st": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_1st"])).squeeze(),
                                    # "coeffs_hf_comp_B_SR_2to1_2nd": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_2nd"])).squeeze(),
                                    "data_cdt_edge"         : torch.from_numpy(np.where(np.array(processed_data_dict["data_cdt_edge"]),1,0))
            }

            # print(f"\ncombined_data_A:{processed_data_dict['combined_data_A'].shape}\n")
            # print(f"coeffs_hf_comp_A_1st:{processed_data_dict['coeffs_hf_comp_A_1st'].shape}\n")
            # print(f"coeffs_hf_comp_A_2nd:{processed_data_dict['coeffs_hf_comp_A_2nd'].shape}\n")
            # print(f"coeffs_hf_comp_A_3rd:{processed_data_dict['coeffs_hf_comp_A_3rd'].shape}\n")
            # print(f"combined_data_A_edge:{processed_data_dict['combined_data_A_edge'].shape}\n")

            return processed_data_dict

        elif self.mode == 'test':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.test_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 

            # Ensure that data is in range [-1,1] --> 이미 subject 별로 min-max norm은 되어있음
            # Center
            # A    = (A - 0.5)/0.5
            # B    = (B - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5

            # Apply DWT and Canny filtering
            # coeffs_hf_comp_b_sr_2to1_1st, coeffs_hf_comp_b_sr_2to1_2nd, b_sr_2to1_edge = self.apply_dwt_canny(B_SR_2to1)
            b_sr_2to1_edge = self.apply_canny(B_SR_2to1)
            
            # if self.config.crf_domain == "t1":
            #     cdt_edge = self.apply_canny(A)
            # elif self.config.crf_domain == "pd":
            #     cdt_edge = self.apply_canny(PD)
            # elif self.config.crf_domain == "t2":
            #     cdt_edge = self.apply_canny(B_HR)

            # Apply resizing        
            resize_2fold = Resize(spatial_size=(64,256), mode="area")
            resize_4fold = Resize(spatial_size=(32,256), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                         "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
                        #  "coeffs_hf_comp_B_SR_2to1_1st": coeffs_hf_comp_b_sr_2to1_1st,
                        #  "coeffs_hf_comp_B_SR_2to1_2nd": coeffs_hf_comp_b_sr_2to1_2nd,
                        #  "data_B_SR_2to1_edge" : b_sr_2to1_edge}                                          
            }
            self.data_dict.update(data_dict)
           
            
            # Apply the Resize with pad or crop
            # processed_data_dict = self.transforms(self.data_dict)
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"  ])),
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
                                    # "coeffs_hf_comp_B_SR_2to1_1st": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_1st"])).squeeze(),
                                    # "coeffs_hf_comp_B_SR_2to1_2nd": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_2nd"])).squeeze(),
                                    # "data_B_SR_2to1_edge"  : torch.from_numpy(np.where(np.array(processed_data_dict["data_B_SR_2to1_edge"]),1,0))
            }

            # print(f"\ncombined_data_A:{processed_data_dict['combined_data_A'].shape}\n")
            # print(f"coeffs_hf_comp_A_1st:{processed_data_dict['coeffs_hf_comp_A_1st'].shape}\n")
            # print(f"coeffs_hf_comp_A_2nd:{processed_data_dict['coeffs_hf_comp_A_2nd'].shape}\n")
            # print(f"coeffs_hf_comp_A_3rd:{processed_data_dict['coeffs_hf_comp_A_3rd'].shape}\n")
            # print(f"combined_data_A_edge:{processed_data_dict['combined_data_A_edge'].shape}\n")

            return processed_data_dict


    def __len__(self):
        """Returns the number of samples in the dataset."""

        if self.mode == 'train':
            # os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 HDF5 파일에서 동시에 읽고 쓰는 작업을 관리하기 위한 환경 변수 설정입니다.
            # HDF5 파일은 여러 프로세스 또는 스레드에서 동시에 접근할 수 있습니다.
            # 이때 HDF5_USE_FILE_LOCKING 환경 변수를 "TRUE"로 설정하면,
            # HDF5 라이브러리가 파일 잠금(file locking) 메커니즘을 사용하여 동시에 발생하는 접근 충돌을 방지합니다.
            # 따라서 os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 현재 스크립트에서 사용되는 HDF5 파일의 동시 접근 문제를 해결하기 위해 파일 잠금(file locking) 기능을 활성화하는 역할을 합니다.
            # Load
            return self.nb_train_imgs
        
        elif self.mode == 'test':

            return self.nb_test_imgs
        
#-----------------------------------------------------------------------------------------------------------#
        
class HCP_Dataset(data.Dataset):
    """Dataset class for the HCP dataset."""

    def __init__(self, config, mode):
        """Initialize and preprocess the HCP dataset."""

        # Train or test
        self.mode = mode

        # Slice thickness of T2
        # self.slice_thickness = config.slice_thickness

        # Data directory
        self.hcp_h5_0_7mm_dir = config.hcp_h5_0_7mm_dir
        # self.hcp_h5_2mm_dir   = config.hcp_h5_2mm_dir
        # self.hcp_h5_3mm_dir   = config.hcp_h5_3mm_dir
        # self.hcp_h5_4mm_dir   = config.hcp_h5_4mm_dir
        # self.hcp_h5_5mm_dir   = config.hcp_h5_5mm_dir 

        if self.mode == "train":
            self.train_data_dir, self.nb_train_imgs = self.preprocess(config)
        elif self.mode == "test":
            self.test_data_dir, self.nb_test_imgs  = self.preprocess(config)

        self.data_dict = {}           

        # Transforms
        # self.resize = Resize(spatial_size=(320,320))

        
    def preprocess(self, config):
        # Set the # of imgs utilized for train and test
        
        if self.mode == 'train':                        
            data_dir = os.path.join(self.hcp_h5_0_7mm_dir, self.mode, f"output_data_0.7mm_1.4mm_2.8mm_5.6mm_50.h5")
            
            if config.nb_train_imgs is not None:
                assert isinstance(config.nb_train_imgs, int) and config.nb_train_imgs > 0, "config.nb_train_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_train_imgs <= len(f["data_B_HR"]), "config.nb_train_imgs should not exceed the total number of samples"
                nb_train_imgs = config.nb_train_imgs
            
            else: # config.nb_train_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 training 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_train_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_train_imgs
            
        elif self.mode == 'test':
            data_dir = os.path.join(self.hcp_h5_0_7mm_dir, self.mode, f"output_data_0.7mm_1.4mm_2.8mm_5.6mm_50.h5")
            
            if config.nb_test_imgs is not None:
                assert isinstance(config.nb_test_imgs, int) and config.nb_test_imgs > 0, "config.nb_test_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_test_imgs <= len(f["data_B_HR"]), "config.nb_test_imgs should not exceed the total number of samples"
                nb_test_imgs = config.nb_test_imgs
            
            else: # config.nb_test_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 testing 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_test_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_test_imgs    
     
        
    def transforms(self, data):

        if self.mode == 'train':                
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"],subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
                # RandSpatialCrop(roi_size=(100,100),
                #                 random_size=False),
                # RandSpatialCrop(roi_size=(80,80),
                #                 random_size=False),
                # RandRotate( prob=0.1,
                #             range_x=[0.1, 0.1],
                #             range_y=[0.1, 0.1],
                #             range_z=[0.1, 0.1],
                #             mode='bilinear')        
            ])
            transformed_data = transforms( data )
            return transformed_data
        
        elif self.mode == 'test':
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"], subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
            ])
            transformed_data = transforms( data )
            return transformed_data

    def apply_dwt_canny(self, x_a):

        coeffs_hf_comp_hr_t1_1st = []
        coeffs_hf_comp_hr_t1_2nd = []
        # coeffs_hf_comp_hr_t1_3rd = []
        x_a_edge = []
        
        # Apply DWT 2 times
        # for i in range(x_a.shape[0]):
        # print(x_a.shape) # (3, 128, 256)
        coeffs_hr_t1_1st = pywt.dwt2(x_a[0], 'haar') # (3, 64, 128)
        coeffs_hr_t1_2nd = pywt.dwt2(coeffs_hr_t1_1st[0], 'haar') # (3, 32, 64)
        # coeffs_hr_t1_3rd = pywt.dwt2(coeffs_hr_t1_2nd[0], 'haar') # (3, 16, 32)
        
        # print(f"\ncoeffs_hr_t1_1st.shape: {coeffs_hr_t1_1st[1].shape}")
        # print(f"\ncoeffs_hr_t1_2nd.shape: {coeffs_hr_t1_2nd[1].shape}")


        coeffs_hf_comp_hr_t1_1st.append(coeffs_hr_t1_1st[1])
        coeffs_hf_comp_hr_t1_2nd.append(coeffs_hr_t1_2nd[1])
        # coeffs_hf_comp_hr_t1_3rd.append(coeffs_hr_t1_3rd[1])

        # print(f"\ncoeffs_hf_comp_hr_t1_1st.shape: {len(coeffs_hf_comp_hr_t1_1st)}")
        # print(f"\ncoeffs_hf_comp_hr_t1_2nd.shape: {len(coeffs_hf_comp_hr_t1_2nd)}")
        # print(f"\nx_a.shape: {x_a.shape}")
        # print(f"\nx_a[0].shape: {x_a[0].shape}")
        # pdb.set_trace()
        
        # Edge detection by using Canny filter
        x_a_edge_ = canny(x_a[0], sigma=1, low_threshold=0.1, high_threshold=0.2) # (128, 256)
        # np.expand_dims(x_a_edge, axis=0) # (1, 128, 256)
        x_a_edge.append(x_a_edge_)

        return coeffs_hf_comp_hr_t1_1st, coeffs_hf_comp_hr_t1_2nd, x_a_edge

    def apply_canny(self, x_a):

        x_a_edge = []
            
        # Edge detection by using Canny filter
        x_a_edge_ = canny(x_a[0], sigma=1, low_threshold=0.1, high_threshold=0.2) # (128, 256)
        # np.expand_dims(x_a_edge, axis=0) # (1, 128, 256)
        x_a_edge.append(x_a_edge_)

        return x_a_edge
    
    def __getitem__(self, index):
        """
        __getitem__ 에서는 batch 차원은 없다고 생각하고 data 크기 따지는 것
        Dataloader에서 뱉어낼 때 batch 만큼 차원이 앞에 생기는 것

        Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """

        if self.mode == 'train':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.train_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                # PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_81      = np.copy( np.array(f["data_B_81"][index],      dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_8to4 = np.copy( np.array(f["data_B_SR_8to4"][index], dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 

            # Ensure that data is in range [-1,1] --> 이미 subject 별로 min-max norm은 되어있음
            # Center
            # A    = (A - 0.5)/0.5
            # B    = (B - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            
            # print(f'A.shape:{A.shape}') # A.shape:(B,1,128,256)
            # print(f'B_SR_2to1.shape:{B_SR_2to1.shape}') # A.shape:(B,1,128,256)
            # pdb.set_trace()

            # Apply DWT and Canny filtering
            # coeffs_hf_comp_b_sr_2to1_1st, coeffs_hf_comp_b_sr_2to1_2nd, b_sr_2to1_edge = self.apply_dwt_canny(B_SR_2to1)
            # cdt_edge = self.apply_canny(B_SR_2to1)

            if self.config.crf_domain == "t1":
                cdt_edge = self.apply_canny(A)
            # elif self.config.crf_domain == "pd":
            #     cdt_edge = self.apply_canny(PD)
            elif self.config.crf_domain == "t2":
                cdt_edge = self.apply_canny(B_HR)
            elif self.config.crf_domain == "srt2":
                cdt_edge = self.apply_canny(B_SR_2to1)

            # Apply resizing
            H = 320
            W = 320        
            resize_2fold = Resize(spatial_size=(H,int(W/2)), mode="area")
            resize_4fold = Resize(spatial_size=(H,int(W/4)), mode="area")
            resize_8fold = Resize(spatial_size=(H,int(W/8)), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)
            B_8fold = resize_8fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                        #  "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_81"     : B_81,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_8fold"  : B_8fold,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
                        #  "coeffs_hf_comp_B_SR_2to1_1st": coeffs_hf_comp_b_sr_2to1_1st,
                        #  "coeffs_hf_comp_B_SR_2to1_2nd": coeffs_hf_comp_b_sr_2to1_2nd,
                         "data_cdt_edge" : cdt_edge}
        
            self.data_dict.update(data_dict)
           
            
            # Apply the Resize with pad or crop
            # processed_data_dict = self.transforms(self.data_dict)
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    # "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"     ])),                                   
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_81"      : torch.from_numpy(np.array(processed_data_dict["data_B_81"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_8fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_8fold"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
                                    # "coeffs_hf_comp_B_SR_2to1_1st": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_1st"])).squeeze(),
                                    # "coeffs_hf_comp_B_SR_2to1_2nd": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_2nd"])).squeeze(),
                                    "data_cdt_edge"         : torch.from_numpy(np.where(np.array(processed_data_dict["data_cdt_edge"]),1,0))
            }

            # print(f"\ncombined_data_A:{processed_data_dict['combined_data_A'].shape}\n")
            # print(f"coeffs_hf_comp_A_1st:{processed_data_dict['coeffs_hf_comp_A_1st'].shape}\n")
            # print(f"coeffs_hf_comp_A_2nd:{processed_data_dict['coeffs_hf_comp_A_2nd'].shape}\n")
            # print(f"coeffs_hf_comp_A_3rd:{processed_data_dict['coeffs_hf_comp_A_3rd'].shape}\n")
            # print(f"combined_data_A_edge:{processed_data_dict['combined_data_A_edge'].shape}\n")

            return processed_data_dict

        elif self.mode == 'test':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.test_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                # PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_81      = np.copy( np.array(f["data_B_81"][index],      dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_8to4 = np.copy( np.array(f["data_B_SR_8to4"][index], dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 

            # Ensure that data is in range [-1,1] --> 이미 subject 별로 min-max norm은 되어있음
            # Center
            # A    = (A - 0.5)/0.5
            # B    = (B - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5
            # B_HR = (B_HR - 0.5)/0.5

            # Apply DWT and Canny filtering
            # coeffs_hf_comp_b_sr_2to1_1st, coeffs_hf_comp_b_sr_2to1_2nd, b_sr_2to1_edge = self.apply_dwt_canny(B_SR_2to1)
            
            # b_sr_2to1_edge = self.apply_canny(B_SR_2to1)
            
            # if self.config.crf_domain == "t1":
            #     cdt_edge = self.apply_canny(A)
            # elif self.config.crf_domain == "pd":
            #     cdt_edge = self.apply_canny(PD)
            # elif self.config.crf_domain == "t2":
            #     cdt_edge = self.apply_canny(B_HR)

            # Apply resizing
            H = 320
            W = 320        
            resize_2fold = Resize(spatial_size=(H,int(W/2)), mode="area")
            resize_4fold = Resize(spatial_size=(H,int(W/4)), mode="area")
            resize_8fold = Resize(spatial_size=(H,int(W/8)), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)
            B_8fold = resize_8fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                        #  "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_81"     : B_81,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_8fold"  : B_8fold,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
                        #  "coeffs_hf_comp_B_SR_2to1_1st": coeffs_hf_comp_b_sr_2to1_1st,
                        #  "coeffs_hf_comp_B_SR_2to1_2nd": coeffs_hf_comp_b_sr_2to1_2nd,
                        #  "data_cdt_edge" : cdt_edge}
            }
            
        
            self.data_dict.update(data_dict)
           
            
            # Apply the Resize with pad or crop
            # processed_data_dict = self.transforms(self.data_dict)
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    # "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"  ])),
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_81"      : torch.from_numpy(np.array(processed_data_dict["data_B_81"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_8fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_8fold"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
                                    # "coeffs_hf_comp_B_SR_2to1_1st": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_1st"])).squeeze(),
                                    # "coeffs_hf_comp_B_SR_2to1_2nd": torch.from_numpy(np.array(processed_data_dict["coeffs_hf_comp_B_SR_2to1_2nd"])).squeeze(),
                                    # "data_cdt_edge"         : torch.from_numpy(np.where(np.array(processed_data_dict["data_cdt_edge"]),1,0))
            }

            # print(f"\ncombined_data_A:{processed_data_dict['combined_data_A'].shape}\n")
            # print(f"coeffs_hf_comp_A_1st:{processed_data_dict['coeffs_hf_comp_A_1st'].shape}\n")
            # print(f"coeffs_hf_comp_A_2nd:{processed_data_dict['coeffs_hf_comp_A_2nd'].shape}\n")
            # print(f"coeffs_hf_comp_A_3rd:{processed_data_dict['coeffs_hf_comp_A_3rd'].shape}\n")
            # print(f"combined_data_A_edge:{processed_data_dict['combined_data_A_edge'].shape}\n")

            return processed_data_dict


    def __len__(self):
        """Returns the number of samples in the dataset."""

        if self.mode == 'train':
            # os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 HDF5 파일에서 동시에 읽고 쓰는 작업을 관리하기 위한 환경 변수 설정입니다.
            # HDF5 파일은 여러 프로세스 또는 스레드에서 동시에 접근할 수 있습니다.
            # 이때 HDF5_USE_FILE_LOCKING 환경 변수를 "TRUE"로 설정하면,
            # HDF5 라이브러리가 파일 잠금(file locking) 메커니즘을 사용하여 동시에 발생하는 접근 충돌을 방지합니다.
            # 따라서 os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 현재 스크립트에서 사용되는 HDF5 파일의 동시 접근 문제를 해결하기 위해 파일 잠금(file locking) 기능을 활성화하는 역할을 합니다.
            # Load
            return self.nb_train_imgs
        
        elif self.mode == 'test':

            return self.nb_test_imgs

    
    