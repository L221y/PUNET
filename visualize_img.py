# loads a trained model and saves some results on the disk

# The trained model dict is loaded from directory 'cpk_directory' and results are saved in 'out_dir/visual_results'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_LIDC_crops import LIDC_CROPS
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pickle
import os
from torchvision import ops

def IoU(mask1,mask2):
    mask1_area=np.count_nonzero(mask1 == 1)
    mask2_area=np.count_nonzero(mask2 ==1)

    intersection = np.count_nonzero(np.logical_and(mask1==1,mask2==1))

    if mask1_area == 0 and mask2_area==0:
        iou=intersection/1
    else:    
        iou=intersection/(mask1_area+mask2_area-intersection)
    return iou   

# checkpoint directory
cpk_directory = 'outputs/1'     # a trained model is provided in this directory. 
print('Using the trained model from directory: ', cpk_directory)
if not os.path.exists(cpk_directory):
    raise ValueError('Please specify the out_dir in visualize.py which contains the trained model dict')

out_dir = 'outputs/4'  # results will be saved in 'out_dir/visual_results'
save_dir = os.path.join(out_dir, 'visual_results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print('Folder already exists, overwriting previous results')

batch_size_val = 4    
save_batches_n = 1     # save this many batches
samples_per_example = 4
    
# data
dataset = LIDC_CROPS(dataset_location = 'lidc_crops_val/val2')
dataset_size = len(dataset)
indices = list(range(dataset_size))

train_indices, test_indices = indices, indices
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=test_sampler)
print("Number of test patches:", len(test_indices))

# network
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.cuda()

# load pretrained model
cpk_name = os.path.join(cpk_directory, 'model_dict.pth')
net.load_state_dict(torch.load(cpk_name))

net.eval()
with torch.no_grad():
    for step, (patch, mask) in enumerate(test_loader):
        if step >= save_batches_n:
            break
        patch = patch.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask,1)
        print(mask.shape)

        mask_zero = np.zeros((128,128),dtype=np.float32)
        mask_zero = torch.from_numpy(mask_zero)
        mask_zero = mask_zero.cuda() 
        mask_zero = torch.unsqueeze(mask_zero,1)
        #print(patch.shape)
        mask_zero = mask_zero.view(1,128,128)
        mask_zero = mask_zero.repeat(batch_size_val,1,1,1)
        #print(mask.shape)
        output_samples = []
        dychapeauy=0

        mask_out1 = (mask[0, 0, :,:].detach().cpu().numpy() >= 0.5).astype(np.float)
        mask_out2 = (mask[1, 0, :,:].detach().cpu().numpy() >= 0.5).astype(np.float)
        mask_out3 = (mask[2, 0, :,:].detach().cpu().numpy() >= 0.5).astype(np.float)
        mask_out4 = (mask[3, 0, :,:].detach().cpu().numpy() >= 0.5).astype(np.float)
        dypy=6-(IoU(mask_out1,mask_out2)+IoU(mask_out1,mask_out3)+IoU(mask_out1,mask_out4)+IoU(mask_out2,mask_out3)+IoU(mask_out2,mask_out4)+IoU(mask_out3,mask_out4))
        E2=dypy/6

        for i in range(samples_per_example):
            net.forward(patch, mask_zero, training=True)
            output_samples.append( torch.sigmoid(net.sample()).detach().cpu().numpy() )
            #print(output_samples.shape)
        
        output_samples1 = (output_samples[0][0, 0, :, :] > 0.5).astype(np.float)
        output_samples2 = (output_samples[1][0, 0, :, :] > 0.5).astype(np.float)
        output_samples3 = (output_samples[2][0, 0, :, :] > 0.5).astype(np.float)
        output_samples4 = (output_samples[3][0, 0, :, :] > 0.5).astype(np.float)

        dycpyc=6-(IoU(output_samples1,output_samples2)+IoU(output_samples1,output_samples3)+IoU(output_samples1,output_samples4)+IoU(output_samples2,output_samples3)+IoU(output_samples2,output_samples4)+IoU(output_samples3,output_samples4))
        E3=dycpyc/6

        #for k in range(0):    # for all items in batch
        patch_out = patch[0, 0, :,:].detach().cpu().numpy()
        mask_zero_out = mask_zero[0, 0, :,:].detach().cpu().numpy()
        # pred_out = pred_mask[k, 0, :,:].detach().cpu().numpy()
        plt.figure()
            
        plt.subplot(3,2,1)
        plt.imshow(patch_out)
        plt.title('patch')
        plt.axis('off')
        plt.subplot(3,2,2)
        plt.imshow(mask_zero_out)
        plt.title('GT Mask')
        plt.axis('off')
        
        for j in range(len(output_samples)):  # for all output samples
            plt.subplot(3, 2, j+3)
            plt.imshow(output_samples[j][0, 0, :, :])
            plt.title('prediction #'+str(j+1))
            plt.axis('off')

        fname = os.path.join(save_dir, 'result_'+str(step)+'_'+str(0)+'.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

        for k in range(patch.shape[0]):    # for all items in batch
            mask_out = mask[k, 0, :,:].detach().cpu().numpy()
        
            for j in range(len(output_samples)):  # for all output samples

                mask_out2 = mask[j, 0, :,:].detach().cpu().numpy()
                mask_out_threshold=mask_out2.astype(np.float)
                output_samples_threshold=output_samples[j][k, 0, :, :].astype(np.float)

                #print(type(mask_out_threshold))

                thresholded_mask = (mask_out_threshold >= 0.5).astype(np.float)
                thresholded_output_samples = (output_samples_threshold> 0.5).astype(np.float)

                iou=IoU(thresholded_mask,thresholded_output_samples)         
                dychapeauy=dychapeauy+1-iou   

        
E1=dychapeauy/(patch.shape[0]*len(output_samples))
print(E1)
print(E2)
print(E3)
D=2*E1-E2-E3
print(D)

 



print('Finished saving images')
