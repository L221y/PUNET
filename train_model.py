import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_LIDC_crops import LIDC_CROPS
from probabilistic_unet import ProbabilisticUnet
import pickle
import os

lr = 1e-5
l2_reg = 1e-6
lr_decay_every = 5   # decay LR after this many epochs
lr_decay = 0.95

out_dir = 'outputs/2'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('Folder already exists. Existing models and training logs will be replaced')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is %s" %(device))
dataset = LIDC_CROPS(dataset_location = 'lidc_crops_train/train')
dataset_test=LIDC_CROPS(dataset_location='lidc_crops_test/test')
dataset_size = len(dataset)
dataset_test_size=len(dataset_test)
indices_train = list(range(dataset_size))
indices_test=list(range(dataset_test_size))
#indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
#np.random.shuffle(indices)
train_indices, test_indices = indices_train, indices_test
#train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=20, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
epochs = 10

train_loss = []
test_loss = []
best_val_loss = 999.0

for epoch in range(epochs):
    net.train()
    loss_train = 0
    loss_segmentation = 0
    for step, (patch, mask) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        loss = -elbo 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.detach().cpu().item()

        if step%1==0:
            print('[Ep ', epoch+1, (step+1), ' of ', len(train_loader) ,'] train loss: ', loss_train/(step+1))
    
    
    loss_train /= len(train_loader)
    net.eval()
    loss_val = 0
    
    with torch.no_grad():
        for step, (patch, mask) in enumerate(test_loader): 
            patch = patch.cuda()
            mask = mask.cuda()
            mask = torch.unsqueeze(mask,1)
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            loss = -elbo 
            
            loss_val += loss.detach().cpu().item()
            
    # end of validation
    loss_val /= len(test_loader)
    
    train_loss.append(loss_train)
    test_loss.append(loss_val)
    
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)
    
    secheduler.step()
    
    # save best model checkpoint
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        fname = 'model_dict_10Epoch.pth'
        torch.save(net.state_dict(), os.path.join(out_dir, fname))
        print('model saved at epoch: ', epoch+1)

print('Finished training')
# save loss curves        
plt.figure()
plt.plot(train_loss)
plt.title('train loss')
fname = os.path.join(out_dir,'loss_train.png')
plt.savefig(fname)
plt.close()

plt.figure()
plt.plot(test_loss)
plt.title('val loss')
fname = os.path.join(out_dir,'loss_val.png')
plt.savefig(fname)
plt.close()
# plt.show()

# Saving logs
log_name = os.path.join(out_dir, "logging.txt")
with open(log_name, 'w') as result_file:
    result_file.write('Logging... \n')
    result_file.write('Validation loss ')
    result_file.write(str(test_loss))
    result_file.write('\nTraining loss  ')
    result_file.write(str(train_loss))
    
