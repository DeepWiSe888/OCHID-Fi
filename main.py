import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import  gc
import scipy.io as sio
from networks.ochid_fi  import OCHID_Fi, RegressionDisparity,PseudoLabelGenerator2d
from torch.optim.lr_scheduler import LambdaLR
from loss import JointsKLLoss,AI_MSELoss
from dataset import * 
from config import *


def train(model, train_normal_iter,train_occlusion_iter, optimizer_f, optimizer_h, optimizer_d, optimizer_h_adv,
          criterion,regression_disparity):
    
    model.train()
    total_normal_loss = total_normal_true = total_false_loss = total_true_loss = 0
    count = 0
    rd_weight = 1
    alpha = 0.5
    for e in tqdm(range(2000)):
        x_no,y_t_no,y_gt_no,phi_no = next(train_normal_iter)
        x_oc,y_t_oc,y_gt_oc,phi_oc = next(train_occlusion_iter)
        count = count+1       

        #Step A 
        optimizer_f.zero_grad()
        optimizer_d.zero_grad()
        optimizer_h.zero_grad()
        optimizer_h_adv.zero_grad()
        xy, depth, xy_map, xy_map_adv, depth_adv = model(x_no,'train')
        loss_no_dis = regression_disparity(xy_map,xy_map_adv,mode='min')*rd_weight
        loss_xy = criterion(xy, y_t_no[:,:,:2], y_gt_no[:,:,:2],phi_no[:,:,:2],alpha)
        loss_depth = 10*criterion(torch.squeeze(depth), y_t_no[:,:,2], y_gt_no[:,:,2],phi_no[:,:,2],alpha)
        loss_y = loss_xy + loss_depth
        loss_no_tot = loss_y + loss_no_dis
        loss_no_tot.backward()
        optimizer_f.step()
        optimizer_d.step()
        optimizer_h.step()
        optimizer_h_adv.step()
        #step B 
        optimizer_h_adv.zero_grad()
        xy, depth, xy_map, xy_map_adv, depth_adv = model(x_oc,'train')
        loss_ground_false = regression_disparity(xy_map,xy_map_adv,mode='max')*rd_weight
        loss_ground_false.backward()
        optimizer_h_adv.step()
        #step C 
        optimizer_f.zero_grad()
        xy, depth, xy_map, xy_map_adv, depth_adv = model(x_oc,'train')
        loss_ground_truth = regression_disparity(xy_map,xy_map_adv,mode='min')*rd_weight
        loss_ground_truth.backward()
        optimizer_f.step()

        model.step()
        total_normal_loss += loss_no_tot.item()
        total_normal_true += loss_no_dis.item()
        total_false_loss += loss_ground_false.item()
        total_true_loss += loss_ground_truth.item()

    total_normal_loss /= count
    total_normal_true /= count
    total_false_loss /= count
    total_true_loss /= count
    return total_normal_loss,total_normal_true,total_false_loss,total_true_loss

def pre_train(model, train_loader, optimizer_f, optimizer_h, optimizer_d, criterion):
    
    model.train()

    total_loss = total_xy = total_depth = 0
    count = 0
    alpha = 0.5
    for train_x,train_y_t,train_y_gt,train_phi in tqdm(train_loader, leave=False):
        count = count + 1
        train_x,train_y_t,train_y_gt,train_phi = train_x.to(DEVICE), train_y_t.to(DEVICE),train_y_gt.to(DEVICE), train_phi.to(DEVICE)
        
        xy, depth = model(train_x,'pre_train')

        loss_xy = criterion(xy, train_y_t[:,:,:2], train_y_gt[:,:,:2],train_phi[:,:,:2],alpha)
        loss_depth = 10*criterion(torch.squeeze(depth), train_y_t[:,:,2], train_y_gt[:,:,2], train_phi[:,:,2], alpha)

        loss_tot = loss_xy + loss_depth

        optimizer_f.zero_grad()
        optimizer_h.zero_grad()
        optimizer_d.zero_grad()
        loss_tot.backward()
        optimizer_f.step()
        optimizer_h.step()
        optimizer_d.step()
        total_loss += loss_tot.item()
        total_xy += loss_xy.item()
        total_depth += loss_depth.item()
    
    total_loss /= count
    total_xy /= count
    total_depth /= count
    return total_loss, total_xy, total_depth

def validate(model, val_loader, criterion):

    model.eval()

    total_loss = total_xy = total_depth = 0
    count = 0
    alpha = 0.5
    for val_x,val_y_t,val_y_gt,val_phi in tqdm(val_loader, leave=False):
        count = count + 1
        val_x,val_y_t,val_y_gt,val_phi = val_x.to(DEVICE), val_y_t.to(DEVICE),val_y_gt.to(DEVICE), val_phi.to(DEVICE)
        
        xy, depth = model(val_x)

        loss_xy = criterion(xy, val_y_t[:,:,:2], val_y_gt[:,:,:2],val_phi[:,:,:2],alpha)
        loss_depth = 10*criterion(torch.squeeze(depth), val_y_t[:,:,2], val_y_gt[:,:,2], val_phi[:,:,2], alpha)

        loss_tot = loss_xy + loss_depth

        total_loss += loss_tot.item()
        total_xy += loss_xy.item()
        total_depth += loss_depth.item()
    
    total_loss /= count
    total_xy /= count
    total_depth /= count
    return total_loss, total_xy, total_depth


if __name__ == '__main__':
    model = OCHID_Fi().to(DEVICE)
    model = model.double()
    need_pre_train = False
    lines1 = []
    lines2 = []

    with open(file_no_path + "name.txt", "r") as f:
        lines1 = f.readlines()
    with open(file_oc_path + "name.txt", "r") as f:
        lines2 = f.readlines()

    np.random.seed(0)
    np.random.shuffle(lines1)
    np.random.seed(None)

    np.random.seed(0)
    np.random.shuffle(lines2)
    np.random.seed(None)

    num_test = int(len(lines1) * 0.2)
    num_train = len(lines1) - num_test

    train_normal_dataset = RFDataset(lines1[:num_train], num_train, x_no_path, y_no_path)
    train_normal_loader = DataLoader(train_normal_dataset, batch_size=8, shuffle=True)

    val_normal_dataset = RFDataset(lines1[num_train:], num_test, x_no_path, y_no_path)
    val_normal_loader = DataLoader(val_normal_dataset, batch_size=8, shuffle=True)

    train_occlusion_dataset = RFDataset(lines2[:num_train], num_train, x_oc_path, y_oc_path)
    train_occlusion_loader = DataLoader(train_occlusion_dataset, batch_size=8, shuffle=True)

    val_occlusion_dataset = RFDataset(lines2[num_train:],num_test, x_oc_path, y_oc_path)
    val_occlusion_loader = DataLoader(val_occlusion_dataset, batch_size=8, shuffle=True)

    train_normal_iter = ForeverDataIterator(train_normal_loader, DEVICE)
    train_occlusion_iter = ForeverDataIterator(train_occlusion_loader, DEVICE)

    criterion_kl = JointsKLLoss()
    criterion_mse = AI_MSELoss()
    pseudo_label_generator = PseudoLabelGenerator2d(21, height=80,width=80,sigma = 1)
    regression_disparity = RegressionDisparity(pseudo_label_generator, criterion_kl)

    optimizer_f = torch.optim.Adam(model.feature_extractor.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(model.head_depth.parameters(), lr=1e-3)
    optimizer_h = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    optimizer_h_adv = torch.optim.Adam(model.head_adv.parameters(), lr=1e-3)

    lr_decay_function = lambda x: 0.1 * (1. + 0.0001 * float(x)) ** (0.75)

    lr_scheduler_f = LambdaLR(optimizer_f, lr_decay_function)
    lr_scheduler_d = LambdaLR(optimizer_d, lr_decay_function)
    lr_scheduler_h = LambdaLR(optimizer_h, lr_decay_function)
    lr_scheduler_h_adv = LambdaLR(optimizer_h_adv, lr_decay_function)
   
    #pre-train
    if need_pre_train:
        epochs = 20
        print('*****pre_train*****')
        for e in tqdm(range(epochs)):
            total_loss, total_mse_xy, total_mse_depth = pre_train(model, train_normal_loader, optimizer_f, optimizer_h, optimizer_d, criterion=criterion_mse)       
            lr_scheduler_f.step()
            lr_scheduler_h.step()
            lr_scheduler_d.step()
            print("Epoch: {}/{}   ".format(e + 1, 30),
              "total_loss: {:.6f}".format(total_loss),
              "total_mse_xy: {:.6f}  ".format(total_mse_xy),
              "total_mse_depth: {:.6f}".format(total_mse_depth))
            torch.save(model.state_dict(), model_path + 'ochid_pre_train_' +str(e)+ '.pth')
    
    #train_al
    epochs = 25
    print('*****training*****')
    for e in tqdm(range(epochs)):
        normal_loss,normal_true,false_loss,true_loss = \
            train(model, train_normal_iter, train_occlusion_iter, optimizer_f=optimizer_f, optimizer_h=optimizer_h, optimizer_d=optimizer_d, 
                  optimizer_h_adv=optimizer_h_adv, criterion = criterion_mse, regression_disparity = regression_disparity)
        lr_scheduler_f.step()
        lr_scheduler_h.step()
        lr_scheduler_d.step()
        lr_scheduler_h_adv.step()
            
        torch.cuda.empty_cache()
        gc.collect()
        print("Epoch: {}/{}   ".format(e + 1, 30),
              "total_normal_loss: {:.6f}".format(normal_loss),
              "normal_true: {:.6f}".format(normal_true),
              "false_loss: {:.6f}  ".format(false_loss),
              "true_loss: {:.6f}  ".format(true_loss)),
        torch.save(model.state_dict(), model_path + 'ochid_al_' +str(e)+ '.pth')