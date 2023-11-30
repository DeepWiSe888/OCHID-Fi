import time
import numpy as np
import math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import  gc
import scipy.io as sio
from networks.ochid_fi  import OCHID_Fi
from dataset import * 
from config import *

def test_epoch(model, test_loader, bound_width, bound_height, bound_depth, pck_result_path, arg):
    print(arg)
    model.eval()

    count = 0
    bound_box = math.sqrt(bound_width**2 + bound_height**2 + bound_depth**2)   
    alpha = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    num_correct_3d = torch.zeros(len(alpha), 21).to(DEVICE)
    
    with torch.no_grad():
        for test_x,test_y_t,test_y_gt,test_phi in tqdm(test_loader, leave=False):
            test_x,test_y_t,test_y_gt,test_phi = test_x.to(DEVICE), test_y_t.to(DEVICE),test_y_gt.to(DEVICE), test_phi.to(DEVICE)
            xy, depth = model(test_x)
            xy = (xy - 0.5)/80
            xy = xy.detach()
            depth = depth.detach()
            sum_square_xy = torch.sum(torch.square(xy - test_y_gt[:,:,:2]), dim=2)
            sum_square_depth = torch.square(torch.squeeze(depth) - test_y_gt[:,:,2])
            dis_sqrt_torch_3d = torch.sqrt(torch.add(sum_square_xy, sum_square_depth))
            for i in range(len(alpha)):
                threshold_val = bound_box * alpha[i]
                torch_correct_3d = torch.le(dis_sqrt_torch_3d,threshold_val)
                num_correct_3d[i,:] += torch.count_nonzero(torch_correct_3d,dim=0)
            count += test_y_gt.shape[0]

        for i in range(len(alpha)):
            print('PCK@ ', alpha[i])
            temp = (num_correct_3d[i,:]/count)
            print(temp)
        keypoints_array = (num_correct_3d/count).cpu().numpy()
        sio.savemat(pck_result_path+'.mat', {'pck': keypoints_array, 'count':count})

