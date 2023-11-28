import torch
import torch.nn as nn
import torch.nn.functional as F


class L_smooth(nn.Module):
    def __init__(self):
        super(L_smooth, self).__init__()
        
    def forward(self, x, seg_map):
        # semantics-guided enhancement smoothness
        batch_size, _, h_x, w_x = x.size()

        seg_map = nn.Softmax(dim=1)(seg_map)
        seg_h_tv = torch.mean(torch.pow((seg_map[:,:,1:,:]-seg_map[:,:,:h_x-1,:]),2), 1)
        seg_w_tv = torch.mean(torch.pow((seg_map[:,:,:,1:]-seg_map[:,:,:,:w_x-1]),2), 1)
        
        h_tv = torch.mean(torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2), 1)
        w_tv = torch.mean(torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2), 1)

        weighted_h_tv = torch.mul(h_tv, torch.exp(-seg_h_tv)).sum()
        weighted_w_tv = torch.mul(w_tv, torch.exp(-seg_w_tv)).sum()
        
        return (weighted_h_tv + weighted_w_tv) / batch_size


class L_brightness(nn.Module):

    def __init__(self):
        super(L_brightness, self).__init__()
    
    def forward(self, x, seg_map):
        # semantic-wise brightness
        x = torch.mean(x, 1, keepdim=True)
        seg_map = nn.Softmax(dim=1)(seg_map)
        

        weighted_x = torch.mul(x, seg_map)
        weighted_mean = torch.sum(weighted_x, [2, 3])
        weight_sum = torch.sum(seg_map, [2, 3])

        return torch.div(weighted_mean, weight_sum)


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x, seg_map):
        # semantic-wise color
        seg_map = nn.Softmax(dim=1)(seg_map)

        weighted_r = torch.div(torch.sum(torch.mul(x[:,0:1,:,:], seg_map), [2, 3]), torch.sum(seg_map, [2, 3]))
        weighted_g = torch.div(torch.sum(torch.mul(x[:,1:2,:,:], seg_map), [2, 3]), torch.sum(seg_map, [2, 3]))
        weighted_b = torch.div(torch.sum(torch.mul(x[:,2:3,:,:], seg_map), [2, 3]), torch.sum(seg_map, [2, 3]))
        weighted_color_ratio = torch.cat((torch.div(weighted_r, weighted_g), torch.div(weighted_g, weighted_b), torch.div(weighted_b, weighted_r)), 1)

        return weighted_color_ratio
