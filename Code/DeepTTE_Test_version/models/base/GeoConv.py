import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, kernel_size, num_filter, elu):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
	self.elu = elu
        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)


    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)

        states = self.state_em(traj['states'].long())

        locs = torch.cat((lngs, lats, states), dim=2)

        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)

        # conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # permute => change the shape  permute(0,2,1) -> shape (original[0] ,original[2], original[1])
        if self.elu == 0:
            if self.kernel_size == -1 : # remove Geo_conv
                conv_locs = locs[:, :, 1:]
                conv_locs = conv_locs.permute(0, 2, 1)
            else: #kernel size = self.kernel_size
                conv_locs = locs[:,:,1:]
                conv_locs = self.conv(conv_locs).permute(0, 2, 1)
        else: # Add elu function
            if self.kernel_size == -1 : # remove Geo_conv
                conv_locs = locs[:, :, 1:]
                conv_locs = F.elu(conv_locs).permute(0, 2, 1)
                print('kernel_size is -1')
            else: #kernel size = self.kernel_size
                conv_locs = locs[:,:,1:]
                conv_locs = F.elu(self.conv(conv_locs)).permute(0, 2, 1)

            # conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)
	        # print('else in kernel_size')

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'],
                                        config['dist_gap_std'])
        local_dist1 = local_dist
        local_dist = torch.unsqueeze(local_dist, dim=2)

        # #FIXME
        # print('START')
        print("size locs : ", locs.size())
        print("size self.conv(locs) : ", self.conv(locs).size())
        print("size conv_locs : ", conv_locs.size())
        print("size local_dist : ", local_dist.size())
        # print("NUM_filter : ",self.num_filter)
        # f = open('conv_locs_result', 'w')
        # f.write('locs : {}\n'.format(locs))
        # f.write('locs size : {}\n'.format(locs.size()))
        # f.write('self.conv(locs) : {}\n'.format(self.conv(locs)))
        # f.write('self.conv(locs) size : {}\n'.format(self.conv(locs).size()))
        # f.write('conv_locs : {}\n'.format(conv_locs))
        # f.write('conv_locs size : {}\n'.format(conv_locs.size()))
        # f.write('local_dist : {}\n'.format(local_dist))
        # f.write('local_dist size : {}\n'.format(local_dist.size()))
        # f.write('local_dist : {}\n'.format(local_dist))
        # f.write('local_dist size : {}\n'.format(local_dist.size()))
        # f.write('NUM_filter : {}\n', format(self.num_filter))
        # f.close()
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)

        return conv_locs
