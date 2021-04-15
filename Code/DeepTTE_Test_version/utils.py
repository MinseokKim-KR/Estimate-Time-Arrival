import torch
from torch.autograd import Variable

import json

from math import radians, cos, sin, asin, sqrt

config = json.load(open('./config/config_3_month.json', 'r'))
#config = json.load(open('./config.json', 'r'))

def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def normalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return (x - mean) / std

def unnormalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return x * std + mean

def pad_sequence(sequences, lengths):
    padded = torch.zeros(len(sequences), lengths[0]).float()
    for i, seq in enumerate(sequences):
        seq = torch.Tensor(seq)
        padded[i, :lengths[i]] = seq[:]
    return padded

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            #print('var:',var)
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = full_seq.size()[1]

    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor(seq_len)
    else:
        indices = torch.LongTensor(seq_len)

    torch.arange(0, seq_len, out = indices)

    indices = Variable(indices, requires_grad = False)

    first_seq = torch.index_select(full_seq, dim = 1, index = indices[kernel_size - 1:])
    # second_seq = torch.index_select(full_seq, dim = 1, index = indices[:-kernel_size])
    second_seq = torch.index_select(full_seq, dim = 1, index = indices[:-kernel_size + 1])

    #print("first_seq : ", first_seq)
    #print("first_seq size(): ", first_seq.size())
    #print("second_seq : ", second_seq)
    #print("second_seq size(): ", second_seq.size())
    # print("full_seq : ", full_seq)
    # print("full_seq size(): ", full_seq.size())
    # f = open('result_first_seq_second_seq', 'w')
    # f.write('First seq : {}\n'.format(first_seq[0]))
    # f.write('Second seq : {}\n'.format(second_seq[0]))
    # f.write('full_seq seq : {}\n'.format(full_seq[0]))
    # f.close()
    if first_seq.size() == second_seq.size():
        local_seq = first_seq
    elif second_seq.size()[1] == 0:
	local_seq = first_seq
    else :
        local_seq = first_seq - second_seq

    # local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq

