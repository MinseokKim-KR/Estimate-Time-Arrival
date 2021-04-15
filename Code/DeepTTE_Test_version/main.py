import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable

import numpy as np
import pandas as pd
import gc
import time

# df = pd.read_csv('./4_summary_msedit.txt')

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)

# evaluation args
parser.add_argument('--weight_file', type=str)
parser.add_argument('--result_file', type=str)

# cnn args
parser.add_argument('--kernel_size', type=int)

# rnn args
parser.add_argument('--pooling_method', type=str)

# multi-task args
parser.add_argument('--alpha', type=float)

# log file name
parser.add_argument('--log_file', type=str)

parser.add_argument('--config_file', type=str)

parser.add_argument('--save_result_file', type=str)
parser.add_argument('--model_num', type=str)
parser.add_argument('--tensorboard', type=str)
parser.add_argument('--elu', type=int)

args = parser.parse_args()
print ("Config file name : ", args.config_file)
# config = json.load(open('./config.json', 'r'))
config = json.load(open(args.config_file, 'r'))


def mkdir_file(path):
    last_string = path.split('/').pop(-1)
    temp_path = path.replace(last_string, '')
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)


if args.task == 'train':
    if not (os.path.isdir(args.weight_file)):
        os.mkdir(os.path.join(args.weight_file))

mkdir_file(args.save_result_file)
f = open(args.save_result_file, 'w')
f_except = open(args.save_result_file + '_except', 'w')


def train(model, elogger, train_set, eval_set, writer, iterate, eval_iterate, START_TIME):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))
    if torch.cuda.is_available():
        model.cuda()
        print('cuda is running')
    learning_rate = 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # my : optimizer = optim.Adam([model.entire_estimate.parameters(), model.local_estimate.parameters(),
    #                         model.spatio_temporal.parameters()], lr=learning_rate)
    writer.add_scalar('learning_rate', learning_rate)
    current_loss = 100000
    for epoch in xrange(args.epochs):
        print 'Training on epoch {}'.format(epoch)
        f.write('Training on epoch {}\n'.format(epoch))
        params = list(model.entire_estimate.parameters()) + list(model.local_estimate.parameters()) + list(model.spatio_temporal.parameters())
        if epoch < 80:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        for input_file in train_set:
            model.train()
            print 'Train on file {}'.format(input_file)
            # print 'model_num : {}'.format(args.model_num)
            f.write('Train on file {}\n'.format(input_file))
            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            running_loss = 0.0
            for idx, (attr, traj) in enumerate(data_iter):
                # print('traj: ',traj)
                # transform the input to pytorch variable
                """
                try:
                    #print('READY to train : ', time.time()-START_TIME)
                    start_time = time.time()
                    attr, traj = utils.to_var(attr), utils.to_var(traj)
                    #if iterate == 0:
                    #	writer.add_graph(model, input_to_model = [attr,traj])
                    print("Train data made")
                            _, loss, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, attr_t, em_list, dist = model.eval_on_batch(attr, traj, config,writer,iterate, flag=0)
                    #print('loss : ', loss.data)
                            if torch.isnan(loss):
                        print('loss is nan')
                                        elogger.log('Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                                        f.write('Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                    elif idx % 400 == 0:
                                        print('idx % 400 == 0')
                                        elogger.log('Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                                        f.write('Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                    # update the model
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.data
                    iterate=iterate+1
                            print '\r model_num : {} Progress {:.2f}%, average loss {}'.format(args.model_num, (idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)),
                            end_time = time.time()
                    print('Time : ',(end_time - start_time)/32)
                except:
                    print("Except DATA!!!!")
                """
                attr, traj = utils.to_var(attr), utils.to_var(traj)
                _, loss, entire_out, local_out, local_length, local_hidden_seqence, local_merged_tensor, attr_t, em_list, dist = model.eval_on_batch(attr, traj, config, writer, iterate, flag=0)

                if torch.isnan(loss):
                    print('loss is nan')
                    elogger.log(
                        'Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(
                            attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence,
                            local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                    f.write(
                        'Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(
                            attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence,
                            local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                elif idx % 400 == 0:
                    print('idx % 400 == 0')
                    elogger.log(
                        'Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(
                            attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence,
                            local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))
                    f.write(
                        'Loss data: \nattr {}, \ntraj {}, \npredict {},\nentire_out {}, \nlocal_out {}, \nlocal_length {}, \nlocal_hidden_seqence {}, \nlocal_merged_tensor {}, \nEpoch {}, \nFile {}, \nLoss {}, \nattr_t {},\nem_list {},\ndist {}'.format(
                            attr, traj, _, entire_out, local_out, local_length, local_hidden_seqence,
                            local_merged_tensor, epoch, input_file, running_loss / (idx + 1.0), attr_t, em_list, dist))

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                iterate = iterate + 1
                print '\r model_num : {} Progress {:.2f}%, average loss {}'.format(args.model_num,
                                                                                   (idx + 1) * 100.0 / len(data_iter),
                                                                                   running_loss / (idx + 1.0)),

            print('')
            # if torch.isnan(loss):
            #	print('Not save weight')
            # else:
            # if loss < current_loss :
            #	torch.save(model.state_dict(), args.weight_file+'/weight')
            #		current_loss = loss
            del data_iter
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))
            f.write('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))
            # gc.collect()

            # evaluate the model after each epoch
            eval_loss = evaluate(model, elogger, eval_set, save_result=False, writer=writer, eval_iterate=eval_iterate,flag=1)
            if torch.isnan(loss):
                print('Not save weight')
            else:
                if eval_loss < current_loss:
                    start_time = time.time()
                    weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
                    elogger.log('Save weight file {}'.format(weight_name))
                    torch.save(model.state_dict(), args.weight_file + '/weight')
                    end_time = time.time()
                    print('Weight time : ', end_time - start_time)
                    current_loss = eval_loss
                    eval_iterate = eval_iterate + 1

                    # save the weight file after each epoch
                    # weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
                    # elogger.log('Save weight file {}'.format(weight_name))

                    # if torch.isnan(loss):
                #	print('Not save weight')
                # else:
                #		if loss < current_loss :
                #			torch.save(model.state_dict(), args.weight_file+'/weight')
                #			current_loss = loss
    f.close()


def write_result(fs, pred_dict, attr, valid):
    # print('pred_dict : ', pred_dict)
    if (pred_dict != 0):
        pred = pred_dict['pred'].data.cpu().numpy()
        label = pred_dict['label'].data.cpu().numpy()

        # FIXME
        # print("attr['calc_time'] : ", attr['calc_time'])
        for i in range(pred_dict['pred'].size()[0]):
            # fs.write('%.6f,%.6f\n' % (label[i][0], pred[i][0]))

            dateID = attr['dateID'].data[i]
            timeID = attr['timeID'].data[i]
            # driverID = attr['driverID'].data[i]
            # MN_predict = df.loc[df["od_id"] ==driverID]['calc_travel_time'].as_matrix()[0]
            ##FIXME
            # MN_predict = attr['calc_time'].data[i]

            fs.write('%.6f,%.6f,%d\n' % (label[i][0], pred[i][0], valid))
            # fs.write('%.6f,%.6f,%.6f,%d\n' % (label[i][0], pred[i][0],MN_predict, valid))
            # fs.write('%.6f,%.6f,%d,%d\n' % (label[i][0], pred[i][0],MN_predict,driverID))
    else:
        # MN_predict = attr['calc_time'].data[0]
        fs.write('%.6f,%.6f,%d\n' % (0, pred_dict, valid))


def evaluate(model, elogger, files, save_result, writer, eval_iterate, flag):
    # model.eval()
    if save_result:
        mkdir_file(args.result_file)
        fs = open('%s' % args.result_file, 'w')
        fs.write('label,predict,valid\n')
    for input_file in files:
        model.eval()
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)
            flag = 0
            try:
                pred_dict, loss = model.eval_on_batch(attr, traj, config, writer, eval_iterate, flag=1)
                valid = 1
                # eval_iterate = eval_iterate+1
                loss = loss.data
            except:
                print("Evaluation Except data")
                pred_dict = 0
                valid = 0
                loss = 0
            if save_result: write_result(fs, pred_dict, attr, valid)
            running_loss += loss
            # writer.add_scalar('loss/total_loss_eval', running_loss, eval_iterate)
            # running_loss += loss.data[0]
        del data_iter

        print 'Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0))
        f.write('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))
	if args.task == 'train':
		writer.add_scalar('loss/eval_train', running_loss / (idx + 1.0), eval_iterate)
	elif args.task == 'test':
		writer.add_scalar('loss/eval_test', running_loss / (idx + 1.0), eval_iterate)
	eval_iterate = eval_iterate + 1



    if save_result:
        fs.close()
        f.close()
        f_except.close()
    return running_loss


def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs


def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)
    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    writer = SummaryWriter()
    iterate = 0
    writer = SummaryWriter(args.tensorboard)
    iterate = 0
    eval_iterate = 0
    START_TIME = time.time()
    if args.task == 'train':
        # print("Config['eval_set'] : ", config['eval_set'])
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'], writer=writer,
              iterate=iterate, eval_iterate=eval_iterate, START_TIME=START_TIME)

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        print("Config['test_set'] : ", config['test_set'])
        evaluate(model, elogger, config['test_set'], save_result=True, writer=writer, eval_iterate=eval_iterate, flag=3)


if __name__ == '__main__':
    run()
