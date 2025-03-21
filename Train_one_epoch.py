# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


def print_summary_val(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, F1, average_F1,pre, average_pre,se, average_se, sp, average_sp,acc, average_acc,recall,average_recall ,mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    string += 'F1:{:.4f} '.format(F1)
    string += '(Avg {:.4f}) '.format(average_F1)
    string += 'pre:{:.4f} '.format(pre)
    string += '(Avg {:.4f}) '.format(average_pre)
    string += 'se:{:.4f} '.format(se)
    string += '(Avg {:.4f}) '.format(average_se)
    string += 'sp:{:.4f} '.format(sp)
    string += '(Avg {:.4f}) '.format(average_sp)
    string += 'Acc:{:.3f} '.format(acc)
    string += '(Avg {:.4f}) '.format(average_acc)
    string += 'recall:{:.3f} '.format(recall)
    string += '(Avg {:.4f}) '.format(average_recall)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    F1_sum =0
    pre_sum =0
    se_sum =0
    sp_sum =0
    acc_sum =0
    rec_sum = 0

    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds = model(images)
        out_loss = criterion(preds, masks.float())  # Loss


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()



        # train_iou = 0
        train_iou = iou_on_batch(masks,preds)
        train_dice = criterion._show_dice(preds, masks.float())
        # if not model.training:
        #     F1_score, precision, sensitivity, specificity, accuracy ,recall=   F1sesp_on_batch(masks, preds)

        batch_time = time.time() - end
        # train_acc = acc_on_batch(masks,preds)
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        # if not model.training:
        #     F1_sum += len(images) * F1_score
        #     pre_sum += len(images) * precision
        #     se_sum += len(images) * sensitivity
        #     sp_sum += len(images) * specificity
        #     acc_sum += len(images) * accuracy
        #     rec_sum += len(images) * recall


        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
            # if not model.training:
            #     F1_ave = F1_sum /  (config.batch_size*(i-1) + len(images))
            #     pre_ave = pre_sum /  (config.batch_size*(i-1) + len(images))
            #     se_ave = se_sum /  (config.batch_size*(i-1) + len(images))
            #     sp_ave = sp_sum /  (config.batch_size*(i-1) + len(images))
            #     acc_ave = acc_sum /  (config.batch_size*(i-1) + len(images))
            #     rec_ave = rec_sum/ (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)
            # if not model.training:
            #     F1_ave = F1_sum/ (i * config.batch_size)
            #     pre_ave = pre_sum/ (i * config.batch_size)
            #     se_ave = se_sum / (i * config.batch_size)
            #     sp_ave = sp_sum / (i * config.batch_size)
            #     acc_ave = acc_sum / (i * config.batch_size)
            #     rec_ave = rec_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            if model.training:
                print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)
            if not model.training:
                print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                              average_loss, average_time, train_iou, train_iou_average,
                              train_dice, train_dice_avg, 0, 0, logging_mode,
                              lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)
                # print_summary_val(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                #               average_loss, average_time, train_iou, train_iou_average,
                #               train_dice, train_dice_avg,  F1_score, F1_ave,precision,pre_ave, sensitivity,se_ave, specificity,sp_ave, accuracy,acc_ave,recall,rec_ave, logging_mode,
                #               lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg

