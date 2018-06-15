#!/usr/bin/env python
# coding=utf-8
import sys
caffe_root = '/Users/shenbingran/code/caffe/'
sys.path.insert(0, caffe_root + 'python')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import numpy as np
import random
import sys
import cPickle as pickle
import caffeutil
import os
import argparse
import time
from util import bes_nparray
import pdb
import util
e = 0.000001
timecut = util.timecut()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test score performance.')
    parser.add_argument('score_files',
                        nargs='+', type=str)
    parser.add_argument('--attrnamefile', type=str, default=None)
    parser.add_argument('--signature', type=str, default='default')
    parser.add_argument('--has_name_followed', action='store_true')
    args = parser.parse_args()
    return args


def eval_score_softmax_multh(images, scores, labels, attr_class_index, threshold=None, fpdir=None):
    if threshold == None:
        threshold = [0.0, 0.000001, 0.00001, 0.0001,
                     0.001, 0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97,
                     0.98, 0.985, 0.99, 0.994, 0.996, 0.999, 0.9999, 0.99999, 0.999999, 1.0]
        # threshold=[0.5]

    # # images, scores, labels = bes_nparray((images, scores, labels))
    # # attr_class_nums = scores.shape[1]
    # import cv2
    # for image in images:
    #     img=cv2.imread(image)
    #     if img is None:
    #         print image
    #         continue
    #     ss=min(img.shape[0],img.shape[1])
    #
    noignores = labels != -1

    labels = (labels == attr_class_index)

    scores = scores[:, attr_class_index]

    result = []
    for th in threshold:

        predict = scores > th

        trueset = predict == labels
        falseset = 1 - trueset

        TPset = trueset & predict & noignores
        FPset = predict & falseset & noignores
        TNset = trueset & (1 - predict) & noignores
        FNset = falseset & (1 - predict) & noignores

        TP = np.sum(TPset, axis=0).astype(np.float32)
        FP = np.sum(FPset, axis=0).astype(np.float32)
        TN = np.sum(TNset, axis=0).astype(np.float32)
        FN = np.sum(FNset, axis=0).astype(np.float32)

        if fpdir is not None:
            with open(fpdir + '/fpth_' + str(attr_class_index) + '_' + str(th) + '.list', 'w') as f:
                for index, fpitem in enumerate(FPset):
                    if fpitem:
                        f.write(images[index] + '\n')

            with open(fpdir + '/tpth_' + str(attr_class_index) + '_' + str(th) + '.list', 'w') as f:
                for index, fpitem in enumerate(TPset):
                    if fpitem:
                        f.write(images[index] + '\n')

            with open(fpdir + '/fnth_' + str(attr_class_index) + '_' + str(th) + '.list', 'w') as f:
                for index, fpitem in enumerate(FNset):
                    if fpitem:
                        f.write(images[index] + '\n')

        TP_rate = 0 if TP == 0 else TP / (TP + FN)
        TN_rate = 0 if TN == 0 else TN / (TN + FP)

        Recall = TP_rate
        Precision = 1 if TP == 0 else TP / (TP + FP)

        Accuracy = 0 if TP == 0 and TN == 0 else (TP + TN) / (TP + FP + TN + FN)

        brs = {'threshold': th, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'TP_rate': TP_rate, 'TN_rate': TN_rate,
               'Recall': Recall, 'Precision': Precision, 'Accuracy': Accuracy}
        result.append(brs)
        # print str(Precision)+'\t'+str(Recall)
    return result


def get_recall_given_precision(result, precision):
    max_precision = 0
    max_recall = 0
    right_threshold = 0.0
    min_precision = 1
    min_recall = 0
    left_threshold = 0.0
    real_max_recall = -1
    for i_rs in xrange(1, len(result)):
        if (result[i_rs]['Precision'] - precision) * (result[i_rs - 1]['Precision'] - precision) <= 0:
            if result[i_rs]['Recall'] > real_max_recall:
                real_max_recall = result[i_rs]['Recall']
                max_precision = result[i_rs]['Precision']
                max_recall = result[i_rs]['Recall']
                right_threshold = result[i_rs]['threshold']
                min_precision = result[i_rs - 1]['Precision']
                min_recall = result[i_rs - 1]['Recall']
                left_threshold = result[i_rs - 1]['threshold']
    if real_max_recall == -1:
        if result[0]['Precision'] > precision:
            return 1.0, 0.5
        else:
            return 0.0, 1.0

    realrecall = (max_recall * abs(precision - min_precision) + min_recall * abs(max_precision - precision)) / abs(
        max_precision - min_precision)
    realth = (right_threshold * abs(precision - min_precision) + left_threshold * abs(max_precision - precision)) / abs(
        max_precision - min_precision)
    return realrecall, realth


def softmax_curve(pt, result, labelname='default'):
    pt.axis([0, 1, 0, 1])
    pt.ylabel('Precision  TP/(TP+FP)')
    pt.xlabel('Recall  TP/(TP+FN)')
    pt.xticks(np.arange(0, 1, 0.1))
    pt.yticks(np.arange(0, 1, 0.1))
    precisions = []
    recalls = []

    default_th_pr = []
    for rs in result:
        precisions.append(rs['Precision'])
        recalls.append(rs['Recall'])
        if rs['threshold'] == 0.5:
            default_th_pr = (rs['Precision'], rs['Recall'])

    pt.plot(recalls, precisions, label=labelname)
    pt.plot(recalls, precisions, 'o')
    pt.plot((default_th_pr[1]), (default_th_pr[0]), '*')
    pt.legend()


def readnpzfiles(score_files,linenames):
    ctartgets = []
    for index, scorefile in enumerate(score_files):
        with open(scorefile, 'rb') as mfile:
            timecut.cut(None)
            mimages = pickle.load(mfile)
            mscores = pickle.load(mfile)
            mlabels = pickle.load(mfile)
            timecut.cut("read_from_file_done")
            mscores = caffeutil.transferblob_origin2seperate(mscores)

            images, scores, labels = bes_nparray((mimages, mscores, mlabels))
            if index >= len(linenames):
                linename = ''
            else:
                linename = linenames[index]
            kfile = {'images': images, 'scores': scores, 'labels': labels, 'linename': linename}
            ctartgets.append(kfile)
            timecut.cut("analyze npzfiles done")
    return ctartgets


def read_label_name(filename):
    labelfile = filename
    labels = []
    sm_num = []
    if filename is None:
        for i in range(200):
            labels.append('label' + str(i))
            sm_num.append(2)
    else:
        labels, o_sm_num = util.read_imagelist(filename)
        o_sm_num = util.format_labels_todigits(o_sm_num)
        for sn in o_sm_num:
            if len(sn) == 0:
                sm_num.append(2)
            else:
                sm_num.append(sn[0])
    return labels, sm_num


def eval_seperate(ctargets, nargs):
    dir_score = 'score_' + nargs.signature + '_' + util.generate_timestr()
    os.mkdir(dir_score)

    ppout = open(os.path.join(os.path.join(dir_score, 'args')), 'w')
    ppout.write(args.__str__())
    attr_label_names, _ = read_label_name(nargs.attrnamefile)
    # pdb.set_trace()
    for i_target, target in enumerate(ctargets):
        # pdb.set_trace()
        images, scores, labels, linename = (target['images'], target['scores'], target['labels'], target['linename'])
        if linename == '':
            linename = str(i_target)
        # pdb.set_trace()
        dir_target = os.path.join(dir_score, linename)
        os.mkdir(dir_target)

        attr_nums = scores.shape[1]
        image_num = len(images)

        cname = []
        cset_0_8 = []
        cset_0_9 = []
        csetth_0_8 = []
        # figcount=1
        for attr_index in range(attr_nums):
            b_images = images

            b_scores = np.array([scores[x, attr_index] for x in range(image_num)])
            b_labels = labels[:, attr_index]
            attr_class_nums = b_scores.shape[1]

            dir_attr = os.path.join(dir_target, 'badcase_' + attr_label_names[attr_index])
            os.mkdir(dir_attr)

            uid = linename + '_' + str(attr_index) + ''
            # pdb.set_trace()
            for attr_class_index in range(attr_class_nums):
                # pdb.set_trace()
                if attr_class_nums == 2 and attr_class_index == 0:
                    continue
                pout = open(os.path.join(dir_target, 'performance_' + attr_label_names[attr_index] + '_' + str(
                    attr_class_index) + '.txt'), 'w')
                # if attr_label_names[attr_index] != 'age':
                # pdb.set_trace()
                result = eval_score_softmax_multh(b_images, b_scores, b_labels, attr_class_index, fpdir=dir_attr)

                pqrecall0_9, pqth0_9 = get_recall_given_precision(result, 0.9)
                pqrecall0_8, pqth0_8 = get_recall_given_precision(result, 0.8)
                pout.write('attr_class_index' + str(attr_class_index) + '\n')
                pout.write('p_0_9:' + str(pqrecall0_9) + '\n')
                pout.write('p_0_8:' + str(pqrecall0_8) + '\n')
                pout.write('result:\n' + result.__str__() + '\n')
                cname.append(attr_label_names[attr_index])
                cset_0_8.append(pqrecall0_8)
                cset_0_9.append(pqrecall0_9)
                csetth_0_8.append(pqth0_8)

                plt.figure(figsize=(8, 8))
                plt.title(attr_label_names[attr_index])
                plt.grid(True)
                ucid = linename + '_' + str(attr_index) + '_' + str(attr_class_index) + ''
                softmax_curve(plt, result, labelname=ucid)
                nsave_figname = str(attr_index) + "_" + attr_label_names[attr_index] + '_' + str(
                    attr_class_index) + '.png'
                plt.savefig(os.path.join(dir_target, nsave_figname))
                plt.close()

                pout.flush()
                pout.close()
        for xxs in range(len(cset_0_8)):
            line = cname[xxs] + ('\t%.2f\t%.2f' % (cset_0_8[xxs] * 100, csetth_0_8[xxs]))
            print line
            ppout.write(line + '\n')
        for xxs in range(len(cset_0_9)):
            line = cname[xxs] + ('\t%.2f' % (cset_0_9[xxs] * 100))
            print line
            ppout.write(line + '\n')
    ppout.close()


if __name__ == '__main__':
    args = parse_args()
    print args

    if args.has_name_followed:
        score_files = [x for index, x in enumerate(args.score_files) if index % 2 == 0]
        linenames = [x for index, x in enumerate(args.score_files) if index % 2 == 1]
    else:
        score_files = args.score_files
        linenames = map(str,range(len(score_files)))
    score_files = [os.path.join(x, 'scorefile.npz') if os.path.isdir(x) else x for x in score_files]

    ctargets = readnpzfiles(score_files,linenames)
    eval_seperate(ctargets, args)
