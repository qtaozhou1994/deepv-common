#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import random
import os
import sys
from os.path import join as opj
import caffeutil
import argparse
import time
import util
import pdb
import cPickle as pickle
import digitsutil

sys.path.append(os.path.dirname(__file__))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Test pedestrian attribute performance.')

    parser.add_argument('-d', '--basepath', default=None, type=str)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-p', '--prototxt', default=None, type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    parser.add_argument('-i', '--image_list', default='', type=str)
    parser.add_argument('-o', '--output_dir', default=None, type=str)

    parser.add_argument('--result_config_file', default=None, type=str)



    parser.add_argument('--result_layer', default=None, type=str, nargs='+')
    parser.add_argument('--mean', type=int, nargs='+', default=-1)
    parser.add_argument('--size', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=24)

    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--scorefile', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def process(args):
    images, labels = util.read_imagelist(args.image_list)

    net = caffeutil.get_net_test(args.prototxt, args.model, gpu=args.gpu)
    scores = caffeutil.batch_process_images(net, images, output_layers=args.result_layer, resize=args.size,
                                            batch_size=args.batch_size, mean=args.mean, scale=args.scale,
                                            return_format='origin')

    # Write the result in bag.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print 'result:',args.output_dir
    scoreoutfile = open(args.scorefile + '.npz', 'wb')
    pickle.dump(images, scoreoutfile, True)
    pickle.dump(scores, scoreoutfile, True)
    pickle.dump(labels, scoreoutfile, True)
    scoreoutfile.close()


def init_param(args):
    basepath = ''
    prototxt = 'deploy.prototxt'
    model = 'deploy.caffemodel'
    if args.basepath is not None and digitsutil.is_did(args.basepath):
        did = args.basepath
        prototxt, model = digitsutil.get_didmodel(did, iters=args.model)
        basepath = os.path.dirname(prototxt)
    else:
        if args.basepath is not None:
            basepath = args.basepath
        print basepath,'----------'
        if args.prototxt is not None:
            prototxt = opj(basepath, args.prototxt)
        if args.model is not None:
            model = opj(basepath, args.model)

    args.basepath = basepath
    args.prototxt = prototxt
    print '----+++',args.prototxt
    args.model = model

    if args.output_dir is None:
        args.output_dir = 'output_%s' % util.generate_timestr()
    args.output_dir = opj(args.basepath, args.output_dir)

    if args.scorefile is None:
        args.scorefile = 'scorefile'
    args.scorefile = opj(args.output_dir, args.scorefile)

    if args.logfile is None:
        args.logfile = 'eval_net.log'
    args.logfile = opj(args.basepath, args.logfile)
    return args


if __name__ == '__main__':
    args = parse_args()
    # print args.image_list
    args = init_param(args)
    # print args.output_layer
    # exit()
    print('Called with args:{}'.format(args))
    process(args)
    print 'all done'
