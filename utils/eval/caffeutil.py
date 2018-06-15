#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy
import cv2
import PIL
import numpy as np
import pdb
import time
import util
from tqdm import tqdm


def caffe_switch(caffepath):
    if caffepath == 0:
        sys.path.insert(0, '/mnt/sde1/liuhao/tool/caffe/python')
    elif caffepath == -1:
        sys.path.insert(0, '/mnt/sde1/liuhao/tool/caffe_cpu/python')
    else:
        sys.path.insert(0, caffepath)
    import caffe


def get_net_test(prototxt, model=None, gpu=0):
    import caffe
    if gpu == -1:
        caffe.set_mode_cpu()
    else:
        print 'using gpu'
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    net = caffe.Net(prototxt, caffe.TEST)
    if not model == None:
        net.copy_from(model)
    return net


def get_solver(prototxt, gpu=0):
    import caffe
    if gpu == -1:
        caffe.set_mode_cpu()
    else:
        print 'using gpu'
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    solver = caffe.SGDSolver(prototxt)
    return solver


IMAGE_CV = 0
IMAGE_CAFFE = 1
IMAGE_PIL = 2

wrong_image_count = 0


def _dealing_images(oimage, mean=None, scale=None, imagetype=IMAGE_CV, resize=None, offset=None, crop_size=None):
    image = oimage
    if type(image) == str:
        image = cv2.imread(image)
        if image is None:
            # wrong_image_count += 1
            print '!:' + oimage
            return None
            # image=image[:,:,::-1]
    else:
        # caffe.io.load 0-1.0 RGB
        # Image 0-255 RGB
        # cv2 0-255 BGR
        if imagetype == IMAGE_CV:
            pass
        elif imagetype == IMAGE_CAFFE:
            image *= 255
            image = image[:, :, ::-1]
        elif imagetype == IMAGE_PIL:
            image = image[:, :, ::-1]

    image = np.asarray(image, dtype=np.float32)
    if mean == -1:
        image -= np.array((104.00698793, 116.66876762,
                           122.67891434), dtype=np.float32)
    elif mean is not None:
        image -= np.array(mean)
    # origin is (height,width,3),so let the resize param is also (height,width)
    # image = cv2.resize(image, (100, 100))
    if resize is not None:

        image = cv2.resize(image, (resize[1], resize[0]))
    if offset is not None and crop_size is not None:
        image = image[offset[0]:offset[0] + crop_size[0], offset[1]:offset[1] + crop_size[1], :]
    if scale is not None:
        image *= scale
    if len(image.shape) == 3:
        image = image.transpose((2, 0, 1))
    elif len(image.shape) == 2:
        image = image[np.newaxis, ...]
    return image


def _forward_net_get_output(net, output_layers=None):
    output = net.forward()
    foutput = {}

    if output_layers == None:
        output_layers = output.keys()

    for olayer in output_layers:
        if olayer in output:
            foutput[olayer] = output[olayer].copy()
        else:
            foutput[olayer] = net.blobs[olayer].data.copy()
    return foutput


def load_images(images, mean=None, scale=None, imagetype=IMAGE_CV, resize=None, offset=None, crop_size=None):
    if type(images) != list:
        images = [images]
    initial = False
    for index, image in enumerate(images):
        image = _dealing_images(image, mean=mean, scale=scale,
                                imagetype=imagetype, resize=resize, offset=offset, crop_size=crop_size)
        if not image is None:
            if not initial:
                blob = np.zeros((len(images), image.shape[0], image.shape[
                    1], image.shape[2]), dtype=np.float32)
                initial = True
            blob[index, ...] = image
    if initial:
        return blob
    return None


def process_images(net, images, input_layer='data', output_layers=None, mean=None, scale=None,
                   imagetype=IMAGE_CV, resize=None, return_format='seperate'):
    """Process image and return the result seperately correspond the input images.
    
    Args:
        net: the caffe net object.
        images: imagelist. image path list or numpy object list
        input_layer: must be str
        output_layers: can be list<str> or str
        mean: mean(BGR). spec: -1:(104.00698793, 116.66876762,122.67891434); none:0
        scale: scale.   image *= scale
        imagetype: IMAGE_CV(0-255 BGR),IMAGE_CAFFE(0-1.0 RGB) ,IMAGE_PIL(0-255 RGB)
        resize: resize(height,width).

    Returns:
        the result seperately correspond the input images. list<list(nparray<float>)

    """
    image_num = len(images)
    if output_layers is not None and isinstance(output_layers, str): output_layers = [output_layers]

    if return_format not in ['origin', 'seperate', 'origin_withnamemap']:   raise ValueError('return_format is unkown')
    blob = load_images(images, mean=mean, scale=scale,
                       imagetype=imagetype, resize=resize)
    output = _process_blob(
        net, blob=blob, input_layer=input_layer, output_layers=output_layers)

    if output_layers is None: output_layers = output.keys()

    if return_format == 'seperate':
        b_score = [[output[ok][i, ...] for ok in output_layers] for i in xrange(image_num)]
        return b_score
    elif return_format == 'origin':
        b_score = [output[ok] for ok in output_layers]
        return b_score
    elif return_format == 'origin_withnamemap':
        return output
    else:
        raise ValueError('Here you are. The return format unkonwn error')


def batch_generator(sums, batch_size):
    """The batch generator util.
    
    Args:
        sums: sum_count
        batch_size: batch_size

    Returns:
        begin_index, num

    """
    iters = sums / batch_size + (sums % batch_size != 0)
    for index in tqdm(range(iters)):
        b_begin_index = index * batch_size
        b_num = min(batch_size, sums - b_begin_index)
        yield b_begin_index, b_num


def batch_process_images(net, images, batch_callback=None, batch_size=16, input_layer='data', output_layers=None,
                         mean=None, scale=None, imagetype=IMAGE_CV, resize=None, return_format='seperate'):
    """Batch-process the images.
    
    Args:
        net: the caffe net object.
        images: imagelist. image path list or numpy object list
        batch_callback: func(b_begin_index, b_num, b_score). this callback will be called when one batch is done. 
        batch_size: batch size
        input_layer: must be str
        output_layers: can be list or str
        mean: mean(BGR).    spec: -1:(104.00698793, 116.66876762,122.67891434); none:0
        scale: scale.   image *= scale
        imagetype: IMAGE_CV(0-255 BGR),IMAGE_CAFFE(0-1.0 RGB) ,IMAGE_PIL(0-255 RGB)
        resize: resize(height,width).
        return_format: ['origin','seperate']
        
    Returns:
        A score list list<list(nparray<float>) if callback is None.None if callback is not None.

    """
    if return_format not in ['origin', 'seperate', 'origin_withnamemap']:   raise ValueError('return_format is unkown')
    img_nums = len(images)
    scores = []
    initial_score=False
    for b_begin_index, b_num in batch_generator(img_nums, batch_size):
        b_images = images[b_begin_index:b_begin_index + b_num]
        b_score = process_images(net, b_images, input_layer=input_layer, output_layers=output_layers,
                                 mean=mean, scale=scale,
                                 imagetype=imagetype, resize=resize, return_format=return_format)
        if batch_callback is None:
            if return_format == 'origin':
                if len(b_score) != 0 and not initial_score:
                    scores=[]
                    for bb in b_score:
                        kshape = list(bb.shape)
                        # first axis is image number.
                        kshape[0] = img_nums
                        scores.append(np.zeros(kshape, dtype=np.float))
                    initial_score=True
                elif initial_score and len(scores) != len(b_score):
                    raise ValueError('the dim of return blob is changed.')
                for i_bb, bb in enumerate(b_score):
                    scores[i_bb][b_begin_index:b_begin_index + b_num, ...] = bb[...]
            elif return_format == 'seperate':
                scores.extend(b_score)
                initial_score=True
            elif return_format =='origin_withnamemap':
                if b_score is not None and not initial_score:
                    scores={}
                    for bkey in b_score:
                        kshape = list(b_score[bkey].shape)
                        # first axis is image number.
                        kshape[0] = img_nums
                        scores[bkey]=np.zeros(kshape, dtype=np.float)
                    initial_score=True
                for bkey in b_score:
                    scores[bkey][b_begin_index:b_begin_index + b_num, ...] = b_score[bkey][...]
            else:
                raise ValueError('The return format unkonwn error')
        else:
            batch_callback(b_begin_index, b_num, b_score)

    if batch_callback is None:
        return scores


def transferblob_origin2seperate(score):
    image_num=score[0].shape[0]
    return [[bbscore[i, ...] for bbscore in score] for i in xrange(image_num)]

def transferblob_seperate2origin(score):
    image_num = len(score)
    attr_num=len(score[0])
    result_score=[]
    for i in xrange(attr_num):
        kshape=list(score[0][i].shape[i])
        kshape.insert(0,image_num)
        sblob=np.zeros(kshape,dtype=np.float)
        for i_img in xrange(image_num):
            sblob[i_img,...]=score[i_img][i]
        result_score.append(sblob)
    return result_score

def _process_blob(net, blob, input_layer='data', output_layers=None):
    if net.blobs[input_layer].data.shape != blob.shape:
        net.blobs[input_layer].reshape(*blob.shape)
    net.blobs[input_layer].data[...] = blob[...]
    output = _forward_net_get_output(net, output_layers=output_layers)
    return output


def _infer(net, input_map, output_key):
    for input_key in input_map:
        net.blobs[input_key][...] = input_map[input_key]
    output = _forward_net_get_output(net, output_layers=output_key)
    return output


def mean_fromfile(mfilepath):
    import caffe
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mfilepath, 'rb').read()
    blob.ParseFromString(data)
    array = np.array(caffe.io.blobproto_to_array(blob))
    return array


def mean3_fromfile(mfilepath):
    array = mean_fromfile(mfilepath)
    mean = np.array([np.mean(array[0, 0, :, :]), np.mean(
        array[0, 1, :, :]), np.mean(array[0, 2, :, :])])
    return mean

def load_prototxt_def(protofile):
    import caffe.proto.caffe_pb2 as caffe_pb2
    from google.protobuf.text_format import Merge
    net = caffe_pb2.NetParameter()
    with open(protofile, 'r') as f:
        Merge(f.read(), net)
    return net



def _test_net(net):
    images = [
        '/mnt/data2/fuhaolin/data/PersonAttr/person_upper_crop/person_struct_crop-2-27263_20160308vimg_datav44_6600.jpg',
        '/mnt/data2/fuhaolin/data/PersonAttr/person_upper_crop/person_struct_crop-1-19519_resizedvimg_datav048932.jpg',
        '/mnt/data2/fuhaolin/data/PersonAttr/PersonAttribCrop05/13-4f9e65252a087ddd5ec9757f4e0ce0fc.jpg']
    blob = load_images(images, resize=[224, 224], mean=-1)
    output = _process_blob(net, blob)
    print output
    predict = [np.argmax(x) for x in output['softmax']]
    print predict


if __name__ == "__main__":
    import os

    # basepath='/mnt/data4/liuhao/tool/digits/digits/jobs/20170322-184003-6e63'
    basepath = '/mnt/data4/liuhao/tool/digits/digits/jobs/20170320-223301-f006'
    prototxt = os.path.join(basepath, 'deploy.prototxt')
    modelfile = os.path.join(basepath, 'snapshot_iter_157560.caffemodel')
    net = get_net_test(prototxt, modelfile)
    from threading import Thread

    z = Thread(target=_test_net, args=(net,))
    z.daemon = True
    z.start()
    z.join()
