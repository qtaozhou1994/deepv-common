#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# import pdb
import os
import random
import time
import logging
import types
from tqdm import tqdm

DEBUG = True


def make_nparray(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr


def fix_nan(arr, value):
    arr[np.where(np.isnan(arr))] = value
    return arr


def show(path, pt=plt):
    img = cv2.imread(path)
    if img is None:
        logging.warn('%s is not legal image file' % path)
        return None
    img = img[:, :, ::-1]
    pt.imshow(img)
    return img


def _shows(images, showimeddiately=False):
    num = len(images)
    if num == 0:
        print 'List is empty.'
        return
    lie = 8
    hang = num / lie + (num % lie != 0)
    plt.figure(1, figsize=(lie * 2, hang * 10))
    # print hang

    for i in range(hang):
        for j in range(lie):
            index = i * lie + j + 1
            if index > num:
                break
            image_path = images[index - 1]

            pt = plt.subplot(hang, lie, index)
            # image = caffe.io.load_image(image_path)
            img = show(image_path, pt)
            logging.info('[%d] %s {%dx%d}' % ((i * lie + j + 1), image_path, img.shape[0], img.shape[1]))
            pt.set_xticks([])
            pt.set_yticks([])

    plt.tight_layout()
    if showimeddiately:
        plt.show()


def _shows_ownfun(images, ofun):
    num = len(images)
    lie = 8
    hang = num / lie + (num % lie != 0)
    plt.figure(1, figsize=(lie * 2, hang * 10))
    # print hang
    for i in range(hang):
        for j in range(lie):
            index = i * lie + j + 1
            if index > num:
                break
            image_path = images[index - 1]
            logging.info('[%d] %s' % ((i * lie + j + 1), image_path))
            pt = plt.subplot(hang, lie, index)
            # image = caffe.io.load_image(image_path)
            ofun(image_path, pt, index)
            pt.set_xticks([])
            pt.set_yticks([])

    plt.tight_layout()
    plt.show()

def copylist_withoutprefix_2folder(source_list_file,target_dir,root_folder):
    import shutil
    imgs,_=read_imagelist(source_list_file)
    for img in tqdm(imgs):
        oimg=os.path.join(root_folder,img)
        target_parent=os.path.join(target_dir,os.path.dirname(img))
        if not os.path.exists(target_parent):
            os.makedirs(target_parent)
        shutil.copyfile(oimg,os.path.join(target_parent,os.path.basename(img)))



def randsub(oset, num=10):
    clist = np.arange(len(oset), dtype=np.int)
    np.random.shuffle(clist)
    if num > len(oset):
        num = len(oset)
    rs = []
    for i in range(num):
        rs.append(oset[clist[i]])
    return np.array(rs)


def list_images(dirname, depth=1):
    import Queue
    stack = Queue.Queue()
    stack.put((dirname, 0))
    mksfile = []
    dir_count = 0
    dir_contain_pic = 0
    while not stack.empty():
        dirname, dep = stack.get()
        if dep > depth:
            break
        files = os.listdir(dirname)
        files = [os.path.join(dirname, x) for x in files]
        add_action = False
        for f in files:
            if os.path.isfile(f) and (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.JPEG')):
                mksfile.append(f)
                add_action = True
            elif os.path.isdir(f):
                stack.put((f, dep + 1))
                dir_count += 1
        if add_action:
            dir_contain_pic += 1
    logging.info('sum images:' + str(len(mksfile)))
    logging.info('sum dir_count:' + str(dir_count))
    logging.info('sum dir_contain_image:' + str(dir_contain_pic))
    return mksfile

def list_files(dirname, depth=1):
    import Queue
    stack = Queue.Queue()
    stack.put((dirname, 0))
    mksfile = []
    dir_count = 0
    while not stack.empty():
        dirname, dep = stack.get()
        if dep > depth:
            break
        files = os.listdir(dirname)
        files = [os.path.join(dirname, x) for x in files]
        for f in files:
            if os.path.isfile(f):
                mksfile.append(f)
            elif os.path.isdir(f):
                stack.put((f, dep + 1))
                dir_count += 1

    logging.info('file_count:' + str(len(mksfile)))
    logging.info('dir_count:' + str(dir_count))
    return mksfile


def show_folder(dirname, itemperpage=24, page=-1, depth=1):
    timestamp = time.time()
    files = list_images(dirname, depth)
    if len(files) == 0:
        print 'No file'
        return
    files = [x for x in files if x.endswith(
        '.jpg') or x.endswith('.png') or x.endswith('.bmp') or x.endswith('.JPEG')]
    if page == -1:
        showfiles = randsub(files, itemperpage)
    else:
        begin = itemperpage * page
        end = min((page + 1) * itemperpage, len(files))
        if begin > len(files):
            logging.warn('No image file found')
            return
        showfiles = files[begin:end]

    logging.debug('generate list time:' + str(time.time() - timestamp))
    timestamp = time.time()
    showfiles = [x for x in showfiles]
    _shows(showfiles)

    logging.debug('show time:' + str(time.time() - timestamp))


def show_listfile(listfile, page=-1, itemperpage=24):
    with open(listfile) as f:
        files = [x.strip().split()[0] for x in f]
        if page == -1:
            showfiles = randsub(files, itemperpage)
        else:
            begin = itemperpage * page
            end = min((page + 1) * itemperpage, len(files))
            if begin > len(files):
                print 'No img to show!'
                return
            showfiles = files[begin:end]
        _shows(showfiles)
        print 'sum image num:' + str(len(files))


def shows(toshow, itemperpage=24, page=-1, depth=1, savefig=None, printinfo=True, largelist=False):
    # oldlevel = logging.getLevelName()
    if printinfo:
        logging.basicConfig(level=logging.INFO)

    if isinstance(toshow, str):
        if os.path.isdir(toshow):
            show_folder(toshow, page=page,
                        itemperpage=itemperpage, depth=depth)
        elif os.path.isfile(toshow):
            show_listfile(toshow, page=page, itemperpage=itemperpage)
        else:
            logging.error('ERROR,not recognized type')
    elif isinstance(toshow, list) or isinstance(toshow, np.ndarray):
        if not largelist and len(toshow) > 24:
            logging.warn(
                'image list is too large {}. will only show random 24 image.please use largelist option'.format(
                    len(toshow)))
            _shows(randsub(toshow, 24))
        else:
            _shows(toshow)
        _shows(toshow)
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()

        # logging.basicConfig(level=oldlevel)


def generate_timestr():
    import time
    return str(int(time.time()))


class countmap:
    def __init__(self):
        self._mmap = {}

    def add(self, value):
        if self._mmap.has_key(value):
            self._mmap[value] += 1
        else:
            self._mmap[value] = 1

    def __str__(self):
        return self._mmap.__str__()

    def getmap(self):
        return self._mmap

    def fromarray(self, mlist):
        for m in mlist:
            self.add(m)

    def get(self, key):
        return self._mmap[key]

    def keys(self):
        return self._mmap.keys()

    def values(self):
        return self._mmap.values()

    def _get_orderkeys(self, order):
        keys = self._mmap.keys()
        if order == 'str':
            keys = sorted(keys)
        elif order == 'num' or order == 'number' or order == 'digit':
            keys = sorted(keys, key=lambda x: int(x))
        return keys

    def printme(self, order='no'):
        keys = self._get_orderkeys(order)
        for i in keys:
            try:
                print str(i) + ':' + str(self._mmap[i])
            except:
                try:
                    print i + ':' + str(self._mmap[i])
                except:
                    print 'error:'
                    print i
                    print self._mmap[i]

    def printcount(self, order='no'):
        self.printme(order)


class mergemap:
    def __init__(self):
        self._mmap = {}

    def add(self, key, value):
        if self._mmap.has_key(key):
            self._mmap[key].append(value)
        else:
            self._mmap[key] = [value]

    def __str__(self):
        return self._mmap.__str__()

    def getmap(self):
        return self._mmap

    def fromarray(self, keylist, valuelist):
        for i, key in enumerate(keylist):
            self.add(key, valuelist[i])

    def _get_orderkeys(self, order):
        keys = self._mmap.keys()
        if order == 'str':
            keys = sorted(keys)
        elif order == 'num' or order == 'number' or order == 'digit':
            keys = sorted(keys, key=lambda x: int(x))
        return keys

    def printcount(self, order='no'):
        keys = self._get_orderkeys(order)
        for i in keys:
            try:
                print str(i) + ':' + str(len(self._mmap[i]))
            except:
                try:
                    print i + ':' + str(len(self._mmap[i]))
                except:
                    print 'error:'
                    print i
                    print self._mmap[i]

    def printme(self, order='no', forceshow=False):
        keys = self._get_orderkeys(order)
        if not forceshow and self._sum_count > 2000:
            logging.warn('show num is too big {} . please use forceshow option'.format(self._sum_count))
            return
        for i in keys:
            try:
                print str(i) + ':' + self._mmap[i].__str__()
            except:
                try:
                    print i + ':' + self._mmap[i].__str__()
                except:
                    print 'error:'
                    print i
                    print self._mmap[i]


def list_to_map(keys, values, repeated=[]):
    kmap = {}
    for i, key in enumerate(keys):
        if kmap.has_key(key):
            print 'repeated:{}'.format(key)
            repeated.append((key, values[i]))
        else:
            if values is not None:
                kmap[key] = values[i]
            else:
                kmap[key] = []
    return kmap


def be_nparray(array):
    if not isinstance(array, np.ndarray):
        return np.array(array)
    return array


def bes_nparray(arraylis):
    nplist = []
    for array in arraylis:
        if not isinstance(array, np.ndarray):
            nplist.append(np.array(array))
        else:
            nplist.append(array)
    return tuple(nplist)


def read_imagelist(filepath):
    images = []
    labels = []
    with open(filepath) as f:
        for line in tqdm(f):
            if line.startswith('#'):
                continue
            items = line.strip().split()
            images.append(items[0])
            if len(items) > 1:
                labels.append(items[1:])
            else:
                labels.append([])
    # try transfer to int
    if len(labels) > 0:
        onelabel = labels[0]
        legal_int = True
        for l in onelabel:
            try:
                int(l)
            except:
                legal_int = False
        if legal_int:
            labels = format_labels_todigits(labels)
    return (images, labels)


def read_imagelist_tomap(filepath):
    images, labels = read_imagelist(filepath)
    return list_to_map(images, labels)


def write_imagelist(filepath, images, labels=None):
    with open(filepath, 'w') as f:
        for i, image in enumerate(images):
            line = image
            if labels is not None:
                for label in labels[i]:
                    line += ' ' + str(label)
            if i != 0:
                f.write('\n')
            f.write(line)
    return True


def write_map(filepath, mmap):
    return write_imagelist(filepath, mmap.keys(), mmap.values())


def write_imagelist_with_order(filepath, images, labels=None, order=None):
    if order is None:
        write_imagelist(filepath, images, labels=labels)
    else:
        kmap = list_to_map(images, labels)
        with open(filepath, 'w') as f:
            for oi in order:
                line = oi
                if labels is not None:
                    for label in kmap[oi]:
                        line += ' ' + str(label)
                f.write(line + '\n')
    return True


def shuffle_imagelist(keys, values):
    indexs = list(range(len(keys)))
    import random
    random.shuffle(indexs)
    sk = []
    sv = []
    for i in indexs:
        sk.append(keys[i])
        sv.append(values[i])
    return (sk, sv)


def _check_exist(imglist, prefolder=""):
    allclear = True
    notexist = []
    chongfuset = set()
    chongfu = []
    for index, imgpath in enumerate(imglist):
        if imgpath not in chongfuset:
            chongfuset.add(imgpath)
        else:
            chongfu.add(imgpath)
        if not os.path.exists(os.path.join(prefolder, imgpath)):
            allclear = False
            print os.path.join(prefolder, imgpath)
            notexist.append(index)

    if len(chongfu) == 0:
        print 'no repeated image'
    else:
        print '{} images is repeated.'.format(len(chongfu))
    if allclear:
        print 'all exist!'
        return notexist
    else:
        print 'check above'
        return notexist


def check_exist(imglist, prefolder=""):
    if isinstance(imglist, str):
        sk, sv = read_imagelist(imglist)
        return _check_exist(sk, prefolder)
    else:
        return _check_exist(imglist, prefolder)

def sub_imagelist(origin,besubed):
    subed_image=[]
    besubed=set(besubed)
    for o in origin:
        if o not in besubed:
            subed_image.append(o)
    return subed_image

def _check_legal(imglist, prefolder=""):
    ilegal=[]
    import cv2
    not_exist=check_exist(imglist,prefolder)
    ilegal.extend(not_exist)
    imglist=sub_imagelist(imglist,not_exist)

    for img in tqdm(imglist):
        im=None
        im=cv2.imread(os.path.join(prefolder,img))
        
        if im is None:
            ilegal.append(img)
            logging.warn(img+' is not legal')

    return ilegal

def check_legal(imglist,prefolder=""):

    if isinstance(imglist, str):
        imglist, _ = read_imagelist(imglist)

    return _check_legal(imglist,prefolder)



def shuffle_file(inputfile, outputfile):
    pk, pv = read_imagelist(inputfile)
    sk, sv = shuffle_imagelist(pk, pv)
    write_imagelist(outputfile, sk, sv)


def format_labels_todigits(labels):
    if isinstance(labels, list):
        labels = [map(int, x) for x in labels]
    if isinstance(labels, dict):
        for k in labels:
            labels[k] = map(int, labels[k])
    return labels


def replace_prefix_in_imagelist(infile, outfile, oldprefix, newprefix):
    sk, sv = read_imagelist(infile)
    nsk = []
    for k in sk:
        if not k.startswith(oldprefix):
            logging.warn('not prefix {} - {}'.format(k, oldprefix))
        nk = newprefix + k[len(oldprefix):]
        nsk.append(nk)
    write_imagelist(outfile, nsk, sv)


class timecut():
    def __init__(self):
        import time
        self.lasttime = time.time()
        self.commentmap = countmap()
        self._isshow = True

    def cut(self, comment='T'):
        if comment is not None:
            passtime = time.time() - self.lasttime
            self.commentmap.add(comment)
            count = self.commentmap.get(comment)
            if self._isshow:
                print "{}[{}]: {}".format(comment, count, passtime)
        self.lasttime = time.time()

    def setshow(self, isshow):
        self._isshow = isshow


def makedirs(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def _filecompare_bytime(x, y):
    stime_x = os.stat(x).st_ctime
    stime_y = os.stat(y).st_ctime
    if stime_x < stime_y:
        return -1
    if stime_x > stime_y:
        return 1
    return 0


def listfun(module):
    klist = dir(module)
    klist = [x for x in klist if isinstance(eval(x), types.FunctionType) and not x.startswith('_')]
    for x in klist:
        print x,


# def h(fun):
#     import inspect
#     for x in inspect.getargspec(eval(fun))[0]:
#         print x,



def h(function):
    import util
    if function in dir(util):
        help('util.' + function)
    else:
        help(function)
