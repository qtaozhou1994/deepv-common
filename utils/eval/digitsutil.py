import os
import re
import logging
base_path='/mnt/data3/changwei/workspace/digits/digits/jobs/'
#base_path = '/mnt/data4/liuhao/digits/digits/jobs/'
model_prefix = 'snapshot_iter_'
model_postfix = '.caffemodel'
proto_file = 'deploy.prototxt'


def is_did(input):
    value = re.compile(r'^[0-9]{8}-[0-9]{6}-.{4}$')
    result = value.match(input)
    return result


def get_didmodel(did, iters='last'):
    if (not is_did(did)):
        logging.error('not legal did')

    did_path = os.path.join(base_path, did)
    did_files = os.listdir(did_path)
    prototxt = os.path.join(did_path, proto_file)

    if iters != 'last' and iters is not None:
        if iters.startswith(model_prefix):
            modelname = os.path.join(did_path, iters)
        elif os.path.isabs(iters):
            modelname = iters
        else:
            modelname = os.path.join(did_path, model_prefix + str(iters) + model_postfix)
    else:
        model_iters = [int(f[len(model_prefix):-len(model_postfix)])
                       for f in did_files if f.startswith(model_prefix) and f.endswith(model_postfix)]
        model_iters.sort()
        model_iter = model_iters[-1]
        modelname = os.path.join(did_path, model_prefix + str(model_iter) + model_postfix)

    if not os.path.exists(prototxt):
        logging.warn('prototxt is not exist.{}'.format(prototxt))
    if not os.path.exists(modelname):
        logging.warn('model is not exist.{}'.format(modelname))
    return prototxt,modelname
