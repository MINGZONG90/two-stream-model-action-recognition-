import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# other util
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

#    print output             # [torch.cuda.FloatTensor of size 10x101 (GPU 0)]
#    print target             # [torch.cuda.LongTensor of size 10 (GPU 0)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)        #pred is index
                                                                          # training
#    print maxk         # 5                                               # 5
#    print batch_size   # 3783                                            # 10
#    print topk         # (1, 5)                                          # (1, 5)
#    print pred         # [torch.LongTensor of size 3783x5]                # [torch.cuda.LongTensor of size 10x5 (GPU 0)]
#    print output         #[torch.FloatTensor of size 3783x101]
#    print target         #[torch.LongTensor of size 3783]


    pred = pred.t()  # [torch.LongTensor of size 5x3783]  transpose       # [torch.cuda.LongTensor of size 5x10 (GPU 0)]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # the size -1 is inferred from other dimensions

#    print '$$$$$$$$$$$$$$$$$$$$$$$$$'
#    print pred                                                                       # training
#    print target.view(1, -1)       #[torch.LongTensor of size 1x3783]                # [torch.cuda.LongTensor of size 1x10 (GPU 0)]
#    print target.view(1, -1).expand_as(pred)  #[torch.LongTensor of size 5x3783]     # [torch.cuda.LongTensor of size 5x10 (GPU 0)]
#    print correct                           #[torch.ByteTensor of size 5x3783]       # [torch.cuda.ByteTensor of size 5x10 (GPU 0)]


    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
#        print '*******************correct_k******************'
#        print correct[:k]   # 1     1     1     1     1     1     1     1     1     1
#        print correct_k     # 100
#        print res           # [100]
#        print 'finish:',k
#    print output[100000000000000000000000]
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))      
        print result  # Time [1.102] Data [0.012]
                      # Loss [1.47386] Prec@1 [64.7793] Prec@5 [83.2128]
                      # LR 0.0005
        print result[100000000000000000000000000000000]

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   


