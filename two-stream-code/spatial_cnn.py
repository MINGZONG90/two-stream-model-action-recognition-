import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=10, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    #Prepare DataLoader
    print '*************************Prepare DataLoader********************************'
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                       # path='/media/ming/DATADRIVE1/UCF101new/UCF-101/',
                        path='/media/ming/DATADRIVE1/UCF101 Dataset/jpegs_256/',
                        ucf_list ='/media/ming/DATADRIVE1/two stream/two-stream-code/UCF_list/',
                        ucf_split ='01', 
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model
    print '*************************Model********************************'
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video
    )
    #Training
    print '*************************Training********************************'
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))# ==> loading checkpoint './record/spatial/checkpoint.pth.tar'
                checkpoint = torch.load(self.resume)                    # ==> loading checkpoint '../spacemodel/model_best.pth.tar'
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            #    print checkpoint
            #    print len(checkpoint)                                   # 4                      4
            #    print self.start_epoch, self.best_prec1                 # 34  82.1305847168      31 82.1305847168
            #    model_dict = self.model.state_dict()                    # print key in model_parameters
            #    for k in model_dict:
            #        print(k)
            #    print self.train_loader[100000000000000000000000]

                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})" # ==> loaded checkpoint './record/spatial/checkpoint.pth.tar' (epoch 34) (best_prec1 82.1305847168)
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        for self.epoch in range(self.start_epoch, self.nb_epochs):  #train one epoch and validate/test
            self.train_1epoch()      #train
            prec1, val_loss = self.validate_1epoch()   #validate/test
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs)) # ==> Epoch:[0/35][training stage]

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):
        #    print data_dict    #{'img2':[torch.FloatTensor of size 10x3x224x224],'img1':10x3x224x224,'img0':10x3x224x224]}
        #    print label      #[torch.LongTensor of size 10]
        #    print len(data_dict['img1'])       #10
        #    print len(data_dict)               #3

            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()#Variable containing:70 73 75 48 51 7 12 7 55 75[torch.cuda.LongTensor of size 10 (GPU 0)]

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda()  #10*101
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)        #Variable containing:[torch.cuda.FloatTensor of size 10x101 (GPU 0)]

            loss = self.criterion(output, target_var) #Variable containing:0.1648[torch.cuda.FloatTensor of size 1 (GPU 0)]

        #    print loss                 #Variable containing:0.1648[torch.cuda.FloatTensor of size 1 (GPU 0)]
        #    print loss.data
        #    print key         #img2
        #    print data     # [torch.FloatTensor of size 10x3x224x224] can print out, block scope is invalid
        #    print data.size(0)           #10

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            print '***********prec***************', i

        #    print prec1                 # 100
        #    print prec5                 # 100
        #    print loss.data[0]          # 0.0796460658312
        #    print data.size(0)          # 10
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
         #   print keys                  #('Fencing_g06_c03', 'MoppingFloor_g04_c01',...,'Archery_g02_c02')  number=10
         #   print data                  # [torch.FloatTensor of size 10x3x224x224]
         #   print label                 #27 54 27 58 80 15 2 63 84 2 [torch.LongTensor of size 10]

            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)   # [torch.cuda.FloatTensor of size 10x101 (GPU 0)]
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]              # 10
        #    print preds        #   [[-1.16014826 ..., -0.74533045] ... [-0.26081595 ..., 0.15930791]]  shape: 10*101
        #    print preds.shape               #(10, 101)
            for j in range(nb_data):
                videoName = keys[j].split('/',1)[0]
               # print keys[j]           #Fencing_g06_c03
               # print videoName         #Fencing_g06_c03
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                   # print preds[j,:]                #[-1.16014826 -0.5870198 ..., -0.74533045] number=101
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
      #  print len(self.dic_video_level_preds)       #3783
      #  print video_level_preds.shape               #(3783, 101)
      #  print video_level_labels.shape              #(3783,)

        for name in sorted(self.dic_video_level_preds.keys()):
        
            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())
    #    print top1
    #    print top5
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()







if __name__=='__main__':
    main()