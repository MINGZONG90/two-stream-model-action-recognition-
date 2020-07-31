import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *
 
class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):  #spatial does not have
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel                 #spatial does not have
        self.img_rows=224                            #spatial does not have
        self.img_cols=224                            #spatial does not have

    def stackopf(self):                              #spatial does not have
        name = 'v_'+self.video   #v_PlayingViolin_g22_c04
        u = self.root_dir+ 'u/' + name   #/media/ming/DATADRIVE1/UCF101 Dataset/tvl1_flow/u/v_PlayingViolin_g22_c04
        v = self.root_dir+ 'v/'+ name    #/media/ming/DATADRIVE1/UCF101 Dataset/tvl1_flow/v/v_PlayingViolin_g22_c04
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)                 #107    #10 or 7 or ...

        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V

        #    print h_image  #/media/ming/DATADRIVE1/UCF101 Dataset/tvl1_flow/u/v_PlayingViolin_g22_c04/frame000107.jpg
        #    print imgH   #<PIL.JpegImagePlugin.JpegImageFile image mode=L size=341x256 at 0x7F4D80719610>
        #    print H    #[torch.FloatTensor of size 1x224x224]
        #    print flow   #[torch.FloatTensor of size 20x224x224]  [18,:,:] and [19,:,:] nonzero, others are all zeros

            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):          #spatial different
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')  # Note: '-' has been removed
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1 
        data = self.stackopf()

        #sample
        if self.mode == 'train':
            sample = (data,label)
        #    print '.................load flow train sample......@index=%d..............'%idx
        #    print sample           #([torch.FloatTensor of size 20x224x224], 66)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        #    print '.................load flow valid sample........@index=%d..............'%idx
        #    print sample           #('CricketShot_g02_c01', [torch.FloatTensor of size 20x224x224], 23)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split):      #spatial does not have

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel            #spatial does not have
        self.data_path=path
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        # {'HandStandPushups_g23_c04': 37, ..., 'CricketShot_g15_c03': 24} (9537 3783)
        self.train_video, self.test_video = splitter.split_video()
        
    def load_frame_count(self):
        print '==> Loading frame number of each video'
        #with open('dic/frame_count.pickle','rb') as file:
        with open('/media/ming/DATADRIVE1/two stream/two-stream-code/dataloader/dic/frame_count.pickle', 'rb') as file:
            dic_frame = pickle.load(file) #{'v_Lunges_g07_c01.avi': 248, 'v_Haircut_g18_c04.avi': 263, ..., 'v_Typing_g09_c06.avi': 249}

        file.close()

        for line in dic_frame :      # v_Lunges_g07_c01.avi
            videoname = line.split('_',1)[1].split('.',1)[0]  # Lunges_g07_c01
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]
    #    print '----frame_count----------------------------length:', len(self.frame_count)   #13320
    #    print self.frame_count  #{'PommelHorse_g01_c03': 373, 'PommelHorse_g01_c02': 400, ..., 'PoleVault_g02_c02': 79}

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        print '==> Generate frame numbers of each training video'
        self.dic_video_train={}
        for video in self.train_video:
        #    print video              #  MilitaryParade_g17_c04
        #    print self.train_video   #{'MilitaryParade_g17_c04': 53, 'MilitaryParade_g17_c05': 53,...}

            nb_clips = self.frame_count[video]-10+1   # 133        note:frame_count result is 142, actually is 143  [spatial different]
            key = video +'-' + str(nb_clips)          # MilitaryParade_g17_c04-133      [spatial different]
            self.dic_video_train[key] = self.train_video[video]  #{'MilitaryParade_g17_c04-133': 53}

    def val_sample19(self):       #spatial different
        print '==> sampling testing frames'
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-10+1)/19)          #19
            for index in range(19):
                clip_idx = index*sampling_interval           #0
                key = video + '-' + str(clip_idx+1)          #PommelHorse_g01_c03-1
                self.dic_test_idx[key] = self.test_video[video]
        #    print self.dic_test_idx          #{'PommelHorse_g01_c03-1': 69,'PommelHorse_g01_c03-20': 69,...}
    #    print '----------------dic_testing_index,length:(', len(self.dic_test_idx), ')'  # 71877


    def train(self):             #spatial different
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Training data :',len(training_set),' videos'
        training_set[1][0].size()
        #print '.............training_set_sucessful......................'

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):       #spatial different
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Validation data :',len(validation_set),' frames'
        validation_set[1][1].size()
        #print validation_set[1]
        #print '........................validate_set_sucessful..........'

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='/home/ubuntu/data/UCF101/tvl1_flow/',
                                        ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
                                        ucf_split='01'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader