import os, pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):    #Obtain class name and corresponding label (char)
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()  #return list according to lines: ['1 ApplyEyeMakeup\n', '2 ApplyLipstick\n', ..., '100 WritingOnBoard\n', '101 YoYo\n']
            content = [x.strip('\r\n') for x in content]  #remove the specified character in the head and tail of the string: ['1 ApplyEyeMakeup', '2 ApplyLipstick', ...,'100 WritingOnBoard', '101 YoYo']
        f.close()
        for line in content:
            label,action = line.split(' ')  #slice a string by specifying a delimeter
            if action not in self.action_label.keys():
                self.action_label[action]=label
      #  print self.action_label   #{'MilitaryParade': '53', 'TrampolineJumping': '94', ...,'Haircut': '34', 'TennisSwing': '92'}

    def split_video(self):  # Obtain Training and Test videos and the corresponding labels
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
         #   print path      # /media/ming/DATADRIVE1/two stream/two-stream-code/UCF_list/
         #   print subdir    # []
         #   print files     #['classInd.txt','testlist01.txt','testlist02.txt','testlist03.txt','trainlist01.txt','trainlist02.txt','trainlist03.txt']
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    print '-----------------split_train_video-----------------------------'
                    # train_video={'HandStandPushups_g23_c04': 37, 'MilitaryParade_g17_c05': 53, ..., 'CricketShot_g15_c03': 24}
                    train_video = self.file2_dic(self.path+filename)

                if filename.split('.')[0] == 'testlist'+self.split:
                    print '-----------------split_test_video-------------------------------'
                    test_video = self.file2_dic(self.path+filename)

        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')'
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)
        print '==> new-----------------Training video, Validation video):(', len(train_video), len(test_video), ')'
    #    print self.train_video[100000000000000000000000000000000]

        return self.train_video, self.test_video

    def file2_dic(self,fname):     #Make all videos in the database have labels (int)
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:                                   # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1 just for trainlist01.txt, testlist01.txt doesn't has labels
            video = line.split('/',1)[1].split(' ',1)[0]       # v_ApplyEyeMakeup_g08_c01.avi
            key = video.split('_',1)[1].split('.',1)[0]        # ApplyEyeMakeup_g08_c01
            label = self.action_label[line.split('/')[0]]      # 1
            dic[key] = int(label)  # char -> int
        return dic

    #lower(S):S->s   HandstandPushups/v_HandStandPushups_g01_c01.avi -> HandstandPushups/v_HandstandPushups_g01_c01.avi
    def name_HandstandPushups(self,dic):        # called function
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print len(train_video),len(test_video)