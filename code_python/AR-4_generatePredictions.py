import shutup
import h5py
import torchvision.models as models
from torch import nn
import torch 
import numpy as np
import os 
import sys 
import torch.nn.functional as F
from torchvision import models 
import os
import pandas as pd
import pingouin as pg
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm



class DenseNet(nn.Module):
    def __init__(self,path):
        super(DenseNet, self).__init__()
        
        # get the pretrained DenseNet201 network
        self.densenet = models.densenet121()
        self.densenet.classifier = nn.Linear(1024,2)

        checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
        self.densenet.load_state_dict(checkpoint)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)

        x = x.view((1, 1024))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

class alexnet(nn.Module):

    def __init__(self, num_classes=2):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            nn.Dropout(0.6),
            nn.Dropout(0.6),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Dropout(0.6),
            nn.Linear(4096, 4096),
        )

        self.attention = nn.Sequential(
            nn.Linear(4096, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        A = None
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier(x)

        A = self.attention(feat)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, feat)  # 1 x N 

        Y_prob = self.final(M)

        return Y_prob,feat,A

class AlexNet(nn.Module):
    def __init__(self,path):
        super(AlexNet, self).__init__()
        
        # get the pretrained DenseNet201 network
        self.alexnet = alexnet()

        checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
        self.alexnet.load_state_dict(checkpoint)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.alexnet.features
        
        # add the average global pool
        self.avgpool = self.alexnet.avgpool
        
        # get the classifier of the vgg19
        self.classifier = self.alexnet.classifier
        self.attention = self.alexnet.attention
        self.final = self.alexnet.final
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        feat = self.classifier(x)

        A = self.attention(feat)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, feat)  # 1 x N 

        Y_prob = self.final(M)

        return Y_prob
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

class SqueezeNet(nn.Module):
    def __init__(self, path):
        super(SqueezeNet, self).__init__()
        
        self.squeezenet = models.squeezenet1_0()
        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.squeezenet.load_state_dict(checkpoint)

        self.features_conv = self.squeezenet.features

        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.classifier = self.squeezenet.classifier

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)

        x = self.global_avg_pool(x)

        x = self.classifier(x)

        return torch.flatten(x,1)

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

def get_prediction(model,timg, label):
    
    # timg = timg[None]
    timg = timg[None]
    
    timg = torch.from_numpy(timg)

    shutup.please()
    pred = model(timg)
    shutup.jk()
    probs = F.softmax(pred,dim = 1)

    probs = probs.detach().data.cpu().numpy().ravel()[-1]

    return probs

def get_auc(ytrue, ypred):
    auc = metrics.roc_auc_score(ytrue, ypred)

    return auc


if __name__ == "__main__":

    psize = (224,224)
    rads = ['ST', 'LB']
    # rads = ['SHT', 'LKB'] # try with these checkpoints

    icc_table = pd.DataFrame(columns=['FileName', 'Rad', 'Pred'])
    for rad in rads:

        dataset = 'newsplits'
        # hdf5_dir = fr"test_set_{dataset}"
        hdf5_dir = fr"hdf5_{rad}_{dataset}"

        # f1path = fr"{DATA_DIR}/outputs/hdf5/test_set/test_UH.h5"
        f1path = fr"/Volumes/GoogleDrive/My Drive/racial-disparity-pc/model-outputs/racial-disparity-hdf5/test.h5"
        # f1path = fr"{DATA_DIR}/outputs/hdf5/hdf5_SHT_48/prostatex_1/val.h5"

        f1 = h5py.File(f1path,'r',libver='latest')
        
        d1 = f1["data"]
        
        labels1 = f1["labels"]
        names1 = f1["names"]


        print(rad)
        # m1path = fr"{DATA_DIR}/outputs/models/newsplits/keep_{rad}/checkpoint.pt"
        m1path = fr"/Volumes/GoogleDrive/My Drive/racial-disparity-pc/model-outputs/racial-disparity-models/rdp-aa-{rad}/early-stop_{rad}.pt"


        m1 = SqueezeNet(m1path)
        m1.eval()

        indiv_table = pd.DataFrame(columns=['FileName', 'Pred', 'PRED', 'TRUE'])
        avg_table = pd.DataFrame(columns=['FileName', 'Pred', 'TRUE'])

        ytrue = []
        ypred = []

        for sample in tqdm(range(d1.shape[0])): 
            
            ###################################################
            # Name = {dataset}_{patnum}_{lsnum}_{slc}_{label} #
            ###################################################
            name = names1[sample]
            name = name.decode('utf-8')

            pat = name.split('_L')[0]

            _img = d1[sample]

            slnos = np.unique(np.nonzero(_img)[0])
            label = labels1[sample]

            slc = name.split('_')[-1]
            ls = name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2]

            ##########################################
            probs = get_prediction(m1, _img, label)
            ytrue.append(label)
            ypred.append(probs)

            # print(fr"Name: {name}, Pred: {probs}, Label: {label}, Progress: {j/d1.shape[0]}")
            ##########################################

            indiv_table.loc[len(indiv_table)+1] = [name, probs, round(probs), label]
            icc_table.loc[len(icc_table)+1] = [name, rad, probs]

            # import pdb; pdb.set_trace()

            if not os.path.exists(outputdir := fr"/Volumes/GoogleDrive/My Drive/racial-disparity-pc/model-outputs/predictions/test_preds_{rad}"):
                os.mkdir(outputdir)

        auc = get_auc(ytrue, ypred)
        acc = metrics.accuracy_score(ytrue, [round(pred) for pred in ypred])
        print(fr"Rad: {rad}, AUC: {auc}, ACC: {acc}")
        indiv_table.to_csv(fr'{outputdir}/predictions_{rad}.csv')

               
    icc = pg.intraclass_corr(data=icc_table, targets='FileName', raters='Rad', ratings='Pred', nan_policy='omit')
    icc.set_index('Type')

    print(icc)

                
    print('done')
     

