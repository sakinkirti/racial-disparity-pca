import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import h5py
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from pytorchtools import EarlyStopping
from random import randint
from sklearn.metrics import roc_auc_score
# from resnet import resnet101
import pandas as pd 
torch.manual_seed(1)
import tables 
import torchvision.models as models
import pickle
# from resnet import resnet18,resnet34,resnet50,resnet101,wide_resnet50_2
# from squeezenet import squeezenet

SOURCECODE_DIR = os.getcwd() + '/sourcecode/classification_lesion'
RAD_NAME = "JR"
AUGMENTATION = [4,8]

class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Dropout(0.5),
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

    def max_pool(self,x):
        maxs, inds = torch.max(x,0)
        maxs = maxs[None,:]
        return maxs  
    
    def avg_pool(self, x):
        avgs = torch.mean(x,0)
        avgs = avgs[None,:]
        return avgs

    def forward(self, x):
        # import pdb; pdb.set_trace()
        A = None
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier(x)

        # M = self.max_pool(feat)
        M = self.avg_pool(feat)
        # A = self.attention(feat)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, feat)  # 1 x N 

        Y_prob = self.final(M)

        return Y_prob,feat,A


class ProstateDatasetHDF5(Dataset):

    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        
        self.file.close()
        self.data = None
        self.mask = None
        self.names = None
        self.labels = None 
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.labels = self.tables.labels
                
        if "names" in self.tables:
            self.names = self.tables.names

        # import pdb; pdb.set_trace()
        data = self.data[index,:,:,:]   
        # img = data[:,(0,1,2),:,:]
        # mask = data[:,2,:,:]
        
        # img = data[:, (0,1,2),:]
        # mask = data[:,2,:]
        img = data[(0,1,2),:,:]
        # mask = data[2,:,:]
    
        # slices = np.unique(np.nonzero(mask)[0])
        
        # if slices.size != 0:
        #     img = img[slices]
        # else:
        #     slices = np.unique(np.nonzero(img)[0])
        #     img = img[slices]
                
        # mask[mask > 1] = 1 

        if self.names is not None:
            name = self.names[index]
        
        # import pdb; pdb.set_trace()
        label = self.labels[index]
        self.file.close()
            
        
        out = torch.from_numpy(img[None])
        
        return out,label,name

    def __len__(self):
        return self.nitems

def get_data(splitspathname,batch_size):
    trainfilename = fr"{splitspathname}/train.h5"
    valfilename = fr"{splitspathname}/val.h5"
    testfilename = fr"{splitspathname}/val.h5"

    train = h5py.File(trainfilename,libver='latest',mode='r')
    val = h5py.File(valfilename,libver='latest',mode='r')
    test = h5py.File(testfilename,libver='latest',mode='r')

    # import pdb; pdb.set_trace()
    
    trainlabels = np.array(train["labels"])
    vallabels = np.array(val["labels"])
    testlabels = np.array(test["labels"])

    train.close()
    val.close()
    test.close()

    data_train = ProstateDatasetHDF5(trainfilename)
    data_val = ProstateDatasetHDF5(valfilename)
    data_test = ProstateDatasetHDF5(testfilename)

    num_workers = 8

    trainLoader = torch.utils.data.DataLoader(dataset=data_train,batch_size = batch_size,num_workers = num_workers,shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset=data_val,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    testLoader = torch.utils.data.DataLoader(dataset=data_test,batch_size = batch_size,num_workers = num_workers,shuffle = False) 

    dataLoader = {}
    dataLoader['train'] = trainLoader
    dataLoader['val'] = valLoader
    dataLoader['test'] = testLoader

    dataLabels = {}
    dataLabels["train"] = trainlabels
    dataLabels["val"] = vallabels
    dataLabels["test"] = testlabels

    return dataLoader, dataLabels

def get_model(modelname,device):
    # model = AlexNet()
    # model = models.densenet121(num_classes=2, drop_rate=0.6)
    model = models.alexnet(num_classes=2)
    model.classifier = nn.Sequential(
            nn.Dropout(0.55),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.55),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,2)
        )


    # model = models.squeezenet1_0()
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.55),
    #     nn.Conv2d(512, 2, kernel_size=1),
    #     nn.ReLU(inplace=True),
    #     nn.AdaptiveAvgPool2d((1,1))
    # )

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, pretrained=True)

    # model = models.resnet34() 

    model.to(device)
    print(model.__class__.__name__)
    return model, model.__class__.__name__

def run(model,arch,modelname,device,num_epochs, learning_rate, weightdecay,batch_size, patience,dataLabels,cv, rad):

    trainlabels = dataLabels["train"]

    zeros = (trainlabels == 0).sum()
    ones = (trainlabels == 1).sum()

    weights = [0,0]

    if zeros > ones:
        weights[0] = 1
        weights[1] = float(zeros)/ones
    else:
        
        weights[1] = 1
        weights[0] = float(ones)/zeros
        
    class_weights = torch.FloatTensor(weights).cuda(device)


    criterion=nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)


    early_stopping = EarlyStopping(patience=patience, verbose=True)
    modelname = fr"{modelname}_{cv}"
    parentfolder = fr"{SOURCECODE_DIR}/outputs/models/preds/{arch}_{rad}"

    niter_total=len(dataLoader['train'].dataset)/batch_size

    display = ["val","test"]

    results = {} 

    results["patience"] = patience

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):

        pred_df_dict = {} 
        results_dict = {} 
        
        
        for phase in ["train","val"]:


            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            confusion_matrix=np.zeros((2,2))
            
            loss_vector=[]
            ytrue = [] 
            ypred = [] 
            ynames = [] 
            features = None 

            
            for ii,(data,label,name) in enumerate(dataLoader[phase]):

                data = data[0]

                label=label.long().to(device)
                            
                data = Variable(data.float().cuda(device))

                with torch.set_grad_enabled(phase == 'train'):

                    # output,feat,att = model(data)
                    output = model(data)

                    # feat = feat.detach().data.cpu().numpy()
                    # features = feat if features is None else np.vstack((features,feat))
                    
                    try:

                        _,pred_label=torch.max(output,1)

                    except:
                        import pdb 
                        pdb.set_trace()

                    loss = criterion(output, label)
                    probs = F.softmax(output,dim = 1)

                    probs = probs[:,1]

                    loss_vector.append(loss.detach().data.cpu().numpy())

                    if phase=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()        

                    ypred.extend(probs.cpu().data.numpy().tolist())
                    ytrue.extend(label.cpu().data.numpy().tolist())
                    ynames.extend(list(name))

                    pred_label=pred_label.cpu()
                    label=label.cpu()
                    for p,l in zip(pred_label,label):
                        confusion_matrix[p,l]+=1
                    
                    torch.cuda.empty_cache()
            
            
            # print(att)
            torch.cuda.empty_cache()
            total=confusion_matrix.sum()        
            acc=confusion_matrix.trace()/total
            loss_avg=np.mean(loss_vector)
            auc = roc_auc_score(ytrue,ypred)

            columns = ["FileName","True", "Pred","Phase"]
            pred_df = pd.DataFrame(np.column_stack((ynames,ytrue,ypred,[phase]*len(ynames))), columns=columns)
            pred_df_dict[phase] = pred_df

            results_dict[phase] = {} 
            results_dict[phase]["loss"] = loss_avg
            results_dict[phase]["auc"] = auc 
            results_dict[phase]["acc"] = acc 

            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
            elif phase in display:
                print("                 Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
                
            for cl in range(confusion_matrix.shape[0]):
                cl_tp=confusion_matrix[cl,cl]/confusion_matrix[:,cl].sum()

            if phase == 'val':
                df = pred_df_dict["val"].append(pred_df_dict["train"], ignore_index=True)

                early_stopping(loss_avg, model)
                output_dir = fr"{SOURCECODE_DIR}/outputs/auc_checkpoints/{modelname}"

                if not os.path.exists(fr"{parentfolder}"):
                    os.makedirs(fr"{parentfolder}")
                    
                torch.save(model.state_dict(), fr"{parentfolder}/checkpoint.pt")
                
                df.to_csv(fr"{parentfolder}/predictions.csv")
                
                with open(fr"{parentfolder}/aucs.pkl", 'wb') as outfile:
                    pickle.dump(results_dict, outfile, protocol=2)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                final_auc = auc

        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    print('new4')
    device = torch.device("cuda:0")
    batch_size = 1
    
    num_epochs = 500

    # learning_rate = 1e-6
    # weightdecay = 1e-5

    lr = 1e-5
    wd = 1e-5
    

    patience = 15

    dataset = "ccf"
    region = "peri6"
    rads = ['JR', 'LKB', 'RDW', 'SHT', 'SV']


    # rad = 'SV'

    cv = 1
    for rad in rads:
        print(fr"LR: {lr}, WD: {wd}, Fold: {cv}, Rad: {rad}, Aug: 48")

        splitspathname = fr"{SOURCECODE_DIR}/outputs/hdf5/hdf5_{rad}_48/prostatex_{cv}"
        # splitspathname = fr"{SOURCECODE_DIR}/outputs/hdf5/hdf5_{rad}_newsplits"
        
        dataLoader, dataLabels = get_data(splitspathname,batch_size)

        modelname = fr"{region}_{dataset}"
        print(modelname,cv)

        model, arch = get_model(modelname,device)

        run(model,arch,modelname,device,num_epochs, lr, wd,batch_size,patience,dataLabels,cv, rad)

    print('done')

