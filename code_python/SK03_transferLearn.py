# import modules
import shutup
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import tables 
import pickle

from torch import device, nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from pytorchtools import EarlyStopping
from torchvision import models
from random import randint
from sklearn import metrics
from pathlib import Path

'''
@author Sakin Kirti
@date 06/17/2022

script to train SqueezeNet model for racial disparity prostate cancer project
'''

# set the torch seed
torch.manual_seed(1234)

# global vars
source_dir = str(Path(os.getcwd()).parent)

class ProstateDatasetHDF5(Dataset):
    '''
    @author Ansh Roge
    a class to define a dataset for easy loading a torch dataLoader
    '''

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

        data = self.data[index,:,:,:]   
        img = data[(0,1,2),:,:]
        
        if self.names is not None:
            name = self.names[index]
        
        label = self.labels[index]
        self.file.close()
        
        out = torch.from_numpy(img[None])
        
        return out,label,name

    def __len__(self):
        return self.nitems

def get_data(path, bs, phases):
    '''
    from the hdf5 file, split the data into a dataLoader with it's different phases
    - train, val, test
    '''
    
    # notify
    print('SEPARATING THE DATA FROM HDF5 FILES INTO DATALOADER OBJECTS')

    # define the returning variables
    dataLoader = {}
    dataLabels = {}

    nw = 2 # num workers
    # iterate through the phases
    for phase in phases:
        # read the h5 file according to the phase and isolate the labels and store
        filename = f'{path}/{phase}.h5'
        file = h5py.File(filename, libver='latest', mode='r')
        labels = np.array(file['labels'])
        file.close()
        dataLabels[phase] = labels

        # generate the loader and store
        data = ProstateDatasetHDF5(filename)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=bs, num_workers=nw, shuffle=True)
        dataLoader[phase] = loader

    # return the loader and labels
    return dataLoader, dataLabels    

def data_check(dataLoader, dataLabels, phases, labels):
    '''
    check the data for:
    - the train/val/test split
    - the class distribution
    '''
    
    # notify
    print('CHECKING THE SPREAD OF THE DATA')

    # print the size of the train, val, test datasets for the user
    phase_n = {}
    phase_label_count = {}
    total_n = 0
    for phase in phases:
        # count per phase
        phase_n[phase] = len(dataLoader[phase].dataset)

        # calculate total n
        total_n += len(dataLoader[phase].dataset)

        # count per label
        for label in labels:
            phase_label_count[f'{phase}_{label}'] = np.count_nonzero(dataLabels[phase] == label)

    # print for the user
    print(f'this is a {phase_n["train"] / total_n} / {phase_n["val"] / total_n} / {phase_n["test"] / total_n} for train / val / test splits')
    print(f'train set [n: {phase_n["train"]}, class 0%: {phase_label_count["train_0"] / phase_n["train"]}, class 1%: {phase_label_count["train_1"] / phase_n["train"]}]')
    print(f'val set: [n: {phase_n["val"]}, class 0%: {phase_label_count["val_0"] / phase_n["val"]}, class 1%: {phase_label_count["val_1"] / phase_n["val"]}]')
    print(f'test set: [n: {phase_n["test"]}, class 0%: {phase_label_count["test_0"] / phase_n["test"]}, class 1%: {phase_label_count["test_1"] / phase_n["test"]}]')

def get_model(device):
    '''
    generate a squeezenet model from the pytorch models library for training
    see SK00_racialDisparitySqueezeNet.py for details
    '''

    # notify
    print('GENERATING A FRESH SQUEEZENET ARCHITECTURE TO TRAIN')
    
    # define the model
    model = models.squeezenet1_0()
    model.classifier = nn.Sequential(
        nn.Dropout(0.55),
        nn.Conv2d(512, 2, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1))
    )

    # set the model to train on device
    model.to(device)
    print(model.__class__.__name__)

    # return
    return model, model.__class__.__name__

def train(model, arch, modelname, device, num_epochs, learning_rate, weight_decay, batch_size, patience, dataLoader, dataLabels, cv, rad):
    '''
    method to train the model
    '''

    # notify
    print('TRAINING IN PROGRESS...')
    
    # get the training labels
    trainlabels = dataLabels['train']
    zeros = (trainlabels == 0).sum()
    ones = (trainlabels == 0).sum()
    
    # set the base weights
    weights = [0, 0]
    if zeros > ones:
        weights[0] = 1
        weights[1] = float(zeros)/ones
    else:
        weights[1] = 1
        weights[0] = float(ones)/zeros
    class_weights = torch.FloatTensor(weights).cuda(device)

    # set the network parameters
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # set up early stopping
    checkpoint_path = f'{source_dir}/model-outputs/racial-disparity-models/{modelname}'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'{checkpoint_path}/early-stop_{rad}.pt')

    # set storage locations
    storage_folder = f'{source_dir}/model-outputs/racial-disparity-models/{modelname}'
    display = ['val', 'test']
    results = {}
    results['patience'] = patience

    # set storage for display options
    result_display = {'train loss': [], 'train acc': [], 'train auc': [], 
                      'val loss': [], 'val acc':[], 'val auc': []}

    # start training
    for epoch in range(num_epochs):
        pred_df_dict = {}
        results_dict = {}

        # iterate through phases
        for phase in ['train', 'val']:
            # set the mode
            if phase == 'train': model.train()
            else: model.eval()

            # to understand false positives, negatives
            confusion_matrix = np.zeros((2,2))

            # set up storage for predictions
            loss_vector = []
            ytrue = []
            ypred = []
            ynames = []

            # iterate through the data
            for (data, label, name) in dataLoader[phase]:
                data = data[0]
                label = label.long().to(device)
                data = Variable(data.float().cuda(device))

                # learn
                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    shutup.please()
                    output = model(data)
                    shutup.jk()

                    # maximize the output (with a max value)
                    try:
                        _, pred_label = torch.max(output, 1)
                    except:
                        import pdb; pdb.set_trace()

                    # loss
                    loss = criterion(output, label)
                    loss_vector.append(loss.detach().data.cpu().numpy())

                    # probability
                    probs = F.softmax(output, dim=1)
                    probs = probs[:,1]

                    # backward pass
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # get the y stats
                    ypred.extend(probs.cpu().data.numpy().tolist())
                    ytrue.extend(label.cpu().data.numpy().tolist())
                    ynames.extend(list(name))

                    # calculate the confusion matrix
                    pred_label=pred_label.cpu()
                    label=label.cpu()
                    for p,l in zip(pred_label,label):
                        confusion_matrix[p,l]+=1
                    
                    # reset for next training
                    torch.cuda.empty_cache()

            # calculate overall stats
            torch.cuda.empty_cache()
            total = confusion_matrix.sum()
            acc = confusion_matrix.trace() / total
            loss_avg = np.mean(loss_vector)
            auc = metrics.roc_auc_score(ytrue, ypred)

            # save the stats
            columns = ['Filename', 'True', 'Pred', 'Phase']
            pred_df = pd.DataFrame(np.column_stack((ynames, ytrue, ypred, [phase]*len(ynames))), columns=columns)
            pred_df_dict[phase] = pred_df

            results_dict[phase] = {}
            results_dict[phase]['loss'] = loss_avg
            results_dict[phase]['auc'] = auc
            results_dict[phase]['acc'] = acc

            # print some data to the terminal and save for displaying results
            if phase == 'train':
                print("Epoch: {}, Phase: {}, Loss: {}, Acc: {}, Auc: {}".format(epoch, phase, loss_avg, acc, auc))
                result_display['train acc'].append(acc); result_display['train auc'].append(auc); result_display['train loss'].append(loss_avg)
            elif phase in display:
                print("EVALUATION --> PLEASE CHECK --> Epoch: {}, Phase: {}, Loss: {}, Acc: {}, Auc: {}".format(epoch, phase, loss_avg, acc, auc))
                result_display['val acc'].append(acc); result_display['val auc'].append(auc); result_display['val loss'].append(loss_avg)

            # save the model until an ideal model is found, then stop
            if phase == 'val':
                df = pd.concat([pred_df_dict['val'], pred_df_dict['train']], axis=0, ignore_index=True)

                early_stopping(loss_avg, model)

                # save the model weights
                if not os.path.exists(fr"{storage_folder}"):
                    os.makedirs(fr"{storage_folder}")
                torch.save(model.state_dict(), fr"{storage_folder}/checkpoint_{rad}.pt")
                
                # save the predictions to a csv
                df.to_csv(fr"{storage_folder}/predictions_{rad}.csv")
                
                # save the AUC values
                with open(fr"{storage_folder}/aucs_{rad}.pkl", 'wb') as outfile:
                    pickle.dump(results_dict, outfile, protocol=2)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                final_auc = auc

        if early_stopping.early_stop:
            break
    
    # return
    return result_display, model

def generate_figures(data, filetype, rad, save_loc):
    '''
    generate figures to show how the learning went for the user to get a quick summary
    - training and validation accuracy
    - training and validation loss
    - training and validation auc
    '''
    
    print('\nGENERATING SOME FIGURES TO DEMONSTRATE LEARNING...')

    # create the save_loc if it doesnt exist
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    # generate x values
    epochs = [i for i in range(1, len(data['train loss'])+1)]

    # graph of training epoch vs loss (both training and val)
    plt.plot(epochs, data['train loss'], label='training loss')
    plt.plot(epochs, data['val loss'], label='validation loss')
    plt.title('average loss for each training epoch'); plt.xlabel('training epoch'); plt.ylabel('average loss'); plt.ylim(0, 1)
    plt.legend(['train loss', 'validation loss'])
    plt.savefig(f'{save_loc}/loss-{rad}.{filetype}')
    plt.clf()

    # graph of training epoch vs accuracy (both training and val)
    plt.plot(epochs, data['train acc'], label='training accuracy')
    plt.plot(epochs, data['val acc'], label='validation accuracy')
    plt.title('accuracy for each training epoch'); plt.xlabel('training epoch'); plt.ylabel('accuracy'); plt.ylim(0, 1)
    plt.legend(['train accuracy', 'validation accuracy'])
    plt.savefig(f'{save_loc}/accuracy-{rad}.{filetype}')
    plt.clf()

    # graph of training epoch vs AUC (both training and val)
    plt.plot(epochs, data['train auc'], label='training auc')
    plt.plot(epochs, data['val auc'], label='validation auc')
    plt.title('auc for each training epoch'); plt.xlabel('training epoch'); plt.ylabel('area under curve'); plt.ylim(0, 1)
    plt.legend(['train AUC', 'validation AUC'])
    plt.savefig(f'{save_loc}/auc-{rad}.{filetype}')
    plt.clf()

def main():
    '''
    main method which iterates over the different label groups 
    and trains a new model for each label groups
    '''

    print('starting learning with SqueezeNet model')

    # define some basic parameters
    device = torch.device('cuda:0')
    batch_size = 1
    num_epochs = 500
    learning_rate = 5e-7
    weight_decay = 1e-5
    patience = 10
    dataset = 'rdp-aa'
    rads = ['ST', 'LB']
    phases = ['train', 'val', 'test']
    labels = [0, 1]
    cv = 1

    # iterate through the rads
    for rad in rads:
        print(f'rad: {rad}, learning rate: {learning_rate}, weight decay: {weight_decay}, fold: {cv}')

        # load the data
        splitspath = f'{source_dir}/model-outputs/rdp-hdf5/hdf5-{rad}'
        dataLoader, dataLabels = get_data(splitspath, batch_size, phases)

        # check the data - just notify of the user
        data_check(dataLoader, dataLabels, phases, labels)

        # initialize the model
        modelname = f'{dataset}-{rad}'
        print(modelname, cv)
        model, arch = get_model(device)

        # train the model
        data_display, model = train(model, arch, modelname, device, num_epochs, learning_rate, weight_decay, batch_size, patience, dataLoader, dataLabels, cv, rad)
        print('done')

        # generate some figures
        generate_figures(data_display, filetype='jpg', rad=rad, save_loc=f'{source_dir}/model-outputs/figures')

if __name__ == '__main__':
    main()