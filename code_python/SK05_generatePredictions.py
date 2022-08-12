import h5py
from sqlalchemy import true
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torchvision import models
from tqdm import tqdm

from SK00_racialDisparitySqueezeNet import SqueezeNet

'''
@author Sakin Kirti
@date 6/26/2022

script to test data on a model for racial disparity prostate cancer on some data
'''

# user input vars
source_dir = f'/Volumes/GoogleDrive/My Drive/racial-disparity-pc/model-outputs'
test_path = f'{source_dir}/racial-disparity-hdf5/test.h5'
checkpoint_path = f'{source_dir}/racial-disparity-models'

save_path = f'{source_dir}/predictions'

rads = ['ST', 'LB']

'''
method to get the data from the h5 files
'''
def get_data(test_data_path):
    print('\nGETTING THE TESTING DATA FROM H5 FILES...')

    # read in the h5 file
    file = h5py.File(test_data_path, mode='r', libver='latest')

    # separating the data, labels, names
    data = file['data']
    labels = file['labels']
    names = file['names']

    # print and return
    print('done')
    return data, labels, names

'''
method to generate a SqueezeNet architecture and load model weights
'''
def get_model(checkpoint):
    print('\nGENERATING A NEW SQUEEZENET ARCHITECTURE AND LOADING IN MODEL WEIGHTS...')

    # generate SqueezeNet architecture
    model = models.squeezenet1_0()
    model.classifier = nn.Sequential(
            nn.Dropout(0.55),
            nn.Conv2d(512, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
    )

    # load weights to squeezenet architecture
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # set model to eval
    model.eval()

    # print and return
    print('done')
    return model

'''
method to run the testing data through the model
'''
def test(model, data, labels, names, rad):
    print(f'\nTESTING THE MODEL ({rad}) ON SOME UNSEEN DATA')

    # set storage
    yprob = []
    ypred = []
    ytrue = []
    table = pd.DataFrame(columns=['FILENAME', 'PROB-PRED', 'FINAL-PRED', 'TRUE'])

    # iterate through the data
    for sample in tqdm(range(data.shape[0])):
        name = names[sample].decode('utf-8')
        label = labels[sample]
        img = data[sample]

        # load into torch
        img = torch.from_numpy(img)

        # predict
        probs = model(img)
        probs = F.softmax(probs)
        probs = probs.detach().data.cpu().numpy().ravel()[-1]

        # add to storage
        ytrue.append(label)
        ypred.append(round(probs))
        yprob.append(probs)

        # add to table
        table.loc[len(table) + 1] = [name, probs, round(probs), label]

    # print and return
    print('done')
    return (ytrue, yprob, ypred), table

'''
calculate and save fit statistics
'''
def calculate_save_model_metrics(prediction_data, table, path, rad):
    print('\nCALCULATING MODEL METRICS...')

    # isolate each list
    ytrue = prediction_data[0]
    yprob = prediction_data[1]
    ypred = prediction_data[2]

    # calculate acc and auc
    acc_score = metrics.accuracy_score(y_true=ytrue, y_pred=ypred)
    auc_score = metrics.roc_auc_score(y_true=ytrue, y_score=yprob)
    print(f'RAD: {rad}, ACC: {acc_score}, AUC: {auc_score}')

    # appent acc and roc
    table['ACC-SCORE'] = [acc_score]
    table['AUC-SCORE'] = [auc_score]

    # save and print
    table.to_csv(path)
    print('done')

'''
main method
'''
def main():
    print('GENERATING PREDICTIONS...')

    # read in the data
    data, labels, names = get_data(test_data_path=test_path)

    # loop through rads
    for rad in rads:
        # generate a model with squeezenet architecture and load weights
        model = get_model(checkpoint=f'{checkpoint_path}/rdp-aa-{rad}/early-stop_{rad}.pt')

        # pass the data through the model
        pred_data, table = test(model=model, data=data, labels=labels, names=names, rad=rad)

        # calculate fit statistics
        calculate_save_model_metrics(prediction_data=pred_data, table=table, path=f'{save_path}/preds_{rad}', rad=rad)

if __name__ == '__main__':
    main()