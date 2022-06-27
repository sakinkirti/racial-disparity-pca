from ssl import VERIFY_X509_STRICT
import h5py
import torchvision.models as models
from torch import nn
import torch 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from skimage import measure 
import os 
from joblib import Parallel, delayed
from skimage.morphology import binary_erosion, selem
from skimage.transform import resize
import sys 
# sys.path.insert(1,fr"../Code_general")
sys.path.append(fr"sourcecode/classification_lesion/Code_general")
from medImageProcessingUtil import MedImageProcessingUtil 
from dataUtil import DataUtil 
import SimpleITK as sitk 
import torch.nn.functional as F
from torchvision import models 
from skimage.transform import resize as skresize
import os

DATA_DIR = 'sourcecode/classification_lesion'


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

def overlay_heatmap(orgimg,heatmap,mask,lb):
    contours = measure.find_contours(mask, 0)

    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray',vmin =0, vmax = 1)

    if lb == 1:
        plt.imshow(heatmap, cmap = 'Reds', interpolation=None, vmin = 0, vmax = 1, alpha = heatmap)
    else:
        plt.imshow(heatmap, cmap = 'Blues', interpolation=None, vmin = 0, vmax = 1, alpha = heatmap)

    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='g')

    plt.xticks([])
    plt.yticks([])

    return plt 



def overlay_heatmaps(orgimg,heatmap1, heatmap2,mask):
    # Add back heatmap2 if needed

    contours = measure.find_contours(mask, 0)

    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray',vmin =0, vmax = 1)


    plt.imshow(heatmap2, cmap = 'Reds', interpolation=None, vmin = 0, vmax = 1, alpha = heatmap2)
    plt.imshow(heatmap1, cmap = 'Blues', interpolation=None, vmin = 0, vmax = 1, alpha = heatmap1)

    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='g')

    plt.xticks([])
    plt.yticks([])

    return plt 


def overlay_mask(orgimg,mask):

    try:
        contours = measure.find_contours(mask, 0)
    except:
        import pdb 
        pdb.set_trace()

    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray', vmin = 0, vmax = 1)

    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='g')

    plt.xticks([])
    plt.yticks([])

    return plt 

def orig_heatmap(orgimg, mask, heatmap, name, pat, slc):

    slc = int(slc)

    lsno = name.split('L')[-1].split('_')[0]
    patno = pat.split('_')[-1].strip('0')

    img = sitk.ReadImage(fr"annotations_master/Original_Sequences/{pat}/T2W_std.nii.gz")
    img = sitk.GetArrayFromImage(img)[slc]

    pm = sitk.ReadImage(fr"annotations_master/Original_Sequences/{pat}/PM.nii.gz")
    pm = sitk.GetArrayFromImage(pm)[slc]

    border = np.zeros(mask.shape)
    
    contours = measure.find_contours(mask, 0)

    for contour in contours[1]:
        border[int(contour[0]), int(contour[1])] = 1

    re_heatmap = skresize(heatmap, mask.shape, order=1)

    comp_img = np.zeros(img.shape)
    comp_img[int((img.shape[0]-re_heatmap.shape[0])/2):int((img.shape[0]+re_heatmap.shape[0])/2), int((img.shape[1]-re_heatmap.shape[1])/2):int((img.shape[1]+re_heatmap.shape[1])/2)] = re_heatmap

    plt.imshow(img, cmap='gray')
    plt.imshow(comp_img, cmap='viridis', alpha=0.5)
    plt.show()

    import pdb; pdb.set_trace()
    


def get_heatmap(model,timg,lb, label):
    # import pdb; pdb.set_trace()
    
    # timg = timg[None]
    timg = timg[None]
    
    timg = torch.from_numpy(timg)

    import pdb; pdb.set_trace()
    pred = model(timg)
    probs = F.softmax(pred,dim = 1)

    pred[:, lb].backward()

    gradients = model.get_activations_gradient()


    pooled_gradients =  torch.mean(gradients, dim = 0)
    pooled_gradients =  torch.mean(pooled_gradients, dim = 1)
    pooled_gradients =  torch.mean(pooled_gradients, dim = 1)


    activations = model.get_activations(timg).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]


    heatmap = torch.mean(activations, dim=1).squeeze()


    print(fr"Probability: {probs.detach().data.cpu().numpy().ravel()[-1]}, Heatmap Max: {torch.max(heatmap)}, TRUE: {label}")

    heatmap = np.maximum(heatmap, 0)

    # if not heatmap.sum() < 0.01:
        # normalize the heatmap

    heatmap /= torch.max(heatmap)
    # heatmap[np.isnan(heatmap)] = 0
    # heatmap = np.nan_to_num(heatmap)


    heatmap = heatmap.detach().data.cpu().numpy()
    probs = probs.detach().data.cpu().numpy().ravel()[-1]
    # heatmap = resize(heatmap, (192,192))



    return heatmap, probs


if __name__ == "__main__":

    psize = (224,224)

    fold = 1

    inputset = 'CCF'
    modality = 'test'

    print(fr'{inputset}_{modality}')
    rads = ['JR', 'LKB', 'RDW', 'SHT', 'SV']
    # rads = ['SV']
    
    for rad in rads:
        print(rad)

        # f1path = fr"outputs/hdf5/hdf5_{rad}_48/prostatex_{fold}/val.h5"
        # if rad == 'LKB':
        #     f1path = fr"outputs/hdf5/hdf5_LKB_48/prostatex_{fold}/val.h5"
        # if rad == 'RDWpot':
        #     f1path = fr"outputs/hdf5/hdf5_RDW_48/prostatex_{fold}/val.h5"

        # f1path = fr"{DATA_DIR}/outputs/hdf5/test_set_{inputset}/{modality}.h5"
        f1path = fr'{DATA_DIR}/outputs/hdf5/hdf5_{rad}_newsplits/{modality}.h5'

        f1 = h5py.File(f1path,'r',libver='latest')
        
        d1 = f1["data"]
        
        labels1 = f1["labels"]
        names1 = f1["names"]

        # m1path = fr"{DATA_DIR}/outputs/models_final/checkpoint_{rad}.pt"
        m1path = fr"{DATA_DIR}/outputs/models/preds/keep_{rad}/checkpoint.pt"
        m1 = SqueezeNet(m1path)
        # m1.load_state_dict(torch.load(m1path, map_location=lambda storage, loc: storage))
        m1.eval()

        activationmaps = None


        for j, sample in enumerate(range(d1.shape[0])):
            heatmaps = []
            
            # Name = ProstateX1_patnum_L1_slc_label
            name = names1[sample]
            name = name.decode('utf-8')

            pat = name.split('_L')[0]

            _orgimg = d1[sample,:,:]
            _img = d1[sample]

            slnos = np.unique(np.nonzero(_img)[0])
            label = labels1[sample]

            print(fr"Name: {name}, Label: {label}, Progress: {j/d1.shape[0]}")


            for lb in [0, 1]:
                slc = name.split('_')[-1]

                ##########################################
                # heatmap,probs = get_heatmap(m1,img,lb)
                heatmap, probs = get_heatmap(m1, _img, lb, label)
                heatmap = np.nan_to_num(heatmap)
                ##########################################

                if (label == 0 and probs > 0.0001 and probs < 0.4) or (label == 1 and probs > 0.6 and probs < 1.0):
                    pass
                outputfolder = fr"{DATA_DIR}/outputs/activationmaps/{inputset}/{rad}/{label}"

                if not os.path.exists(outputfolder):
                    os.makedirs(outputfolder)

                heatmap = skresize(heatmap,psize,order=1)


                heatmaps.append(heatmap)
                

            if len(heatmaps) > 0:
                mask = _orgimg[2]
                # mask = orgimg[2]
                # heatmap = np.ma.masked_where(mask == 0, heatmap)

                if mask.ndim != 2:
                    import pdb 
                    pdb.set_trace()

                # orgimg = orgimg[0]
                orgimg = _orgimg[0]

                # orig_heatmap(orgimg, mask, heatmaps[1], name, pat, slc)

                plt_org = overlay_mask(orgimg,mask)
                # plt_org.show()
                plt_org.savefig(fr"{outputfolder}/{name}_({label},{probs})_org_{lb}.png", bbox_inches = 'tight',pad_inches = 0)
                plt.close()

                plt_heat = overlay_heatmaps(orgimg,heatmaps[0],heatmaps[1],mask)
                # plt_heat.show()
                plt_heat.savefig(fr"{outputfolder}/{name}_({label},{probs})_map_{lb}.png", bbox_inches = 'tight',pad_inches = 0)
                plt.close()

                
    print('done')
     

