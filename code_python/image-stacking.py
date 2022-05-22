# import necessary packages
from socketserver import DatagramRequestHandler
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np 
from skimage import measure
from skimage.measure import regionprops
from dataUtil import DataUtil

# function to overlay the line defining lesions over the image
def overlay_mask(orgimg,masks,colors=None):
    """
    orgimg: The original image as numpy array (2D)
    masks: List of masks as numpy array (2D)
    colors: List of respective colors to plot
    """

    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray')


    if colors is None:
      colors=cm.rainbow(np.linspace(0,1,len(masks)))

    for i,mask in enumerate(masks): 
      color = colors[i]
      contours = measure.find_contours(mask, 0)
      for n, contour in enumerate(contours):
          plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)

    plt.xticks([])
    plt.yticks([])

    return plt

# function to convert dicom files to nifti
def dicom_to_nifti(dir, output):

    # define the reader and get the filenames
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(str(dir))
    reader.SetFileNames(dcm_names)
    
    # get the images
    image = reader.Execute()
        
    # output the nifti file
    sitk.WriteImage(image, output)

# defines the main method, called when the file is called
def main():
    # get the image directories from the input path
    dcm_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000011/19000101/T2W_TSE_ax/801'

    # isolate the folders
    imgs = DataUtil.getSubDirectories(dcm_path)
    path_to_dicoms = imgs[0]
    imgID = str(path_to_dicoms).split('/')[-1]
    
    # define the output path
    nii_path = fr'/Volumes/GoogleDrive/My Drive/Research/Shiradkar Lab/to_nifti/{imgID}.nii.gz'
    
    # call the conversion function
    dicom_to_nifti(path_to_dicoms, nii_path)

    # plt = overlay_mask()

if __name__ == "__main__":
    main()