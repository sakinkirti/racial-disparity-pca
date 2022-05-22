import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from skimage import measure

img_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/to_nifti/prostate-000011/T2W.nii.gz"
ls_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/to_nifti/prostate-000011/LS1.nii.gz"

def main():
    # read the image and convert to np array
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    lesion_arr = sitk.GetArrayFromImage(sitk.ReadImage(ls_path))

    # flip the img
    img_arr = flip_np(img_arr)

    # get the slice
    slice, slices = find_mask(lesion_arr)
    print(slices)

    # overlay mask
    overlayedImg = overlay_mask(img_arr[slice], [lesion_arr[slice]])

    # save the image
    plt.show()
    plt.savefig('test/dcm_flipped_overlayed.jpg')
    plt.close()

def flip_np(img):
    return np.flip(img, 0)

def find_mask(mask):
    slice, y_ind, x_ind = np.where(mask != 0)
    return slice[0], np.unique(slice)

def overlay_mask(orgimg,masks,colors=None):
    # plot the image
    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray')

    # overlay the mask contour over the image
    if colors is None:
      colors = ['yellow']

    for i,mask in enumerate(masks): 
      color = colors[i]
      contours = measure.find_contours(mask, 0)
      for n, contour in enumerate(contours):
          plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)

    plt.xticks([])
    plt.yticks([])

    return plt

if __name__ == "__main__":
    main()