import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    FolderPath= "/Users/sakinkirti/Desktop/test/mask"
    PatienltList= [ f for f in os.listdir(FolderPath) if os.path.isdir(os.path.join(FolderPath,f))]
    PatienltList =[ f for f in PatienltList if f.find("prostate")>=0]
    for pt in PatienltList:
        SegMaskList=[ f for f in os.listdir(FolderPath+pt) if f.find("PIRADS")>0 and f.upper().find("MOD")<0]
        SegMaskList = [f for f in SegMaskList if f.endswith("nii.gz")]
        for i in range(0, len(SegMaskList)):
            SegImg_sitk= sitk.ReadImage(FolderPath+pt+"\\"+ SegMaskList[i])
            Origin=SegImg_sitk.GetOrigin()
            Spacing= SegImg_sitk.GetSpacing()
            Direction= SegImg_sitk.GetDirection()
            PixelData= sitk.GetArrayFromImage(SegImg_sitk)
            SegImg2= np.zeros(PixelData.shape,dtype=PixelData.dtype)
            for sl in range(0, PixelData.shape[0]-1):
                SegImg2[sl,:, :]=PixelData[sl+1, :, :]
                plt.imshow(SegImg2[sl-1,:,:])
            SegImg2[PixelData.shape[0]-1,:, :] = PixelData[0, :, :]
            SegImg2_sitk= sitk.GetImageFromArray(SegImg2, isVector=False)
            SegImg2_sitk.SetOrigin(Origin)
            SegImg2_sitk.SetSpacing(Spacing)
            SegImg2_sitk.SetDirection(Direction)
            writer1 = sitk.ImageFileWriter()
            writer1.SetFileName(FolderPath+pt+"\\"+ "Mod_"+SegMaskList[i])
            writer1.Execute(SegImg2_sitk)
        print("Processed->", pt)



