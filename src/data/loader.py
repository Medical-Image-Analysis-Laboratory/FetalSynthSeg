from pathlib import Path
from pathlib import Path
from monai.transforms import (
    LoadImage,
    SignalFillEmptyd,
    CropForegroundd,
    SpatialPadd,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
)
import nibabel as nib
import warnings

warnings.filterwarnings("ignore")
class ImageLoader:
    def __init__(self,
                 rescale_res: float = 0.5,
                 transforms=None,
                 ):

        self.loader = LoadImage(reader='NibabelReader')
        self.filler = SignalFillEmptyd(keys=['image', 'label'],
                                       replacement=0,)
        self.rescale_res = rescale_res
        self.transforms = transforms
        self.orientationd = Orientationd(keys=['image', 'label'], axcodes='RAS',
                                         allow_missing_keys=True,)

    def load_image(self, img_path):
        image = self.loader(img_path)

        # add channel dim
        image = image.unsqueeze(0)

        data = {'image': image, 'label': image}
        data = self.orientationd(data)

        # fill nans with zeros
        data = self.filler(data)

        self.cropper = CropForegroundd(keys=['image', 'label'],
                                       source_key='image',
                                       allow_smaller=True,
                                       allow_missing_keys=True,
                                       margin=0)
        # crop pad to 256x256x256 (since resampling can change the size of the image)
        data = self.cropper(data)

        # apply resampling transforms to homogenize
        # the data across the datasets to 0.5mm isotropic

        self.resample = Spacingd(keys=['image', 'label'],
                                 pixdim=(self.rescale_res,
                                         self.rescale_res,
                                         self.rescale_res),
                                 mode=('bilinear    ', 'nearest'),
                                 allow_missing_keys=True,)
        data = self.resample(data)
        self.padder = SpatialPadd(keys=['image', 'label'],
                                  spatial_size=(256, 256, 256),
                                  allow_missing_keys=True,
                                  mode='constant')
        data = self.padder(data)
        self.cropper_spat = CenterSpatialCropd(keys=['image', 'label'],
                                               allow_missing_keys=True,
                                               roi_size=(256, 256, 256))
        data = self.cropper_spat(data)

        # apply additional augmentations if needed
        if self.transforms:
            data = self.transforms(data)

        return data

    def save_image(self, pred_lab, out_path, img_path):
        # invert pre-processing steps
        img_path = Path(img_path)
        data = {'label': pred_lab}
        data = self.cropper_spat.inverse(data)
        data = self.padder.inverse(data)
        data = self.resample.inverse(data)
        data = self.cropper.inverse(data)
        data = self.orientationd.inverse(data)
        out_path = Path(out_path)

        pred_lab_array = data['label']
        pred_lab_array = pred_lab_array.detach().cpu().numpy()
        pred_lab_array = pred_lab_array.astype('int16')

        result_image = nib.Nifti1Image(pred_lab_array[0], affine=pred_lab.affine)
        nib.loadsave.save(img=result_image, filename=str(out_path))
