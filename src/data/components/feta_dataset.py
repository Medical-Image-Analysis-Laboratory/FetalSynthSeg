import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import (
    LoadImage,
    SignalFillEmptyd,
    CropForegroundd,
    SpatialPadd,
    Orientation,
    Spacingd,
    CenterSpatialCropd,
)

import nibabel as nib


class FeTADataset(Dataset):
    """Dataset class for the FeTA images.
    Responsible for reading real (not synthetic) FeTA images and segmentations."""

    def __init__(self, feta_path: str,
                 split: str,
                 split_file: str,
                 transforms=None,
                 # keep it for consistency with SynthDataset
                 synth_val_size: int | None = None,
                 validation: bool = False,
                 include_all: bool = False,
                 rescale_res: float = 0.5,
                 ):
        super().__init__()

        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split
        self.split_file = split_file
        self.transforms = transforms
        self.bids_path = Path(feta_path)
        self.rescale_res = rescale_res
        self.synth_val_size = synth_val_size
        self.loader = LoadImage(reader='NibabelReader',)
        self.filler = SignalFillEmptyd(keys=['image', 'label'],
                                       replacement=0,)
        self.validation = validation
        self.include_all = include_all
        self.__load_data()

    def __load_data(self):
        subj_df = pd.read_csv(self.split_file,)

        # select only subjects from the split (SRR-Site)
        split_subjects = subj_df[subj_df['Splits'].isin(self.split)]['participant_id'].values

        if not self.include_all:
            # if dataset used for training, omit internal validation subjects
            if not self.validation:
                internal_valid_subj = subj_df['InternalValidation'].astype(bool)
                internal_valid_subj = subj_df[internal_valid_subj]['participant_id'].values
                split_subjects = [x for x in split_subjects if x not in internal_valid_subj]
            # else use only internal validation subjects
            else:
                internal_valid_subj = subj_df['InternalValidation'].astype(bool)
                internal_valid_subj = subj_df[internal_valid_subj]['participant_id'].values
                split_subjects = [x for x in split_subjects if x in internal_valid_subj]

        subjects = [x for x in self.bids_path.glob('sub-*') if x.name in split_subjects]

        self.subjects = [x.name for x in subjects]
        self.img_paths = [str(list(x.glob('anat/*_T2w.nii.gz'))[0]) for x in subjects]
        self.segm_paths = [(x).replace('T2w', 'dseg') for x in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.loader(self.img_paths[idx])
        segm = self.loader(self.segm_paths[idx])
        name = self.segm_paths[idx]

        # add channel dim
        image = image.unsqueeze(0)
        segm = segm.unsqueeze(0)

        # orient to RAS for consistency
        image = Orientation('RAS')(image)
        segm = Orientation('RAS')(segm)

        data = {'image': image, 'label': segm, 'names': name}

        # fill nans with zeros
        data = self.filler(data)

        # crop pad to 256x256x256 (since resampling can change the size of the image)
        data = CropForegroundd(keys=['image', 'label'],
                               source_key='image',
                               allow_smaller=True,
                               margin=0)(data)

        # apply resampling transforms to homogenize
        # the data across the datasets to 0.5mm isotropic
        data = Spacingd(keys=['image', 'label'],
                        pixdim=(self.rescale_res,
                                self.rescale_res,
                                self.rescale_res),
                        mode=('bilinear', 'nearest'),
                        )(data)

        data = SpatialPadd(keys=['image', 'label'],
                           spatial_size=(256, 256, 256),
                           mode='constant',
                           )(data)

        data = CenterSpatialCropd(keys=['image', 'label'],
                                  roi_size=(256, 256, 256),
                                    )(data)

        # apply additional augmentations if needed
        if self.transforms:
            data = self.transforms(data)

        data['label'] = data['label'].long()

        return data['image'],  data['label'], name
