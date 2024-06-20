import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import (
    LoadImage,
    ScaleIntensityd,
    SignalFillEmptyd,
)
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from src.data.components.synth_gen import SynthGenerator


class SynthDataset(Dataset):
    def __init__(self,
                 feta_path: str,
                 split: str,
                 split_file: str,
                 transforms=None,
                 # keep it for consistency with SynthDataset
                 synth_val_size: int | None = None,
                 validation: bool = False,
                 include_all: bool = False,
                 rescale_res: float = 0.5,

                 generator_params: dict | str | None = None,
                 min_subclasses: int = 1,
                 max_subclasses: int = 6,
                 seed_path: str = None,
                 mlab_subclasses: None | dict = None,
                 segm_path: str = None,
                 ):
        """Args:
            feta_path (str): Path to the bids folder with the data.
            split (str): Split name to use.
            split_file (str): Path to .csv file with splits into sets.
            transforms (_type_, optional): Transforms (train/val)
                applied to images when loading. Defaults to None.
            synth_val_size (int | None, optional): Number of synthetic
                images used for validation. Defaults to None.
            validation (bool, optional): Whether to select
                just the validation subjets. Defaults to False.
            include_all (bool, optional): Whether to include all subjects
                (train + val) from a split. Defaults to False.
            rescale_res (float, optional): Resolution to rescale
                all images to. Defaults to None, no rescaling.
            generator_params (dict | str | None, optional): Path to yaml
                or a dictionary with the generator parameters. Defaults to None.
            min_subclasses (int, optional): Min number of subclasses to randomly
                select from for each meta-label. Defaults to 1.
            max_subclasses (int, optional): Max number of subclasses to randomly
                select from for each meta-label. Defaults to 6.
            seed_path (str, optional): Path to the seeds folder with
                subclasses_N/subject/anat/mlabel_M division. Defaults to None.
            mlab_subclasses (None | dict, optional): Dictionary containing
                {meta_lables(int): subclasses_to_use(int)}.
                Defaults to None which will randomly select # of subclasses
                for each meta label, given min/max_subclasses .
            segm_path (str, optional): Path to a bids root
                from which to look for GT segmentations. Defaults to None.
        """
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
        self.loader = LoadImage(reader='NibabelReader')
        self.filler = SignalFillEmptyd(keys=['image', 'label'],
                                       replacement=0,)
        self.scaler = ScaleIntensityd(keys=['image'],
                                      minv=0, maxv=1,)
        self.validation = validation
        self.include_all = include_all
        self.segm_path = Path(segm_path)

        self.min_subclasses = min_subclasses
        self.max_subclasses = max_subclasses
        self.generator_params = generator_params
        self.mlab_subclasses = mlab_subclasses
        if seed_path is not None:
            self.seed_root = Path(seed_path)
        else:
            self.seed_root = self.bids_path/f'derivatives/seeds'
        if not self.seed_root.exists():
            raise FileNotFoundError(f'Provided seed path {self.seed_root} does not exist.')
        self.__load_data()

        self.generator = SynthGenerator(self.generator_params,
                                             aff=self.segm_paths[0].affine,
                                             header=self.segm_paths[0].header,
                                             device='cuda') # TODO: CHECK OF SPECIFYING IT LIKE THIS IS OK

    def __load_data(self):
        """Pre-loads and parses paths and images for all segmentations and seeds.
        """
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

        self.seed_paths = {x: defaultdict(dict) for x in self.subjects}
        self.segm_paths = [nib.load(str(list(self.segm_path.glob(f'{x}/anat/{x}*_dseg.nii.gz'))[0])) for x in self.subjects]

        # load seeds corresponding to subjects
        for seed in range(self.min_subclasses, self.max_subclasses + 1):
            seed_paths = self.seed_root/f'subclasses_{seed}'
            seed_subjects = [x for x in seed_paths.glob('sub-*') if x.name in self.subjects]
            for subj in seed_subjects:
                for mlab in range(1, 5):
                    self.seed_paths[subj.name][seed][mlab] = nib.load(str(list(subj.glob(f'anat/*_mlabel_{mlab}.nii.gz'))[0]))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        # 1. Sample random meta label subclasses number, if needed
        if self.mlab_subclasses is None:
            mlab_subclasses = {}
            for mlab in range(1, 5):
                mlab_subclasses[mlab] = np.random.randint(self.min_subclasses,
                                                          self.max_subclasses + 1)
        else:
            mlab_subclasses = self.mlab_subclasses
        # 2. Crease a random seed map with the selected complexity
        seeds = []
        for ml in range(1, 5):
            seeds.append(self.seed_paths[self.subjects[idx]][mlab_subclasses[ml]][ml])

        seeds = np.array([x.get_fdata('unchanged') for x in seeds])

        with torch.inference_mode():
            seeds = torch.tensor(np.sum(seeds, axis=0),
                                dtype=torch.long, device='cuda') # TODO: REMOVE DEVICE
            segmentation = self.segm_paths[idx]

            # define extras (segmentations to be deformed together with seeds)
            extras = [segmentation]  # + [seeds] if needed for debug

            affine = segmentation.affine
            header = segmentation.header
            shape = segmentation.shape

            subj_meta, samples = self.generator.generate(self.subjects[idx],
                                                        shape,
                                                        affine,
                                                        header,
                                                        seeds,
                                                        extras)
            # for debugging
            subj_meta['mlab_subclasses'] = mlab_subclasses

            data = {'image': samples['synth_image'],
                    'label': samples['extras'][0].unsqueeze(0),
                    'name': self.subjects[idx]}

            # fill nans with zeros
            data = self.filler(data)

            data = self.scaler(data)

            data['label'] = data['label'].long()
            # return data
            return data['image'].cpu(),  data['label'].cpu(), self.subjects[idx]