"""Run this scripts to generate label maps seeds
for the FETA challenge dataset segmentations.

Based on the tissue-type mapping from tissue_map
it fuses similar classes from the segmentation into the same label.

Then they are split into N clusters using EM clustering. (MAX_SUBCLUSTS)

All non-zero voxels from the image that are background in the segmentation
are clustered into 4th clusters (MAX_BACK_SUCLUST defines how mby subclasses
to simulate for the background tissue).
"""

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import monai
import torch

# BIDS fodler with the segmentations and images for seeds generation
bids_path = Path('./data').absolute()
# Which pairs to use
input_path = bids_path
# SAVE PATH
out_path = bids_path/'derivatives/seeds'

# MAX_SUBCLASSES defines how many subclasses to simulate for each tissue type
MAX_SUBCLUSTS = 10

# defining meta lables for the segmentation
tissue_map = {'CSF': [1, 4], # meta_label_name: [segmentation_labels]
              'WM': [3, 5, 7],
              'GM': [2, 6]}

# mapping fetal labels to meta labels
feta2meta = {1: 1, 4: 1,
             2: 2, 6: 2,
             5: 3, 7: 3, 3: 3} # skull is 4

# defining transforms
loader = monai.transforms.LoadImaged(keys=['image', 'label'])

subjects = list(input_path.glob('sub-*'))

print(f'Found {len(subjects)} in {bids_path}')


def subsplit_label(img, segm, label2assign=10, n_clusters=3):
    img_voxels = img[segm > 0]
    # cluster non-zero image voxels that are zero in the mask
    brain_backg = segm*0

    clust = GaussianMixture(n_components=n_clusters, n_init=5,
                            init_params='k-means++').fit_predict(
        img_voxels.reshape(-1, 1))
    clust = torch.tensor(clust).long()
    brain_backg[segm > 0] = clust + label2assign # clusters are from 0 to n_clusters-1
    return brain_backg


def split_lables(image, segmentation, subclasses):
    # fuse feta labels into meta labels
    meta_segm = segmentation*0
    for fetalab, metalab in feta2meta.items():
        meta_segm[segmentation == fetalab] = metalab

    # set skull as class 4
    meta_segm[(segmentation == 0) & (image != 0)] = 4
    sublclasses = {} # dictionary to store the subsegmentations
    # in a format { number_of_subclasses: {meta_labels: subclasses_mask} }
    if subclasses == 1:
        sublclasses[subclasses] = {x: (meta_segm == x)*x*10 for x in range(1, 5, 1)}
        return sublclasses
    else:
        sublclasses[subclasses] = {}
        for metalabel in range(1, 5):
            mlabel_mask = (meta_segm == metalabel)
            split_segm = subsplit_label(image,
                                        mlabel_mask,
                                        label2assign=10*metalabel,
                                        n_clusters=subclasses)
            sublclasses[subclasses][metalabel] = split_segm
        return sublclasses


for sub in tqdm(subjects):
    print(f'Processing {sub}')
    for subclasses in range(1, MAX_SUBCLUSTS):
        imgs = list(sub.glob('**/*_T2w.nii.gz'))[0]
        label = list(sub.glob('**/*_dseg.nii.gz'))[0]
        data = loader({'image': str(imgs), 'label': str(label)})
        data['image'] = data['image'].unsqueeze(0)
        data['label'] = data['label'].unsqueeze(0)

        subclasses_splits = split_lables(data['image'],
                                         data['label'],
                                         subclasses)
        for n_subclasses, subsegms in subclasses_splits.items():
            for mlabel, subsegm in subsegms.items():

                saver = monai.transforms.SaveImaged(keys=['image', 'label'],
                                                    output_dir=out_path/f'subclasses_{n_subclasses}/{sub.name}/anat/',
                                                    output_postfix=f'mlabel_{mlabel}',
                                                    resample=False,
                                                    separate_folder=False,
                                                    print_log=False,
                                                    allow_missing_keys=True,
                                                    mode='nearest')

                subclas_data = {'label': subsegm}
                saver(subclas_data)
