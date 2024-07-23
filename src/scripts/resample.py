"""First resamples the images to {res} isotropic resolution and then crop-pads them to {target_size}."""

from tqdm import tqdm
from pathlib import Path

import monai

# RESAMPLING RESOULTION
res = 0.5

# INPUT PATH (BIDS)
bids_path = Path('/bids/path')

# INPUT TARGET SIZE
target_size = (256, 256, 256)

# SAVE PATH (RELATIVE TO THE BIDS ROOT)
out_path = bids_path/'derivatives/resampled05'

# defining transforms
loader = monai.transforms.LoadImaged(keys=['image', 'label'])
resampe_transform = monai.transforms.Spacingd(keys=['image', 'label'],
                                              pixdim=(res, res, res),
                                              mode=('bilinear', 'nearest'),
                                              allow_missing_keys=True,)

cropper = monai.transforms.CenterSpatialCropd(keys=['image', 'label'],
                                              allow_missing_keys=True,
                                              roi_size=target_size)
orientation = monai.transforms.Orientationd(keys=['image', 'label'],
                                            axcodes='RAS',
                                            allow_missing_keys=True)

padder = monai.transforms.SpatialPadd(keys=['image', 'label'],
                                      spatial_size=(256, 256, 256),
                                      mode='constant',
                                      allow_missing_keys=True)


subjects = list(bids_path.glob('sub-*'))

print(f'Found {len(subjects)} in {bids_path}')

for sub in tqdm(subjects):
    saver = monai.transforms.SaveImaged(keys=['image', 'label'],
                                        output_dir=out_path/f'{sub.name}/anat/',
                                        output_postfix='',
                                        resample=False,
                                        separate_folder=False,
                                        print_log=False,
                                        allow_missing_keys=True,
                                        mode='nearest')

    try:
        imgs = list(sub.glob('**/*_T2w.nii.gz'))[0]  # CHANGE THE PATTERN TO MATCH YOUR IMAGES
        label = list(sub.glob('**/*_dseg.nii.gz'))[0]  # CHANGE THE PATTERN TO MATCH YOUR LABELS
        data = loader({'image': str(imgs), 'label': str(label)})
        data['image'] = data['image'].unsqueeze(0)
        data['label'] = data['label'].unsqueeze(0)
        data['label'] = data['label'].squeeze(-1)
        data = resampe_transform(data)
        data = orientation(data)
        data = cropper(data)
        data = padder(data)
        saver(data)
    except Exception as e:
        print(f'Error processing {sub} due to {e}')
        continue