import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from hydra import compose, initialize

from src.data.components.synth_gen import SynthGenerator

# use gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the test image and segmentation
test_img = nib.load(Path("./data/sub-sta30/anat/sub-sta30_rec-irtk_T2w.nii.gz").absolute())
test_segm = nib.load(Path("./data/sub-sta30/anat/sub-sta30_rec-irtk_T2w_dseg.nii.gz").absolute())

# you need to know the affine and header of the 
# segmentation image to generate synthetic data
# you can do it like this for example
header = test_img.header
affine = test_img.affine

# convert the images to tensors
test_img = test_img.get_fdata()
test_img = torch.tensor(test_img).unsqueeze(0).unsqueeze(0).float().to(device)
test_segm = test_segm.get_fdata()
test_segm = torch.tensor(test_segm).unsqueeze(0).long().to(device)

# load the generator configuration
with initialize(version_base=None, config_path="../../configs/data",):
    cfg = compose(config_name="fetsynthgen", )

# set the number of subclasses to generate
min_subclasses = cfg.min_subclasses
max_subclasses = cfg.max_subclasses

# define the seed paths
seedspath = Path("./data/derivatives/seeds").absolute()
seed_paths = {mlab: [seedspath/f'subclasses_{x}/sub-sta30/anat/sub-sta30_rec-irtk_T2w_dseg_mlabel_{mlab}.nii.gz' for x in range(1, 5)] for mlab in range(1, 5)}

# Sample random meta label subclasses number
mlab_subclasses = {}
for mlab in range(1, 5):
    mlab_subclasses[mlab] = np.random.randint(min_subclasses,
                                              max_subclasses)
# Crease a random seed map with the selected complexity
seeds = []
for ml in range(1, 5):
    seeds.append(seed_paths[ml][mlab_subclasses[ml]])

seeds = np.array([nib.loadsave.load(x).get_fdata('unchanged') for x in seeds])
seeds = torch.tensor(np.sum(seeds, axis=0),
                     dtype=torch.long,
                     device=device)

synth_gen = SynthGenerator(args=cfg.generator_params,
                           aff=affine,
                           header=header,
                           device=device)  # or 'cpu', but it will be slower

subject_meta, samples = synth_gen.generate(name='test',
                                           Gshp=(256, 256, 256),
                                           aff=affine,
                                           header=header,
                                           Gimg=seeds,  # use seeds to generate the synthetic data
                                           extras=[test_segm])

# plot the input image and segmentation
data = {'image': test_img.squeeze(0).squeeze(0),
        'label': test_segm.squeeze(0)}
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for i, (k, v) in enumerate(data.items()):
    ax[i].imshow(v[:, 125, :].cpu().numpy(), cmap='gray')
    ax[i].set_title(k)
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# plot the generated image and segmentation
data = {'image': samples['synth_image'].squeeze(0),
        'label': samples['extras'][0].squeeze(0).squeeze(0)}
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for i, (k, v) in enumerate(data.items()):
    ax[i].imshow(v[:, 125, :].cpu().numpy(), cmap='gray')
    ax[i].set_title(k)
    ax[i].axis('off')
plt.tight_layout()
plt.show()
