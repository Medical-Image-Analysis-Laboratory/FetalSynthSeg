<div align="center">

# [FetalSynthSeg](https://arxiv.org/abs/2403.15103)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
</div>


This repository contains the code for the paper ["Improving cross-domain brain tissue segmentation in fetal MRI with synthetic data"](https://arxiv.org/abs/2403.15103)  by Zalevskyi et al.. The paper was accepted for  [MICCAI 2024 conference](https://conferences.miccai.org/2024/en/default.asp).

## Introduction
FetalSynthSeg is a **domain randomization** method designed to enhance the **segmentation of fetal brain MRI data**, **especially in the face of significant domain shifts**. This technique leverages synthetic data, inspired by SynthSeg [1], to train models capable of handling the high variability found in clinical data, which often includes different acquisition parameters, MRI field strengths, and super-resolution reconstruction (SR) algorithms. We train the segmentation models using synthetic images generated based on the framework depicted in the Figure 1.

<div align="center">

<img src="markdown_assets/FSS_recap.png" width="800">
<p>Figure 1. Synthetic image generation framework.</p>

</div>

**In this repository we provide the code necessary to run the synthetic data generation framework as well as models and their weights used in the paper.**

## Docker 🐳 
We provide a Docker image `vzalevskyi/fetalsynthseg:v1` to run the inference code in a containerized environment. 

See the [vzalevskyi/fetalsynthseg
](https://hub.docker.com/r/vzalevskyi/fetalsynthseg) for more details.



## Requirements and Installation
The code was developed with `Python 3.10.13` and `PyTorch 2.1.2`. and tested on `Ubuntu 20.04.3 LTS`. To install the required packages, run the following command:

```bash
# create a new conda environment with python 3.10.13
conda create -n FetalSynthSeg python=3.10.13
conda activate FetalSynthSeg

# install the required packages
pip install -r requirements.txt
```

It is optimized to work with super-resolution reconstructed fetal MRI images (GA 18-36) (preferably skull stripped).

## Usage
There are 3 main components in this repository:

* **Synthetic data generation** Based on the Pytorch re-implementations of SynthSeg's generator from the [Brain-ID repository](https://github.com/peirong26/Brain-ID/tree/main?tab=readme-ov-file)  [2]. The generator code is located in the [`src/data/components/synth_gen.py`](src/data/components/synth_gen.py). See [`src/test/visualize_synth_generation.py`](src/test/visualize_synth_generation.py) for usage example. We use `hydra` and `omegaconfig` to track configuration of the runs and the generator, see [`configs/data/fetsynthgen.yaml`](configs/data/fetsynthgen.yaml) for an example of the synthetic generator configuration used in our experiments.

* **Data** While we do not provide access to all fetal MRI data or segmentations used in the paper, it is possible to request access to the KISPI dataset on the [Fetal Tissue Annotation Challenge - FeTA MICCAI](https://www.synapse.org/Synapse:syn25649159/wiki/610007) Synapse page [3].

    We do however provide an example of a fetal MRI image and its corresponding segmentation in the [`data/`](data/) folder as well as the *'seeds'* used to generated synthetic images based on this data, following the framework in Figure 1. The seeds are located in the [`data/seeds`](data/seeds) folder. Image and segmentation borrowed from the [IMAGINE Fetal T2-weighted MRI Atlas by Gholipour et al.](https://doi.org/10.7910/DVN/WE9JVR) [4]. Use [`src/scripts/generate_seeds.py`](src/scripts/generate_seeds.py) to create seeds for your own data.

* **Pre-trained models and inference** [`src/models/UnetModule.py`](src/models/UnetModule.py) contains the code of the U-Net model used in the paper. See [`weights/`](weights/) folder to learn how to download our pre-trained weights.

    To run the inference, use the [`src/scripts/inference.py`](src/scripts/inference.py) script. It will load the model and weights and segment given image.

    ```
    usage: inference.py [-h] [--chkp_path CHKP_PATH] --input INPUT [INPUT ...] --output OUTPUT [--gpu]

    options:
    -h, --help                  show this help message and exit
    --chkp_path CHKP_PATH       Path to the checkpoint (.ckpt file of the model).
    --input INPUT [INPUT ...]   Path to the input(s) for the model to predict. Can be a single file path or a list of file paths
                                (space separated) (.with nii.gz extension) or
                                a BIDS directory. If a directory is given, it is expected to be in the BIDS format, with the images in "sub-*/**/*_T2w.nii.gz"
    
    --output OUTPUT       Output path for the prediction(s). Should match the input format.
                          If a list of files is given will save all the files     into a given folder.
    
    --gpu                 Use GPU for prediction
  ```

## Citation
Zalevskyi, V., Sanchez, T., Roulet, M., Verddera, Jordina Aviles, Hutter, J., Kebiri, H., & Cuadra, M. B. (2024). Improving cross-domain brain tissue segmentation in fetal MRI with synthetic data. ArXiv.org. https://arxiv.org/abs/2403.15103

if you have any questions, please contact [vladyslav.zalevskyi@unil.ch](mailto:vladyslav.zalevskyi@unil.ch) or open an issue in this repository.

## Acknowledgements
This research was funded by the Swiss National Science Foun-
dation (215641), ERA-NET Neuron MULTI-FACT project (SNSF 31NE30_203977),
UKRI FLF (MR/T018119/1) and DFG Heisenberg funding (502024488); We acknowl-
edge the Leenaards and Jeantet Foundations as well as CIBM Center for Biomedical
Imaging, a Swiss research center of excellence founded and supported by CHUV, UNIL,
EPFL, UNIGE and HUG
## References
1. SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias
Medical Image Analysis (2023). https://doi.org/10.1016/j.media.2023.102789

2. Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging. Liu, P., Puonti, O., Hu, X., Alexander, D. C., & Iglesias, J. E. (2023).   ArXiv.org. https://arxiv.org/abs/2311.16914

3. An automatic multi-tissue human fetal brain segmentation benchmark using the Fetal Tissue Annotation Dataset. Payette, K., de Dumast, P., Kebiri, H. et al.. Sci Data 8, 167 (2021). https://doi.org/10.1038/s41597-021-00946-3

4. Gholipour, Ali; Velasco-Annis, Clemente; Rollins, Caitlin K.; Vasung, Lana; Ouaalam, Abdelhakim; Ortinau, Cynthia; Akhondi-Asl, Alireza; Clancy, Sean; Yang, Edward; Estroff, Judy; Warfield, Simon K., 2023, "IMAGINE Fetal T2-weighted MRI Atlas", https://doi.org/10.7910/DVN/WE9JVR, Harvard Dataverse, V1
