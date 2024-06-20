import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import os
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import hydra
import monai
from omegaconf import OmegaConf


from src.data.loader import ImageLoader
from src.models.UnetModule import Unet


parser = argparse.ArgumentParser()
parser.add_argument('--chkp_path', type=str, default='weights/KISPI-all_fss.ckpt',
                    help='Path to the checkpoint (.ckpt file of the model).', required=False)

parser.add_argument('--input', type=str, required=True, 
                    help='Path to the input(s) for the model to predict. Can be a single file path or a list of file paths (space separated) (.with nii.gz extension) or a BIDS directory.\
                        If a directory is given, it is expected to be in the BIDS format,\
                            with the images in "sub-*/**/*_T2w.nii.gz"',
                    nargs='+', default=[])

parser.add_argument('--output', type=str, required=True,
                    help='Output path for the prediction(s). Should match the input format. If a list of files is given will save all the files into a given folder.',
                    )
parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')
args = parser.parse_args()


def main():
    # load configs
    augmentation_config = OmegaConf.load(Path('./configs/data/augmentations.yaml').absolute())
    model_config = OmegaConf.load(Path('./configs/model/Unet.yaml').absolute())

    unet = hydra.utils.instantiate(model_config['net'])
    device = torch.device("cuda" if args.gpu else "cpu")
    model = Unet.load_from_checkpoint(args.chkp_path, net=unet).to(device)
    model.eval()
    # set pre-processing parameters
    rescale_res = 0.5
    transforms = augmentation_config['val_augm']
    transforms = hydra.utils.instantiate(transforms)
    input_path = Path(args.input[0]) if len(args.input) == 1 else [Path(p) for p in args.input]
    if isinstance(input_path, list):
        for img in input_path:
            outpath = Path(args.output) / img.name
            print(f'Processing {img}')
            print(f'Output will be saved at {outpath}')
            predict_image(rescale_res, transforms, device, model, args, outpath, img)
            os.chmod(outpath, 0o777)

    elif isinstance(input_path, Path) and '.nii.gz' in input_path.name:
        outpath = Path(args.output)
        print(f'Processing {input_path}')
        print(f'Output will be saved at {outpath}')
        predict_image(rescale_res, transforms, device, model, args, outpath, input_path)
        os.chmod(outpath, 0o777)

    elif isinstance(input_path, Path) and input_path.is_dir():
        all_subjs = list(input_path.glob('sub-*'))
        print(f'Processing all {len(all_subjs)} subjects from  {input_path}')
        print(f'Output will be saved in the same structure as the input at {args.output}')
        for subj in tqdm(all_subjs):
            for img in subj.glob('*/**/*_T2w.nii.gz'):
                outpath = Path(args.output) / img.relative_to(input_path)
                outpath.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
                outpath = outpath.with_name(outpath.stem + '_pred.nii.gz')
                predict_image(rescale_res, transforms, device, model, args, outpath, img)
                os.chmod(outpath, 0o777)


def predict_image(rescale_res, transforms, device, model, args, outpath, input_path):
    # predict and save
    loader = ImageLoader(rescale_res=rescale_res, transforms=transforms)
    data = loader.load_image(input_path)
    data['image'] = data['image'].unsqueeze(0)
    data['label'] = data['label'].unsqueeze(0)
    with torch.no_grad():
        label = data['label'].to(device)
        img = data['image'].to(device)
        pred = model(img)
        # apply argmax to get the final segmentation
        pred_lab = pred.argmax(1)
        # change interpolation to nearest for label data
        pred_lab = monai.data.MetaTensor(x=pred_lab.as_tensor(),
                                         meta=label.meta,
                                         applied_operations=label.applied_operations)
        loader.save_image(pred_lab, outpath, img_path=input_path)


if __name__ == '__main__':
    main()
