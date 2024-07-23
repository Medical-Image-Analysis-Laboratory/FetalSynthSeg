import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.components.feta_dataset import FeTADataset
from src.data.components.synth_dataset import SynthDataset
from monai import transforms

# 'fabian' mentions are spoilers for the journal paper, to be ignored for now

class FetaDataModuleOnlineSynth(pl.LightningDataModule):
    def __init__(self,
                 feta_path: str,
                 synth_path: str,
                 split_file: str,
                 train: str = 'feta',
                 batch_size: int = 1,
                 num_workers: int = 1,
                 train_split: str | None = None,
                 valid_split: str | None = None,
                 fabian_path: str = '',
                 validate: str = 'feta',
                 synth_val_size: int | None = None,
                 train_augm: transforms.Compose | None = None,
                 val_augm: transforms.Compose | None = None,
                 generator_params=None,
                 min_subclasses=None,
                 max_subclasses=None,
                 seed_path=None,
                 mlab_subclasses=None,
                 segm_path=None,
                 rescale_res: float = 0.5,
                 pin_memory: bool = False):
        super().__init__()
        assert train in ['feta', 'synth', 'fabian']
        assert validate in ['feta', 'synth', 'feta+synth']
        self.train = train
        self.split_file = split_file
        self.feta_path = feta_path
        self.train_split = train_split
        self.valid_split = valid_split
        self.synth_path = synth_path
        self.batch_size = batch_size
        self.validate = validate
        self.rescale_res = rescale_res
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.synth_val_size = synth_val_size
        self.fabian_path = fabian_path
        self.train_transform = train_augm
        self.val_transform = val_augm

        self.generator_params = generator_params
        self.min_subclasses = min_subclasses
        self.max_subclasses = max_subclasses
        self.seed_path = seed_path
        self.mlab_subclasses = mlab_subclasses
        self.segm_path = segm_path

        train_ds = None
        if self.train == 'synth':
            train_ds = SynthDataset
            train_dir = self.synth_path
            self.train_dataset = train_ds(train_dir,
                                          transforms=self.train_transform,
                                          split=self.train_split,
                                          split_file=self.split_file,
                                          rescale_res=self.rescale_res,
                                          validation=False,
                                          # online generator specific params
                                          generator_params=self.generator_params,
                                          min_subclasses=self.min_subclasses,
                                          max_subclasses=self.max_subclasses,
                                          seed_path=self.seed_path,
                                          mlab_subclasses=self.mlab_subclasses,
                                          segm_path=self.segm_path)

        if self.train == 'feta':
            train_ds = FeTADataset
            train_dir = self.feta_path
            self.train_dataset = train_ds(train_dir,
                                          transforms=self.train_transform,
                                          split=self.train_split,
                                          split_file=self.split_file,
                                          rescale_res=self.rescale_res,
                                          validation=False)

        valid_ds = SynthDataset if self.validate == 'synth' else FeTADataset
        valid_dir = self.synth_path if self.validate == 'synth' else self.feta_path
        if self.validate != 'feta+synth':
            # use separate validation dataset
            self.val_datasets = []
            for spl in self.valid_split:
                if self.validate == 'synth':
                    self.val_datasets.append(valid_ds(valid_dir,
                                                      transforms=self.val_transform,
                                                      synth_val_size=self.synth_val_size,
                                                      split_file=self.split_file,
                                                      split=self.valid_split,
                                                      validation=True,
                                                      # online gen parameters                                                # online generator specific params
                                                      generator_params=self.generator_params,
                                                      min_subclasses=self.min_subclasses,
                                                      max_subclasses=self.max_subclasses,
                                                      seed_path=self.seed_path,
                                                      mlab_subclasses=self.mlab_subclasses,
                                                      segm_path=self.segm_path)),
                else:
                    self.val_datasets.append(valid_ds(valid_dir,
                                                      transforms=self.val_transform,
                                                      synth_val_size=self.synth_val_size,
                                                      split=[spl],
                                                      split_file=self.split_file,
                                                      rescale_res=self.rescale_res,
                                                      validation=spl in self.train_split))

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=False,
                          shuffle=True)

    def val_dataloader(self):
        if self.validate != 'feta+synth':
            dl = [DataLoader(ds, batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             pin_memory=self.pin_memory,
                             persistent_workers=False,
                             shuffle=False) for ds in self.val_datasets]
            dl = dl[0] if len(dl) == 1 else dl
            return dl
        else:
            return [DataLoader(dataset=FeTADataset(feta_path=self.feta_path,
                                                      transforms=self.val_transform,
                                                      synth_val_size=self.synth_val_size,
                                                      split_file=self.split_file,
                                                      split=self.valid_split,
                                                      rescale_res=self.rescale_res,
                                                      validation=True,
                                                      ),
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               pin_memory=self.pin_memory,
                               persistent_workers=False,

                               shuffle=False),
                    DataLoader(dataset=SynthDataset(feta_path=self.synth_path,
                                                         transforms=self.val_transform,
                                                         synth_val_size=self.synth_val_size,
                                                         split_file=self.split_file,
                                                         split=self.valid_split,
                                                         validation=True,
                                                         # online gen parameters                                                # online generator specific params
                                                         generator_params=self.generator_params,
                                                         min_subclasses=self.min_subclasses,
                                                         max_subclasses=self.max_subclasses,
                                                         seed_path=self.seed_path,
                                                         mlab_subclasses=self.mlab_subclasses,
                                                         segm_path=self.segm_path),
                               batch_size=self.batch_size,
                               persistent_workers=False,

                               num_workers=self.num_workers,
                               pin_memory=self.pin_memory,
                               shuffle=False,
                               )]

    # retrieves the test dataloaders containg the validation data
    def test_dataloader(self, train_split: str | None = None,
                        synth: bool = False, synth_val_size: int | None = 40):

        splits = ['KISPI - rec-mial', 'KISPI - rec-irtk', 'CHUV - rec-mial']
        # select only validation cases from the train split
        val_ds = []
        # select all other splits for validation and all cases from them
        for spl in splits:
            if spl != train_split:
                val_ds.append(FeTADataset(feta_path=self.feta_path,
                                             transforms=self.val_transform,
                                             synth_val_size=None,
                                             split_file=self.split_file,
                                             split=[spl],
                                             validation=False,
                                             rescale_res=self.rescale_res,

                                             include_all=True,))
            else:
                val_ds.append(FeTADataset(feta_path=self.feta_path,
                                             transforms=self.val_transform,
                                             synth_val_size=None,
                                             split_file=self.split_file,
                                             split=[train_split],
                                             validation=True,
                                             rescale_res=self.rescale_res,
                                             include_all=False,))

        if synth:
            # select only validation cases from the train split
            synth_val_ds = [SynthDataset(self.synth_path,
                                              transforms=self.val_transform,
                                              synth_val_size=synth_val_size,
                                              split_file=self.split_file,
                                              split=[train_split],
                                              validation=True,
                                              include_all=False,
                                              # online gen parameters                                                # online generator specific params
                                              generator_params=self.generator_params,
                                              min_subclasses=self.min_subclasses,
                                              max_subclasses=self.max_subclasses,
                                              seed_path=self.seed_path,
                                              mlab_subclasses=self.mlab_subclasses,
                                              segm_path=self.segm_path),
                                              ]
            # select all other splits for validation and all cases from them
            for spl in splits:
                if spl != train_split:
                    synth_val_ds.append(SynthDataset(self.synth_path,
                                                          transforms=self.val_transform,
                                                          synth_val_size=synth_val_size,
                                                          split_file=self.split_file,
                                                          split=[spl],
                                                          validation=True,
                                                          include_all=False,
                                                          # online gen parameters                                                # online generator specific params
                                                          generator_params=self.generator_params,
                                                          min_subclasses=self.min_subclasses,
                                                          max_subclasses=self.max_subclasses,
                                                          seed_path=self.seed_path,
                                                          mlab_subclasses=self.mlab_subclasses,
                                                          segm_path=self.segm_path))
            val_ds = val_ds + synth_val_ds

        dls = [DataLoader(ds, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False) for ds in val_ds]
        return dls

if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../../configs/data",):
        cfg = compose(config_name="feta_onlinesynth", )
        print(OmegaConf.to_yaml(cfg))
        # instantiate datamodule
        dm = instantiate(cfg)

        train_ds = dm.train_dataloader().dataset
        valid_df = dm.val_dataloader().dataset

        print(f"Train dataset: {len(train_ds)}")
        print(f"Validation dataset: {len(valid_df)}")

        gen_data = train_ds[1]
        trimg = gen_data[0]
        valimg = valid_df[1][0]
        trsegm = gen_data[1]
        valsegm = valid_df[1][1]
        print(f'TNan: {trimg.isnan().sum()}')
        print(f'VNan: {valimg.isnan().sum()}')

        # print(f'Train image path: {train_ds.img_paths[0]}')
        # print(f'Validation image path: {valid_df.img_paths[0]}')

        print(f"Train image shape: {trimg[0].shape}")
        print(f"Validation image shape: {valimg[0].shape}")
        print(f"Train segm shape: {trsegm.shape}")
        print(f"Validation sehm shape: {valsegm.shape}")
        print(f'Train Max {trimg.max()} type {type(trimg)} min {trimg.min()}')
        print(f'Val Max {valimg.max()} type {type(valimg)} min {valimg.min()}')
        # plot images in the sample plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(trimg[0, 128, :, :].cpu(), cmap='gray')
        ax[0].set_title('Train image')
        ax[1].imshow(valimg[0, 128, :, :].cpu(), cmap='gray')
        ax[1].set_title('Validation image')
        plt.tight_layout()
        plt.show()


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(trsegm[0, 128, :, :].cpu(), cmap='gray')
        ax[0].set_title('Train image')
        ax[1].imshow(valsegm[0, 128, :, :].cpu(), cmap='gray')
        ax[1].set_title('Validation image')
        plt.tight_layout()
        plt.show()