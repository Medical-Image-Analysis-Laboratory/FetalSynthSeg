"""Adopted from https://github.com/peirong26/Brain-ID/tree/main
Please refer to the original code for more details.
"""

import os

import torch
import numpy as np
import nibabel as nib
from scipy.io.matlab import loadmat


import src.data.components.interpol as interpol
from src.data.components.brainid_utils import (fast_3D_interp_torch, myzoom_torch,
                                               make_affine_matrix, gaussian_blur_3d)

class SynthGenerator:
    """Generator for synthetic data
    Use SynthGenerator.generate() to synthesize new images
    """

    def __init__(self, args, aff, header, device='cuda'):
        """
        Args:
            args (omegaconf.dictconfig.DictConfig): Configuration of the generator.
                See Brain-ID paper for details description of parameters.
            aff (np.ndarray): Affine matrix of the seed image used for the generation.
                (nib_image.affine). (4, 4) array of floats
            header (nibabel.nifti1.Nifti1Header): Header of the seed image used for the generation.
                (nib_image.header)
            device (str, optional): Where to generate the image ['cpu', 'cuda']. Defaults to 'cuda'.
                GPU generation with 'cuda' ~10x faster.
        """
        self.args = args
        self.device = device
        self.task = args.task

        # base generation parameters
        self.seed_lables = args.base_generator.seed_lables
        self.generation_classes = args.base_generator.generation_classes
        self.output_labels = args.base_generator.output_labels

        self.n_steps_svf_integration = args.base_generator.n_steps_svf_integration

        self.nonlinear_transform = args.base_generator.nonlinear_transform
        self.deform_one_hots = args.base_generator.deform_one_hots
        self.integrate_deformation_fields = args.base_generator.integrate_deformation_fields
        self.produce_surfaces = args.base_generator.produce_surfaces
        self.bspline_zooming = args.base_generator.bspline_zooming

        self.size = args.base_generator.size
        self.max_rotation = args.base_generator.max_rotation
        self.max_shear = args.base_generator.max_shear
        self.max_scaling = args.base_generator.max_scaling
        self.nonlin_scale_min = args.base_generator.nonlin_scale_min
        self.nonlin_scale_max = args.base_generator.nonlin_scale_max
        self.nonlin_std_max = args.base_generator.nonlin_std_max
        self.bf_scale_min = args.base_generator.bf_scale_min
        self.bf_scale_max = args.base_generator.bf_scale_max
        self.bf_std_min = args.base_generator.bf_std_min
        self.bf_std_max = args.base_generator.bf_std_max
        self.bag_scale_min = args.base_generator.bag_scale_min
        self.bag_scale_max = args.base_generator.bag_scale_max
        self.gamma_std = args.base_generator.gamma_std
        self.noise_std_min = args.base_generator.noise_std_min
        self.noise_std_max = args.base_generator.noise_std_max
        self.preserve_resol = args.base_generator.preserve_resol

        self.min_resampling_iso_res = args.base_generator.min_resampling_iso_res
        self.max_resampling_iso_res = args.base_generator.max_resampling_iso_res

        self.exvixo_prob = args.base_generator.exvixo_prob
        self.photo_prob = args.base_generator.photo_prob
        self.bag_prob = args.base_generator.bag_prob
        self.pv = args.base_generator.pv

        # load label maps if str (experting .npy files)
        if isinstance(self.seed_lables, str):
            self.seed_lables = np.load(self.seed_lables)
        if isinstance(self.generation_classes, str):
            self.generation_classes = np.load(self.generation_classes)
        if isinstance(self.output_labels, str):
            self.output_labels = np.load(self.output_labels)

        # Get resolution of training data
        self.res_training_data = np.sqrt(np.sum(aff[:-1, :-1], axis=0))
        self.orig_resol = header['pixdim'][1:4]

        # prepare grid
        # print('Preparing grid...')
        with torch.no_grad():

            xx, yy, zz = np.meshgrid(range(self.size[0]),
                                    range(self.size[1]), range(self.size[2]),
                                    sparse=False, indexing='ij')
            self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
            self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
            self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
            self.c = torch.tensor((np.array(self.size) - 1) / 2,
                                dtype=torch.float, device=self.device)
            self.xc = self.xx - self.c[0]
            self.yc = self.yy - self.c[1]
            self.zc = self.zz - self.c[2]

            # Matrix for one-hot encoding (includes a lookup-table)
            n_labels = len(set(self.output_labels))
            self.lut = torch.zeros(1000, dtype=torch.long, device=self.device)
            for l in range(n_labels):
                self.lut[self.output_labels[l]] = l
            self.onehotmatrix = torch.eye(n_labels,
                                        dtype=torch.float, device=self.device)
            self.n_neutral_labels = n_labels
            nlat = int((n_labels - self.n_neutral_labels) / 2.0)
            self.vflip = np.concatenate([np.array(range(self.n_neutral_labels)),
                                        np.array(range(self.n_neutral_labels + nlat,
                                                            n_labels)),
                                        np.array(range(self.n_neutral_labels,
                                                            self.n_neutral_labels + nlat))])

        # print('BaseSynth Generator is ready!')

    def random_affine_transform(self, shp, max_rotation, max_shear, max_scaling,
                                random_shift = True):
        rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
        shears = (2 * max_shear * np.random.rand(3) - max_shear)
        scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
        # we divide distance maps by this, not perfect, but better than nothing
        scaling_factor_distances = np.prod(scalings) ** .33333333333
        A = torch.tensor(make_affine_matrix(rotations, shears, scalings),
                         dtype=torch.float, device=self.device)
        # sample center
        if random_shift:
            max_shift = (torch.tensor(np.array(shp[0:3]) - self.size,
                                      dtype=torch.float, device=self.device)) / 2
            max_shift[max_shift < 0] = 0
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2,
                              dtype=torch.float, device=self.device) + \
                                  (2 * (max_shift * torch.rand(3, dtype=float,
                                                               device=self.device)) - max_shift)
        else:
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2,
                              dtype=torch.float, device=self.device)
        return scaling_factor_distances, A, c2

    def random_nonlinear_transform(self, photo_mode, spac, nonlin_scale_min,
                                   nonlin_scale_max, nonlin_std_max):
        nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if self.integrate_deformation_fields: # NOTE: slow
            steplength = 1.0 / (2.0 ** self.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in range(self.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0],
                                             self.yy + Fsvf[:, :, :, 1],
                                             self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0],
                                                 self.yy + Fsvf_neg[:, :, :, 1],
                                                 self.zz + Fsvf_neg[:, :, :, 2], 'linear')
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg

    def generate_deformation(self, photo_mode, spac,
                             Gshp, random_shift=True):

        # sample affine deformation
        scaling_factor_distances, A, c2 = self.random_affine_transform(Gshp, self.max_rotation,
                                                                       self.max_shear,
                                                                       self.max_scaling,
                                                                       random_shift)

        # sample nonlinear deformation 
        if self.nonlinear_transform:
            F, _ = self.random_nonlinear_transform(photo_mode, spac,
                                                   self.nonlin_scale_min,
                                                   self.nonlin_scale_max,
                                                   self.nonlin_std_max) 
        else:
            F = None

        # deform the images 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(Gshp, A, c2, F)

        return scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2

    def deform_image(self, shp, A, c2, F):
        if F is not None:
            # deform the images (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
        else:
            xx1 = self.xc
            yy1 = self.yc
            zz1 = self.zc

        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]  
        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2

    def generate_surface(self, idx, Fneg, A, c2):
        filename = os.path.basename(self.names[idx])
        if filename.endswith('.nii.gz'):
            filename = filename[:-7] + '.mat'
        else:
            filename = filename[:-4] + '.mat'
        mat = loadmat(os.path.join(self.surface_dir, filename ))
        Vlw = torch.tensor(mat['Vlw'], dtype=torch.float, device=self.device)
        Flw = torch.tensor(mat['Flw'], dtype=torch.int, device=self.device)
        Vrw = torch.tensor(mat['Vrw'], dtype=torch.float, device=self.device)
        Frw = torch.tensor(mat['Frw'], dtype=torch.int, device=self.device)
        Vlp = torch.tensor(mat['Vlp'], dtype=torch.float, device=self.device)
        Flp = torch.tensor(mat['Flp'], dtype=torch.int, device=self.device)
        Vrp = torch.tensor(mat['Vrp'], dtype=torch.float, device=self.device)
        Frp = torch.tensor(mat['Frp'], dtype=torch.int, device=self.device)

        Ainv = torch.inverse(A)
        Vlw -= c2[None, :]
        Vlw = Vlw @ torch.transpose(Ainv, 0, 1)
        Vlw += fast_3D_interp_torch(Fneg, Vlw[:, 0] + self.c[0],
                                    Vlw[:, 1]+self.c[1],
                                    Vlw[:, 2] + self.c[2], 'linear')
        Vlw += self.c[None, :]
        Vrw -= c2[None, :]
        Vrw = Vrw @ torch.transpose(Ainv, 0, 1)
        Vrw += fast_3D_interp_torch(Fneg, Vrw[:, 0] + self.c[0],
                                    Vrw[:, 1]+self.c[1],
                                    Vrw[:, 2] + self.c[2], 'linear')
        Vrw += self.c[None, :]
        Vlp -= c2[None, :]
        Vlp = Vlp @ torch.transpose(Ainv, 0, 1)
        Vlp += fast_3D_interp_torch(Fneg, Vlp[:, 0] + self.c[0],
                                    Vlp[:, 1] + self.c[1],
                                    Vlp[:, 2] + self.c[2], 'linear')
        Vlp += self.c[None, :]
        Vrp -= c2[None, :]
        Vrp = Vrp @ torch.transpose(Ainv, 0, 1)
        Vrp += fast_3D_interp_torch(Fneg, Vrp[:, 0] + self.c[0],
                                    Vrp[:, 1] + self.c[1],
                                    Vrp[:, 2] + self.c[2], 'linear')
        Vrp += self.c[None, :]
        return Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp

    def read_data(self, img, loc_list,  extras=None):
        x1, x2, y1, y2, z1, z2 = loc_list

        # read the original segmentation map (used for GMM sampling)
        G = torch.squeeze(img)

        if extras is not None:
            for i, x in enumerate(extras):
                # extra = nib.load(x)
                if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
                    extras[i] = torch.squeeze(x[x1:x2, y1:y2, z1:z2])
                else:
                    extras[i] = torch.squeeze(torch.tensor(x.get_fdata()[x1:x2, y1:y2, z1:z2].astype(int),
                                              dtype=torch.int, device=self.device))

        return G, extras

    def process_sample(self, photo_mode, spac, thickness, resolution, flip, mus,
                       sigmas, G, loc_list, gamma_std, bf_scale_min, bf_scale_max,
                       bf_std_min, bf_std_max, noise_std_min, noise_std_max, extras=None):
        xx2, yy2, zz2 = loc_list
        # G - input seed map

        # obtain generation class map from the seed map
        # if generation_classes are the same as seed_lables
        # no need to relable
        if self.generation_classes == self.seed_lables:
            Gr = G
        else:
            Gr = torch.zeros_like(G)
            for i in range(len(self.seed_lables)):
                if i != 0:
                    Gr[G == self.seed_lables[i]] = self.generation_classes[i]

        Gr = torch.round(Gr).long()
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape, dtype=torch.float, device=self.device)

        SYN[SYN < 0] = 0

        SYN_def = fast_3D_interp_torch(SYN, xx2, yy2, zz2, 'linear')

        # interpolate the segmentation maps from extras
        if extras is not None and extras:
            for i, x in enumerate(extras):
                extras[i] = fast_3D_interp_torch(x, xx2, yy2, zz2, 'nearest')

        # Gamma transform
        gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]),
                             dtype=float, device=self.device)
        SYN_gamma = 300.0 * (SYN_def / 300.0) ** gamma

        # Bias field
        bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
        size_BF_small = np.round(bf_scale * np.array(self.size)).astype(int).tolist()

        if photo_mode:
            size_BF_small[1] = np.round(self.size[1]/spac).astype(int)
        BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1),
                               dtype=torch.float, device=self.device) * \
            torch.randn(size_BF_small, dtype=torch.float, device=self.device)
        BFlog = myzoom_torch(BFsmall, np.array(self.size) / size_BF_small)
        BF = torch.exp(BFlog)
        SYN_bf = SYN_gamma * BF

        # Model Resolution
        stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) /np.pi * thickness / self.res_training_data
        stds[thickness<=self.res_training_data] = 0.0 # no blur if thickness is equal to the resolution of the training data
        SYN_blur = gaussian_blur_3d(SYN_bf, stds, self.device)
        new_size = (np.array(self.size) * self.res_training_data / resolution).astype(int)

        factors = np.array(new_size) / np.array(self.size)
        delta = (1.0 - factors) / (2.0 * factors)
        vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
        vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
        vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
        II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
        II = torch.tensor(II, dtype=torch.float, device=self.device)
        JJ = torch.tensor(JJ, dtype=torch.float, device=self.device)
        KK = torch.tensor(KK, dtype=torch.float, device=self.device)

        SYN_small = fast_3D_interp_torch(SYN_blur, II, JJ, KK, 'linear') 
        noise_std = torch.tensor(noise_std_min + (noise_std_max - noise_std_min) * np.random.rand(1),
                                 dtype=torch.float, device=self.device)
        SYN_noisy = SYN_small + noise_std * torch.randn(SYN_small.shape,
                                                        dtype=torch.float, device=self.device)
        SYN_noisy[SYN_noisy < 0] = 0

        # Back to original resolution
        if self.bspline_zooming:
            SYN_resized = interpol.resize(SYN_noisy, shape=self.size, anchor='edge',
                                          interpolation=3, bound='dct2', prefilter=True) 
        else:
            SYN_resized = myzoom_torch(SYN_noisy, 1 / factors) 
        maxi = torch.max(SYN_resized)
        SYN_final = SYN_resized / maxi

        # Flip 50% of times
        if flip:
            SYN_final = torch.flip(SYN_final, [0]) 
            if extras is not None and extras:
                for i, x in enumerate(extras):
                    extras[i] = torch.flip(x, dims=[0]).cpu()

        # prepare for input
        SYN_final = SYN_final[None, ...] # add one channel dimension

        sample = {'synth_image': SYN_final.cpu(),
                  'extras': extras}

        return sample

    def get_contrast(self, photo_mode):
        # Sample Gaussian image
        mus = 25 + 200 * torch.rand(10000, dtype=torch.float, device=self.device)
        sigmas = 5 + 20 * torch.rand(10000, dtype=torch.float, device=self.device)
        if photo_mode or np.random.rand(1) < 0.5:  # set the background to zero every once in a while (or always in photo mode)
            mus[0] = 0
        return mus, sigmas

    def random_sampler(self): 
        if self.preserve_resol >= np.random.rand():
            resolution = self.orig_resol
            thickness = self.orig_resol
        else:
            resolution, thickness = self.iso_resolution_sample(self.min_resampling_iso_res,
                                                               self.max_resampling_iso_res)
        return resolution, thickness

    def generate(self,
                 name: str,
                 Gshp: tuple,
                 aff: torch.tensor,
                 header: torch.tensor,
                 Gimg: torch.tensor,
                 extras: None | list[nib.Nifti1Image] = None):
        """Method used to generate synthetic images

        Args:
            name (str): Image name, passed as additional metadata in the returned

            Gshp (tuple): Shape of the generated image (256x256x256).
            aff (torch.tensor): Affine of the generated image (use the seed affine)
            header (torch.tensor): Header of the generated image (use the seed header)
            Gimg (torch.tensor): Seed image (with labels corresponding to
            cfg.generator_params.base_generator.seed_lables)
            extras (None | list[nib.Nifti1Image], optional): Any additional image/segmentation
                that would need to be deformed together with the seed image. Use 
                extras to transform the ground truth segmentation or any other 
                annotation (using exactly the same spatial deformations applied to the 
                seed image). Defaults to None.

        Returns:
            tuple: (subject_meta, samples)
                subject_meta: dictionary with keys 'name' - named passed to the generator
                                                   'aff' - resulting affine matrix of the generated image
                                                   'header' - header of the generated image
                                                   'resolution' - voxel size
                samples: dictionary with keys 'synth_image': torch.Tensor image with the Gshp shape 
                                                             containing synthesized image
                                               'extras': list[torch.Tensor] list of deformed extra images
                                                         in the same order as passed to the generator 
                                                         in the same space as 'synt_image' (spatially deformed)
                                                         but interpolated using nearest_neighbor
        """

        with torch.no_grad():
            photo_mode = False

            spac = 2.0 + 10 * np.random.rand() if photo_mode else None
            flip = np.random.randn() < 0.5

            # sample affine deformation
            scaling_factor_distances, A, c2 = self.random_affine_transform(Gshp,
                                                                        self.max_rotation,
                                                                        self.max_shear,
                                                                        self.max_scaling)

            # sample nonlinear deformation
            F, Fneg = self.random_nonlinear_transform(photo_mode,
                                                    spac,
                                                    self.nonlin_scale_min,
                                                    self.nonlin_scale_max,
                                                    self.nonlin_std_max)
            # deform the images 
            xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(Gshp, A, c2, F)

            # Read in data
            G, extras = self.read_data(Gimg, [x1, x2, y1, y2, z1, z2],
                                    extras=extras)
            # Sampler
            resolution, thickness = self.random_sampler() 

            mus, sigmas = self.get_contrast(photo_mode)

            subject_meta = {'name': os.path.basename(name).split('.nii')[0],
                            'aff': aff, 'header': header, 'resolution': resolution}

            samples = self.process_sample(photo_mode, spac,
                                        thickness, resolution,
                                        flip, mus, sigmas,
                                        G,  [xx2, yy2, zz2],
                                        self.gamma_std, self.bf_scale_min,
                                        self.bf_scale_max, self.bf_std_min,
                                        self.bf_std_max, self.noise_std_min,
                                        self.noise_std_max, extras=extras)
            return subject_meta, samples

    def generate_extra(self, sub: str, extra: list[str]):
        idx = self.sub2idx[sub]
        return self.generate(idx, extra)

    @staticmethod
    def iso_resolution_sample(min_res=0.5, max_res=1.5):
        r = np.random.randint(min_res*100, max_res*100)/100
        resolution = np.array([1.0, 1.0, 1.0])*r
        thickness = np.array([1.0, 1.0, 1.0])*r
        return resolution, thickness

    def get_seeds(self):
        return self.names
