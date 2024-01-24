from PIL import Image
import blobfile as bf
import os, glob
from mpi4py import MPI
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
import torch
from pytorch_wavelets import DWTForward

def interpolate(input_arr, target_arr):
    y_zoom = target_arr.shape[0]/input_arr.shape[0]
    x_zoom = target_arr.shape[1]/input_arr.shape[1]
    zoom_factors = (y_zoom, x_zoom, 1)
    interpolated_arr = zoom(input_arr, zoom_factors, order = 3)
    return interpolated_arr

def extract_prefix(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True
        )
    while True:
        yield from loader


def load_pair_data(
        *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False, full_resolution=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified input data directory")
    if not target_dir:
        raise ValueError("unspecified target data directory")

    all_input_files = _list_image_files_recursively(input_dir)
    all_target_files = _list_image_files_recursively(target_dir)
    classes = None

    if full_resolution:
        dataset = PairedImageDataset(
            None,
            all_input_files,
            all_target_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    else:
        dataset = PairedImageDataset(
            image_size,
            all_input_files,
            all_target_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False
        )
    while True:
        yield from loader


def load_paired_npy_data(
    *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '*.npy'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '*.npy'))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = PairedNPYDataset(
        image_size,
        all_inputs,
        all_targets,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True
        )
    while True:
        yield from loader


def load_paired_mat_data(
    *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '**/input_reg/*.mat'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '**/target_aligned_style2/*.mat'))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = PairedMATDataset(
        image_size,
        all_inputs,
        all_targets,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class PairedImageDataset(Dataset):
    def __init__(self, resolution, input_image_paths, target_image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution

        # input_images_dict = {extract_prefix(path): path for path in input_image_paths}
        # target_images_dict = {extract_prefix(path): path for path in target_image_paths}
        #
        # all_keys = set(input_images_dict.keys()) | set(target_images_dict.keys())
        #
        # reordered_input_images = []
        # reordered_target_images = []
        #
        # for key in sorted(all_keys):
        #     input_image = input_images_dict.get(key, None)
        #     target_image = target_images_dict.get(key, None)
        #     # Append the file paths or placeholders to the reordered lists.
        #     # if extract_prefix(input_image) != extract_prefix(target_image):
        #     #     raise ValueError("input and target images don't match")
        #     if input_image:
        #         reordered_input_images.append(input_image)
        #     if target_image:
        #         reordered_target_images.append(target_image)
        #
        # self.local_input_images = reordered_input_images[shard:][::num_shards]
        # self.local_target_images = reordered_target_images[shard:][::num_shards]

        self.local_input_images = input_image_paths
        self.local_target_images = target_image_paths

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        if len(self.local_input_images) == len(self.local_target_images):
            return len(self.local_input_images)
        else:
            raise ValueError('different number of input and target images')

    def __getitem__(self, idx):

        # padding function
        def pad_to_resolution(arr, resolution):
            pad_height = max(resolution - arr.shape[0], 0)
            pad_width = max(resolution - arr.shape[1], 0)

            if pad_height > 0 or pad_width > 0:
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left

                arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

            return arr

        # input
        input_path = self.local_input_images[idx]
        with bf.BlobFile(input_path, "rb") as f:
            input_pil_image = Image.open(f)
            input_pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        # while min(*input_pil_image.size) >= 2 * self.resolution:
        #     input_pil_image = input_pil_image.resize(
        #         tuple(x // 2 for x in input_pil_image.size), resample=Image.BOX
        #     )

        # input_scale = self.resolution / min(*input_pil_image.size)
        # input_pil_image = input_pil_image.resize(
        #     tuple(round(x * input_scale) for x in input_pil_image.size), resample=Image.BICUBIC
        # )

        input_arr = np.array(input_pil_image.convert("RGB"))
        target_path = self.local_target_images[idx]
        with bf.BlobFile(target_path, "rb") as f:
            target_pil_image = Image.open(f)
            target_pil_image.load()
        target_arr = np.array(target_pil_image.convert("RGB"))
        input_arr = interpolate(input_arr, target_arr)
        if self.resolution is not None:
            input_arr = pad_to_resolution(input_arr, self.resolution)
            target_arr = pad_to_resolution(target_arr, self.resolution)
            crop_y = (input_arr.shape[0] - self.resolution) // 2
            crop_x = (input_arr.shape[1] - self.resolution) // 2
            input_arr = input_arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]

        input_arr = input_arr.astype(np.float32) / 127.5 - 1

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        # while min(*target_pil_image.size) >= 2 * self.resolution:
        #     target_pil_image = target_pil_image.resize(
        #         tuple(x // 2 for x in target_pil_image.size), resample=Image.BOX
        #     )

        # target_scale = self.resolution / min(*target_pil_image.size)
        # target_pil_image = target_pil_image.resize(
        #     tuple(round(x * target_scale) for x in target_pil_image.size), resample=Image.BICUBIC
        # )
        if self.resolution is not None:
            crop_y = (target_arr.shape[0] - self.resolution) // 2
            crop_x = (target_arr.shape[1] - self.resolution) // 2
            target_arr = target_arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]
        target_arr = target_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(input_arr, [2, 0, 1]), np.transpose(target_arr, [2, 0, 1]), out_dict


# class PairedImageDataset(Dataset):
#     def __init__(self, resolution, input_image_paths, target_image_paths, classes=None, shard=0, num_shards=1):
#         super().__init__()
#         self.resolution = resolution
#
#         input_images_dict = {extract_prefix(path): path for path in input_image_paths}
#         target_images_dict = {extract_prefix(path): path for path in target_image_paths}
#
#         all_keys = set(input_images_dict.keys()) | set(target_images_dict.keys())
#
#         reordered_input_images = []
#         reordered_target_images = []
#
#         for key in sorted(all_keys):
#             input_image = input_images_dict.get(key, None)
#             target_image = target_images_dict.get(key, None)
#             # Append the file paths or placeholders to the reordered lists.
#             # if extract_prefix(input_image) != extract_prefix(target_image):
#             #     raise ValueError("input and target images don't match")
#             if input_image:
#                 reordered_input_images.append(input_image)
#             if target_image:
#                 reordered_target_images.append(target_image)
#
#         self.local_input_images = reordered_input_images[shard:][::num_shards]
#         self.local_target_images = reordered_target_images[shard:][::num_shards]
#
#         self.local_classes = None if classes is None else classes[shard:][::num_shards]
#
#     def __len__(self):
#         if len(self.local_input_images) == len(self.local_target_images):
#             return len(self.local_input_images)
#         else:
#             raise ValueError('different number of input and target images')
#
#     def __getitem__(self, idx):
#         # input
#         input_path = self.local_input_images[idx]
#         with bf.BlobFile(input_path, "rb") as f:
#             input_pil_image = Image.open(f)
#             input_pil_image.load()
#
#         # We are not on a new enough PIL to support the `reducing_gap`
#         # argument, which uses BOX downsampling at powers of two first.
#         # Thus, we do it by hand to improve downsample quality.
#         # while min(*input_pil_image.size) >= 2 * self.resolution:
#         #     input_pil_image = input_pil_image.resize(
#         #         tuple(x // 2 for x in input_pil_image.size), resample=Image.BOX
#         #     )
#
#         # input_scale = self.resolution / min(*input_pil_image.size)
#         # input_pil_image = input_pil_image.resize(
#         #     tuple(round(x * input_scale) for x in input_pil_image.size), resample=Image.BICUBIC
#         # )
#
#         input_arr = np.array(input_pil_image.convert("RGB"))
#         target_path = self.local_target_images[idx]
#         with bf.BlobFile(target_path, "rb") as f:
#             target_pil_image = Image.open(f)
#             target_pil_image.load()
#         target_arr = np.array(target_pil_image.convert("RGB"))
#         input_arr = interpolate(input_arr, target_arr)
#
#         crop_y = (input_arr.shape[0] - self.resolution) // 2
#         crop_x = (input_arr.shape[1] - self.resolution) // 2
#         input_arr = input_arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
#         input_arr = input_arr.astype(np.float32) / 127.5 - 1
#
#
#         # We are not on a new enough PIL to support the `reducing_gap`
#         # argument, which uses BOX downsampling at powers of two first.
#         # Thus, we do it by hand to improve downsample quality.
#         # while min(*target_pil_image.size) >= 2 * self.resolution:
#         #     target_pil_image = target_pil_image.resize(
#         #         tuple(x // 2 for x in target_pil_image.size), resample=Image.BOX
#         #     )
#
#         # target_scale = self.resolution / min(*target_pil_image.size)
#         # target_pil_image = target_pil_image.resize(
#         #     tuple(round(x * target_scale) for x in target_pil_image.size), resample=Image.BICUBIC
#         # )
#
#         crop_y = (target_arr.shape[0] - self.resolution) // 2
#         crop_x = (target_arr.shape[1] - self.resolution) // 2
#         target_arr = target_arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
#         target_arr = target_arr.astype(np.float32) / 127.5 - 1
#
#         out_dict = {}
#         # if self.local_classes is not None:
#         #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
#         return np.transpose(input_arr, [2, 0, 1]), np.transpose(target_arr, [2, 0, 1]), out_dict


class PairedNPYDataset(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            inp = np.load(f).astype('float32')

        crop_y = (inp.shape[1] - self.resolution) // 2
        crop_x = (inp.shape[2] - self.resolution) // 2
        inp = inp[:, crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        inp = (inp - inp.mean(axis=(1,2), keepdims=True)) / (inp.std(axis=(1,2), keepdims=True) + 1e-6)
        inp = inp[:3,...]
        # inp = inp.astype(np.float32) / 127.5 - 1

        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            tag = np.load(f).astype('float32')

        crop_y = (tag.shape[1] - self.resolution) // 2
        crop_x = (tag.shape[2] - self.resolution) // 2
        tag = tag[:, crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        tag = tag.astype(np.float32) / 127.5 - 1
        # tag = (tag - tag.mean(axis=(1,2), keepdims=True)) / (tag.std(axis=(1,2), keepdims=True) + 1e-6)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag, inp, out_dict


class PairedMATDataset(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            inp = sio.loadmat(f)['input'].astype('float32')

        # inp: [H, W, C], range [0, >1]
        crop_y = (inp.shape[0] - self.resolution) // 2
        crop_x = (inp.shape[1] - self.resolution) // 2
        inp = inp[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        inp = (inp - inp.mean(axis=(1,2), keepdims=True)) / (inp.std(axis=(1,2), keepdims=True) + 1e-6)
        inp = inp[...,:3]
        # inp = inp.astype(np.float32) / 127.5 - 1

        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            tag = sio.loadmat(f)['target'].astype('float32')
        
        # tag: [H, W, C], range [0, 1]
        crop_y = (tag.shape[0] - self.resolution) // 2
        crop_x = (tag.shape[1] - self.resolution) // 2
        tag = tag[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        tag = tag.astype(np.float32) * 2 - 1
        # tag = tag.astype(np.float32) / 127.5 - 1
        # tag = (tag - tag.mean(axis=(1,2), keepdims=True)) / (tag.std(axis=(1,2), keepdims=True) + 1e-6)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag.transpose([2,0,1]), inp.transpose([2,0,1]), out_dict

def wavelet_transform(img_tensors: torch.Tensor, layers):
    xfm = DWTForward(J=1, mode='periodization', wave='db1')
    for i in range(layers):
        Yls = []
        for channel in range(3):
            channel_tensor = img_tensors[:, channel, :, :].unsqueeze(1)
            Yl, Yh = xfm(channel_tensor)
            Yls.append(Yl)
        img_tensors = torch.cat(Yls, dim=1)

    return img_tensors