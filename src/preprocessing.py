import os
import numpy as np
import nibabel as nib


def compute_train_stats(train_files, data_folder_path):
    """Compute mean and std from all nonzero voxels of training T1/T2 volumes."""
    voxels = []
    for filename in train_files:
        T1name = filename[:-10] + "-T1.img"
        T2name = filename[:-10] + "-T2.img"

        T1 = nib.load(os.path.join(data_folder_path, T1name)).get_fdata()
        T2 = nib.load(os.path.join(data_folder_path, T2name)).get_fdata()

        voxels.append(T1[T1 > 0])
        voxels.append(T2[T2 > 0])

    all_voxels = np.concatenate(voxels)
    mu = np.mean(all_voxels)
    std = np.std(all_voxels)

    return mu, std


def normalize_with_stats(img, mu, std):
    """Normalize a volume using fixed training-set mean and std."""
    return (img - mu) / std
