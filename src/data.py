import os
import numpy as np
import nibabel as nib

from .preprocessing import compute_train_stats, normalize_with_stats



def train_val_test_split(
    data_folder_path,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seullaire=100,
    random_seed=42,
):
    """
    Split patients into train/val/test sets and return normalized slices.
    Normalization is done using mean/std from the *training set only*.
    """

    all_label_files = [
        f for f in os.listdir(data_folder_path)
        if "label" in f and "hdr" not in f
    ]
    all_label_files.sort()

    np.random.seed(random_seed)
    np.random.shuffle(all_label_files)

    num_patients = len(all_label_files)
    n_train = int(num_patients * train_ratio)
    n_val = int(num_patients * val_ratio)
    n_test = num_patients - n_train - n_val

    train_files = all_label_files[:n_train]
    val_files = all_label_files[n_train:n_train + n_val]
    test_files = all_label_files[n_train + n_val:]

    print("=== Patients assigned to training ===")
    for f in train_files: print(f"  {f}")
    print("=== Patients assigned to validation ===")
    for f in val_files: print(f"  {f}")
    print("=== Patients assigned to test ===")
    for f in test_files: print(f"  {f}")


    mu, std = compute_train_stats(train_files, data_folder_path)


    def process_patient(file_list, x_list, y_list):
        for filename in file_list:
            T1name = filename[:-10] + "-T1.img"
            T2name = filename[:-10] + "-T2.img"

            T1 = nib.load(os.path.join(data_folder_path, T1name)).get_fdata()
            T2 = nib.load(os.path.join(data_folder_path, T2name)).get_fdata()
            labels = nib.load(os.path.join(data_folder_path, filename)).get_fdata()

            sx, sy, sz, _ = labels.shape
            labels = np.array(labels).reshape((sx, sy, sz))

            T1norm = normalize_with_stats(T1, mu, std).reshape((sx, sy, sz))
            T2norm = normalize_with_stats(T2, mu, std).reshape((sx, sy, sz))

            for z in range(sz):
                slice_label = labels[:, :, z]
                if np.sum(slice_label >= 0) >= seullaire:
                    x_list.append(np.stack([T1norm[:, :, z], T2norm[:, :, z]], axis=-1))
                    y_list.append(slice_label[:, :, np.newaxis])


    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    process_patient(train_files, x_train_list, y_train_list)
    process_patient(val_files, x_val_list, y_val_list)
    process_patient(test_files, x_test_list, y_test_list)

    x_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    x_val = np.array(x_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    x_test = np.array(x_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test
