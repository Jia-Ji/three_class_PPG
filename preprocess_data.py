import os
import scipy.io
import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

# Load .mat file
def load_mat_any(path):
    try:
        import h5py, hdf5storage  # v7.3
        if h5py.is_hdf5(path):
            return hdf5storage.loadmat(path)
    except Exception:
        pass
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


mat_file_path = "./PPG_data/Compiled/merged_data_8022_3850_labeled.mat"
# mat_data = scipy.io.loadmat(mat_file_path)
mat_data = load_mat_any(mat_file_path)

PPG_ECG = mat_data['S']
labels = mat_data['labels']
PPG = PPG_ECG[:, :, 0]
ECG = PPG_ECG[:, :, 1]
PPG = np.transpose(PPG)
ECG = np.transpose(ECG)

# PPG = np.expand_dims(PPG, axis=1)
PPG = np.expand_dims(PPG, axis=1)

# load .csv file
csv_file_path = "./PPG_data/Compiled/merged_data_8022_3850_labeled.csv"
df = pd.read_csv(csv_file_path)
df['index'] = df.index

# Select target ECG classes and encode them (Require Action)
label_mapping = {
    'NORM': 0,
    'PVC': 1,
    # 'PAC': 2,
}

target_class_order = ['NORM', 'PVC']


target_classes = set(label_mapping.keys())

# Remove rows that are not part of the target classes (Require Action)
df_filtered = df[df['ECGcat'].isin(target_classes)].copy()

# print(df_filtered['ECGcat'].value_counts() )


summary = (
    df_filtered.groupby(['ID0', 'ECGcat'])
    .size()
    .reset_index(name='count')
)

pivot = summary.pivot(index='ID0', columns='ECGcat', values='count').fillna(0)
X = np.array(pivot.index).reshape(-1, 1)
y = pivot.values


# First split: train (60%) vs temp (40%)
X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.4)

# Second split: val (20%) vs test (20%) from temp
X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

train_patients = X_train.flatten()
val_patients = X_val.flatten()
test_patients = X_test.flatten()

df_train = df_filtered[df_filtered['ID0'].isin(train_patients)]
df_val = df_filtered[df_filtered['ID0'].isin(val_patients)]
df_test = df_filtered[df_filtered['ID0'].isin(test_patients)]

df_train_index = df_train["index"].tolist()
df_val_index = df_val["index"].tolist()
df_test_index = df_test["index"].tolist()

x_train = PPG[df_train_index]
x_val = PPG[df_val_index]
x_test = PPG[df_test_index]

ecg_x_train = ECG[df_train_index]
ecg_x_val = ECG[df_val_index]
ecg_x_test = ECG[df_test_index]

y_train = df_train['ECGcat'].map(label_mapping).to_numpy()
y_val = df_val['ECGcat'].map(label_mapping).to_numpy()
y_test = df_test['ECGcat'].map(label_mapping).to_numpy()

print(f"The shape of x_train is {x_train.shape}")
print(f"The shape of x_val is {x_val.shape}")
print(f"The shape of x_test is {x_test.shape}")
print(f"The shape of y_train is {y_train.shape}")
print(f"The shape of y_val is {y_val.shape}")
print(f"The shape of y_test is {y_test.shape}")

print(f"The shape of ecg_x_train is {ecg_x_train.shape}")
print(f"The shape of ecg_x_val is {ecg_x_val.shape}")
print(f"The shape of ecg_x_test is {ecg_x_test.shape}")

print(f"The type of y_train is {type(y_train)}")
print(f"The type of x_train is {type(x_train)}")
print(f"The type of y_val is {type(y_val)}")
print(f"The type of x_val is {type(x_val)}")
print(f"The type of y_test is {type(y_test)}")
print(f"The type of x_test is {type(x_test)}")
#  check stratification
print('Check stratification.......')

def compute_distribution(df_split):
    dist = df_split['ECGcat'].value_counts(normalize=True) * 100
    return dist.reindex(target_class_order).fillna(0).round(2)

train_distribution = compute_distribution(df_train)
val_distribution = compute_distribution(df_val)
test_distribution = compute_distribution(df_test)

def show_distribution(name, distribution):
    print(f"\n{name} PPG distribution (%):")
    print(distribution)

show_distribution("Train", train_distribution)
show_distribution("Validation", val_distribution)
show_distribution("Test", test_distribution)

# Should be empty sets
print(set(train_patients) & set(val_patients))
print(set(train_patients) & set(test_patients))
print(set(val_patients) & set(test_patients))

command = input("Enter your command (yes/no): ")
if command == "yes":
    output_dir = './PPG_data/splitted_data'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'x_train.npy'), 'wb') as f:
        np.save(f, x_train)
    with open(os.path.join(output_dir, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train)
    with open(os.path.join(output_dir, 'x_val.npy'), 'wb') as f:
        np.save(f, x_val)
    with open(os.path.join(output_dir, 'y_val.npy'), 'wb') as f:
        np.save(f, y_val)
    with open(os.path.join(output_dir, 'x_test.npy'), 'wb') as f:
        np.save(f, x_test)
    with open(os.path.join(output_dir, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test)

    distribution_path = os.path.join(output_dir, 'class_distribution.txt')
    with open(distribution_path, 'w') as f:
        for name, distribution in [
            ("Train", train_distribution),
            ("Validation", val_distribution),
            ("Test", test_distribution),
        ]:
            f.write(f"{name} PPG distribution (%):\n")
            f.write(distribution.to_string())
            f.write("\n\n")

    # Save paired ECG splits for downstream visualization/analysis
    with open(os.path.join(output_dir, 'ECG_x_train.npy'), 'wb') as f:
        np.save(f, ecg_x_train)
    with open(os.path.join(output_dir, 'ECG_x_val.npy'), 'wb') as f:
        np.save(f, ecg_x_val)
    with open(os.path.join(output_dir, 'ECG_x_test.npy'), 'wb') as f:
        np.save(f, ecg_x_test)
elif command == "no":
    pass