import numpy as np
from random import shuffle
import scipy.io as io
import argparse
from helper import *

#For fix point arithmetic
from decimal import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines', help='default:Indian_pines, options: Salinas, KSC, Botswana')
parser.add_argument('--patch_size', type=int, default=5, help='Feature size, odd number integer')
parser.add_argument('--train_ratio', type=float, default=0.1, help='Fraction for training from data')
parser.add_argument('--validation_ratio', type=float, default=0.05, help='Fraction for validation from data')
parser.add_argument('--dtype', type=str, default='float32', help='Data type (Eg float64, float32, float16, int64...')
parser.add_argument('--plot', type=bool, default=False, help='Set TRUE for visualizing the statlie images and ground truth')
opt = parser.parse_args()

# Try loading data from the folder... Otherwise download from online
# Download dataset or extract dataset if existed in 'data' folder
input_mat, target_mat = maybeDownloadOrExtract(opt.data)

# np.float64; np.float32, np.float16; np.int64; np.int32; np.int16; np.int8; np.uint64; np.uint32; np.uint16
datatype = getdtype(opt.dtype)
PATCH_SIZE = opt.patch_size
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
OUTPUT_CLASSES = np.max(target_mat)

# Normalize image data and select datatype
input_mat = input_mat.astype(np.float64)
input_mat = input_mat - np.min(input_mat)
input_mat = input_mat / np.max(input_mat)

# List that contains classes for training
list_labels = getListLabel(opt.data)

print("+-------------------------------------+")
print('Input_mat shape: ' +  str(input_mat.shape) )

MEAN_ARRAY = np.ndarray(shape=(BAND,1))
new_input_mat = []


calib_val_pad = int((PATCH_SIZE-1)/2)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:, i])
    new_input_mat.append(np.pad(input_mat[:,:, i], calib_val_pad, 'constant', constant_values=0))

new_input_mat = np.transpose(new_input_mat, (1, 2, 0))

print("+-------------------------------------+")
input_mat = new_input_mat

def Patch(height_index, width_index):

    # Input:
    # Given the index position (x,y) of spatio dimension of the hyperspectral image,

    # Output:
    # a data cube with patch size S (24 neighbours), with label based on central pixel

    transpose_array = input_mat
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)

    patch = transpose_array[height_slice, width_slice, :]
    mean_normalized_patch = []
    for i in range(BAND):
        mean_normalized_patch.append(patch[:, :, i] - MEAN_ARRAY[i])

    mean_normalized_patch = np.array(mean_normalized_patch).astype(datatype)
    mean_normalized_patch = np.transpose(mean_normalized_patch, (1, 2, 0))
    return mean_normalized_patch


# Assign empty array to store patched images
CLASSES = [[] for i in range(OUTPUT_CLASSES)]

# Assign empty array to count samples for each class
class_label_counter = [0] * OUTPUT_CLASSES

# Start timing for loading
# t = threading.Thread(target=animate).start()
from tqdm import tqdm
count = 0
for i in tqdm(range(HEIGHT-1)):

    for j in range(WIDTH-1):
        curr_inp = Patch(i, j)

        curr_tar = target_mat[i, j]

        if curr_tar:
            CLASSES[curr_tar-1].append(curr_inp)
            class_label_counter[curr_tar-1] += 1
            count += 1

end_loading = True
print('Total number of samples: ' + str(count))

# Show number of samples for each class in the data set
showClassTable(class_label_counter)

# Split the dataset into training, validation abd testing,
# as well as dropping classes with insufficient samples

TRAIN_PATCH, TRAIN_LABELS = [], []
TEST_PATCH, TEST_LABELS =[], []
VAL_PATCH, VAL_LABELS = [], []

train_ratio = opt.train_ratio
val_ratio = opt.validation_ratio
# test_ratio = reminder of data

counter = 0  # train_index position
info = {}    # Dictionary type to check [train, validation, test] for each class
for i, data in enumerate(CLASSES):
    datasize = []
    if i + 1 in list_labels:

        shuffle(data)
        print('Class ' + str(i + 1) + ' is accepted')

        size = round(class_label_counter[i]*train_ratio)

        TRAIN_PATCH += data[:size]
        TRAIN_LABELS += [counter] * size
        datasize.append(size)

        size1 = round(class_label_counter[i]*val_ratio)
        VAL_PATCH += data[size:size+size1]
        VAL_LABELS += [counter] * (size1)
        datasize.append(size1)

        TEST_PATCH += data[size+size1:]
        TEST_LABELS += [counter] * len(data[size+size1:])
        datasize.append(len(TEST_PATCH))

        counter += 1

        info[counter] = datasize
    else:
        print('-Class ' + str(i + 1) + ' is rejected due to insufficient samples')

# print(info) # Samples sizes for each classes

TRAIN_LABELS = np.array(TRAIN_LABELS)
TRAIN_PATCH = np.array(TRAIN_PATCH)
TEST_PATCH = np.array(TEST_PATCH)
TEST_LABELS = np.array(TEST_LABELS)
VAL_PATCH = np.array(VAL_PATCH)
VAL_LABELS = np.array(VAL_LABELS)

print("+-------------------------------------+")
print("Size of Training data: " + str(len(TRAIN_PATCH)) )
print("Size of Validation data: " + str(len(VAL_PATCH))  )
print("Size of Testing data: " + str(len(TEST_PATCH)) )
print("+-------------------------------------+")

processed_data = {}

train_idx = list(range(len(TRAIN_PATCH)))
shuffle(train_idx)
TRAIN_PATCH = TRAIN_PATCH[train_idx]
TRAIN_LABELS = TRAIN_LABELS[train_idx]
TRAIN_LABELS = OnehotTransform(TRAIN_LABELS)
processed_data["train_patch"] = TRAIN_PATCH
processed_data["train_labels"] = TRAIN_LABELS

test_idx = list(range(len(TEST_PATCH)))
shuffle(test_idx)
TEST_PATCH = TEST_PATCH[test_idx]
TEST_LABELS = TEST_LABELS[test_idx]
TEST_LABELS = OnehotTransform(TEST_LABELS)
processed_data["test_patch"] = TEST_PATCH
processed_data["test_labels"] = TEST_LABELS

val_idx = list(range(len(VAL_PATCH)))
shuffle(val_idx)
VAL_PATCH = VAL_PATCH[val_idx]
VAL_LABELS = VAL_LABELS[val_idx]
VAL_LABELS = OnehotTransform(VAL_LABELS)
processed_data["val_patch"] = VAL_PATCH
processed_data["val_labels"] = VAL_LABELS

io.savemat("./data/Processed_" + opt.data + "_patch_" + str(PATCH_SIZE) + ".mat", processed_data)

print(TRAIN_PATCH.dtype)
print(TEST_PATCH.dtype)
print(VAL_PATCH.dtype)

print("+-------------------------------------+")
print("Summary")
print('Train_patch.shape: '+ str(TRAIN_PATCH.shape) )
print('Train_label.shape: '+ str(TRAIN_LABELS.shape) )
print('Test_patch.shape: ' + str(TEST_PATCH.shape))
print('Test_label.shape: ' + str(TEST_LABELS.shape))
print("Validation batch Shape: " + str(VAL_PATCH.shape) )
print("Validation label Shape: " + str(VAL_LABELS.shape) )
print("+-------------------------------------+")
print("\nFinished processing.......")


if opt.plot:
    print('\n Looking at some sample images')
    plot_random_spec_img(TRAIN_PATCH, TRAIN_LABELS)
    plot_random_spec_img(TEST_PATCH, TEST_LABELS)
    plot_random_spec_img(VAL_PATCH, VAL_LABELS)

    # Show origin statlie image
    plotStatlieImage(input_mat, bird=True)
    print(target_mat.dtype)
    # Show transposed statlie image (reflection along x=y asix)
    target_mat = np.array(target_mat)
    print(target_mat.dtype)
    print(target_mat.shape)
    GroundTruthVisualise(target_mat, opt.data)

