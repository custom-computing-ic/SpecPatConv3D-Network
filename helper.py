# Get Dataset
def maybeExtract(data, patch_size):

    import scipy.io
    try:
        data = scipy.io.loadmat("./data/Processed_" + data + "_patch_" + str(patch_size) + ".mat")
        train = (data['train_patch'], data['train_labels'])
        validation = (data['val_patch'], data['val_labels'])
        test = (data['test_patch'], data['test_labels'])

    except:
        raise Exception('--data options are: Indian_pines, Salinas, KSC, Botswana OR data files not existed')

    return train, validation, test


def maybeDownloadOrExtract(data):
    import scipy.io as io
    import os
    
    if data in ('KSC', 'Botswana'):
        gtfile = data
        filename = data
        readfile = data
        readgt = data        

    else:
        gtfile = data
        filename = data + '_corrected'          
        readfile = filename.lower()
        readgt = gtfile.lower()

    print("Dataset: " + filename)

    try:
        print("Try using images from Data folder...")
        input_mat = io.loadmat('./data/' + filename + '.mat')[readfile]
        target_mat = io.loadmat('./data/' + gtfile + '_gt.mat')[readgt + '_gt']

    except:
        print("Data not found, downloading input images and labelled images!\n\n")
        if data == "Indian_pines":
            url1 = "www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

        elif data == "Salinas":
            url1 = "www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

        elif data == "KSC":
            url1 = "http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat"

        elif data == "Botswana":
            url1 = "http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat"
            url2 = "http://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat"

        else:
            raise Exception("Available datasets are:: Indian_pines, Salinas, KSC, Botswana")

        os.system('wget -P' + ' ' + './data/' + ' ' + url1)
        os.system('wget -P' + ' ' + './data/' + ' ' + url2)


        input_mat = io.loadmat('./data/' + filename + '.mat')[readfile]
        print(input_mat)
     
        target_mat = io.loadmat('./data/' + gtfile + '_gt.mat')[readgt + '_gt']
        print(target_mat)

    return input_mat, target_mat


def getListLabel(data):
    if data == 'Indian_pines':
        return [2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15]

    elif data == 'Salinas':
        return list(range(1,16+1))

    elif data == 'Botswana':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  13, 14]

    elif data == 'KSC':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    else:
        raise Exception("Type error")


def OnehotTransform(labels):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)

    labels = np.reshape(labels, (len(labels), 1))
    labels = onehot_encoder.fit_transform(labels).astype(np.uint8)

    return labels


def getdtype(t):
    import numpy as np
    if t == 'float64':
        return np.float64
    elif t == 'float32':
        return np.float32
    elif t == 'float16':
        return np.float16
    elif t == 'int64':
        return np.int64
    elif t == 'int32':
        return np.int32
    elif t == 'int16':
        return np.int16
    elif t == 'int8':
        return np.int8
    else:
        # Default value
        return np.float64


# Depreciated function
def getTestDataset(test, test_label, size=250):
    '''
    Arguments: whole test data, test label,
    return randomized test data, test label of 'size'
    '''
    from numpy import array
    from random import shuffle

    assert test.shape[0] == test_label.shape[0]

    idx = list(range(test.shape[0]))
    shuffle(idx)
    idx = idx[:size]
    accuracy_x, accuracy_y = [], []
    for i in idx:
        accuracy_x.append(test[i])
        accuracy_y.append(test_label[i])

    return array(accuracy_x), array(accuracy_y)


def plot_random_spec_img(pic, true_label):
    '''
    Take first hyperspectral image from dataset and plot spectral data distribution
    Arguements pic = list of images in size (?, height, width, bands), where ? represents any number > 0
                true_labels = lists of ground truth corrospond to pic
    '''
    pic = pic[0]  #Take first data only
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import mean, argmax

    print("Image Shape: " + str(pic.shape) )
    print("Label of this image is -> " + str(true_label[0] ) )

    title = argmax(true_label[0], axis=0)
    # Calculate mean of all elements in the 3d element
    mean_value = mean(pic)
    # Replace element with less than mean by zero
    pic[pic < mean_value] = 0
    
    x = []
    y = []
    z = []
    # Coordinate position extractions
    for z1 in range(pic.shape[0]): 
        for x1 in range(pic.shape[1]):
            for y1 in range(pic.shape[2]):
                if pic[z1,x1,y1] != 0:
                    z.append(z1)
                    x.append(x1)
                    y.append(y1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('True class = '+ str(title))
    ax.scatter(x, y, z, color='#0606aa', marker='o', s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Spectral Label')
    ax.set_zlabel('Y Label')
    plt.show()


def GroundTruthVisualise(data, dataset, original=True):
    from matplotlib.pyplot import imshow, show, colorbar, set_cmap, clim
    import matplotlib.pyplot as plt
    import numpy as np

    labels = []

    if dataset == 'Indian_pines':
        if original:
            labels = ['Unlabelled','Corn-notil', 'Corn-mintill','Corn', 'Grass-pasture','Grass-trees','Hay-windrowed','Soybean-notil','Soybean-mintil','Soybean-clean','Woods','BGTD']
        else:
            labels = []

    elif dataset == 'Salinas':        
        labels = ['Unlabelled', 'Brocoli green weeds 1', 'Brocoli green weeds 2', 'Fallow', 'Fallow rough plow', 'Fallow smooth', 'Stubble', 'Celery','Grapes untrained', 'Soil vinyard develop', 'Corn senesced green weeds', 'Lettuce romaine 4wk', 'Lettuce romaine 5wk', 'Lettuce romaine 6wk', 'Lettuce romaine 7wk', 'Vinyard untrained', 'Vunyard vertical trellis']

    elif dataset == 'KSC':
        labels = ['Unlabelled','Scrub','Williw swamp','SP hammock','Slash pine','Oak/Broadleaf','Hardwood','Swamp','Gramminoid marsh','Spartina marsh','Cattail marsh','Salt marsh','Mud flats','Water']

    def discrete_matshow(data):
        #get discrete colormap
        cmap = plt.get_cmap('tab20', np.max(data)-np.min(data)+1)
        # set limits .5 outside true range
        mat = plt.matshow(data, cmap=cmap, vmin=np.min(data)-0.5, vmax=np.max(data)+0.5)
        #tell the colorbar to tick at integers
        cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))

        cax.ax.set_yticklabels(labels)

    imshow(data)
    discrete_matshow(data)
    show()

# Arguement: data = 3D image in size (h,w,bands)
def plotStatlieImage(data, bird=False):
    from matplotlib.pyplot import imshow, show, subplots, axis, figure
    print('\nPlotting a band image')
    fig, ax = subplots(nrows=3, ncols=3)
    i = 1
    for row in ax:
        for col in row:
            i += 11
            if bird:
                col.imshow(data[i,:,:])
            else:
                col.imshow(data[:,:,i])
            axis('off')
    show()


def showClassTable(number_of_list, title='Number of samples'):
    import pandas as pd 
    print("\n+------------Show Table---------------+")
    lenth = len(number_of_list)
    column1 = range(1, lenth+1)
    table = {'Class#': column1, title: number_of_list}
    table_df = pd.DataFrame(table).to_string(index=False)
    print(table_df)   
    print("+-----------Close Table-----------------+")


def get_available_gpus():

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    a = [ x.name.replace('/device:GPU:','') for x in local_device_protos if x.device_type == 'GPU']
    if len(a) > 2:
        a = a
    else:
        a = None

    return len(a), a 


if __name__ == '__main__':
    print('You re now in helper function')
    _, a = get_available_gpus()
    print(a)
