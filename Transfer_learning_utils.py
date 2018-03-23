import numpy as np
import glob
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def loading_dataset(folderpath, classes, train_pct = 0.8, verbose = False):
    
    images_data = []
    labels = []
    
    for iclass,class_name in enumerate(classes):
        this_folder = folderpath + class_name
        sub_images = glob.glob(this_folder + '/*')
        for image_path in sub_images:
            if verbose:
                print('reading %s with label %s' % (image_path, classes[iclass]))
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            #x = preprocess_input(x)
            images_data.append(x)
            labels.append(iclass)
    
    images_data = np.array(images_data)
    labels = np.array(labels)
    
    m = images_data.shape[0]
    permuted_index = np.random.permutation(m)
    split_m = int(np.ceil(m*train_pct))

    train_ind = list(permuted_index[0:split_m])
    test_ind = list(permuted_index[split_m:])

    x_train, y_train = images_data[train_ind], labels[train_ind]
    x_test,  y_test  = images_data[test_ind],  labels[test_ind]
    
    return x_train, y_train, x_test, y_test


def evaluate_model(model, X_train, Y_train, X_test, Y_test, maxItem = 100):
    #test model on training data set and testing data set
    nTrain = min([maxItem, X_train.shape[0]])
    print('Performance on Training data set (%d)' % nTrain)
    preds = model.evaluate(X_train[0:nTrain], Y_train[0:nTrain])
    print ("Loss = " + str(preds[0]))
    print ("Train Accuracy = " + str(preds[1]))

    nTest = min([maxItem, X_test.shape[0]])
    print('Performance on Testing data set (%d)' % nTest)
    preds = model.evaluate(X_test[0:nTest], Y_test[0:nTest])
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

