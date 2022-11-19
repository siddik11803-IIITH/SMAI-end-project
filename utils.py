import PIL, os, pickle
import numpy as np
import features as fe



def get_test_train_data():
    train_face_dir= "Data/faces/face.train/train/face/"
    train_nonface_dir = "Data/faces/face.train/train/non-face/"
    test_face_dir = "Data/faces/face.test/test/face/"
    test_nonface_dir = "Data/faces/face.test/test/non-face/"
    
    X_train = []
    y_train = []
    for image_name in os.listdir(train_face_dir):
        image = PIL.Image.open(train_face_dir + image_name)
        X_train.append(np.asarray(image))
        y_train.append(1)
    for image_name in os.listdir(train_nonface_dir):
        image = PIL.Image.open(train_nonface_dir + image_name)
        X_train.append(np.asarray(image))
        y_train.append(0)

    X_test = []
    y_test = []
    for image_name in os.listdir(test_face_dir):
        image = PIL.Image.open(test_face_dir + image_name)
        X_test.append(np.asarray(image))
        y_test.append(1)
    for image_name in os.listdir(test_nonface_dir):
        image = PIL.Image.open(test_nonface_dir + image_name)
        X_test.append(np.asarray(image))
        y_test.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def store_in_pickle(filename):
    X_train_img, X_test_img, y_train, y_test = ut.get_test_train_data()
    print(X_train_img.shape)
    print(X_test_img.shape)

    rects = fe.get_rectanges(X_train_img.shape[1], X_train_img.shape[2])
    X_train = [fe.get_features(image, rects) for image in X_train_img]
    X_test = [fe.get_features(image, rects) for image in X_test_img]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Data = [X_train, X_test, y_train, y_test]
    for data in Data:
        print(Data.shape)
        
    file = open(filename, 'wb')
    pickle.dump(Data, file)

