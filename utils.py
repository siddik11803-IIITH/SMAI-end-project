import PIL, os, pickle
import numpy as np
import features as fe
from PIL import Image
from sklearn.model_selection import train_test_split


def get_test_train_data():
    train_face_dir= "Data/faces/face.train/train/face/"
    train_nonface_dir = "Data/faces/face.train/train/non-face/"
    test_face_dir = "Data/faces/face.test/test/face/"
    test_nonface_dir = "Data/faces/face.test/test/non-face/"

    X_train_face = []
    y_train_face = []
    for image_name in os.listdir(train_face_dir):
        image = Image.open(train_face_dir + image_name)
        X_train_face.append(np.asarray(image))
        y_train_face.append(1)

    X_train_nonface = []
    y_train_nonface = []
    for image_name in os.listdir(train_nonface_dir):
        image = Image.open(train_nonface_dir + image_name)
        X_train_nonface.append(np.asarray(image))
        y_train_nonface.append(0)

    X_test = []
    y_test = []
    for image_name in os.listdir(test_face_dir):
        image = Image.open(test_face_dir + image_name)
        X_test.append(np.asarray(image))
        y_test.append(1)
    for image_name in os.listdir(test_nonface_dir):
        image = Image.open(test_nonface_dir + image_name)
        X_test.append(np.asarray(image))
        y_test.append(0)

    X_train_face = np.array(X_train_face)
    X_train_nonface = np.array(X_train_nonface)
    y_train_face = np.array(y_train_face)
    y_train_nonface = np.array(y_train_nonface)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train_face, X_train_nonface, X_test, y_train_face, y_train_nonface, y_test


def test_get_test_train_data():
    a = get_test_train_data()
    assert(a[0].shape == (2429, 19, 19))
    assert(a[1].shape == (4548, 19, 19))
    assert(a[2].shape == (24045, 19, 19))
    assert(a[3].shape == (2429,))
    assert(a[4].shape == (4548,))
    assert(a[5].shape == (24045,))


def random_subset(Images, lables, N):
    index = np.random.choice(Images.shape[0], N, replace=False)  
    sub_images = Images[index]
    sub_labels = lables[index]
    return sub_images, sub_labels


def Create_data(Dataset_Size):
    rect = fe.get_rectanges(19, 19)
    no_rect = fe.get_no_rectangles(19, 19)
    X_train_face_img, X_train_nonface_img, X_test_img, y_train_face, y_train_nonface, y_test = get_test_train_data()
    X_train_sub_nonface_img, y_train_sub_nonface = random_subset(X_train_nonface_img, y_train_nonface, len(X_train_face_img))
    assert(len(X_train_face_img) == len(X_train_sub_nonface_img))
    assert(len(y_train_face) == len(y_train_sub_nonface))
    assert(len(X_train_face_img) == len(y_train_face))
    assert(len(X_train_sub_nonface_img) == len(y_train_sub_nonface))

    X_train_img = np.concatenate((X_train_face_img, X_train_sub_nonface_img))
    y_train = np.concatenate((y_train_face, y_train_sub_nonface))

    X_data, y_data = random_subset(X_train_img, y_train, Dataset_Size)
    X_data_fe = fe.par_feature_extraction_images(X_data, rect, no_rect)
    return X_data_fe, y_data
    

def Create_data_imgs(Dataset_Size):
    rect = fe.get_rectanges(19, 19)
    no_rect = fe.get_no_rectangles(19, 19)
    X_train_face_img, X_train_nonface_img, X_test_img, y_train_face, y_train_nonface, y_test = get_test_train_data()
    X_train_sub_nonface_img, y_train_sub_nonface = random_subset(X_train_nonface_img, y_train_nonface, len(X_train_face_img))
    assert(len(X_train_face_img) == len(X_train_sub_nonface_img))
    assert(len(y_train_face) == len(y_train_sub_nonface))
    assert(len(X_train_face_img) == len(y_train_face))
    assert(len(X_train_sub_nonface_img) == len(y_train_sub_nonface))

    X_train_img = np.concatenate((X_train_face_img, X_train_sub_nonface_img))
    y_train = np.concatenate((y_train_face, y_train_sub_nonface))

    X_data, y_data =random_subset(X_train_img, y_train, Dataset_Size)
    X_data_fe = fe.par_feature_extraction_images(X_data, rect, no_rect)
    return X_data_fe, X_data, y_data
 

def Split_Data(X_data, y_data):
    return train_test_split(X_data, y_data, test_size=0.2)