import features as fe
import numpy as np
from utils import *
from classifier import *

def Cascade_Classifier_predict(X_test, y_test, Strong_Classifiers):
    y_preds = []
    for i in range((len(Strong_Classifiers))):
        y_pred = Strong_Classifiers[i].predict(X_test)
        y_preds.append(y_pred)

    ans = np.zeros(len(y_test)) 
    for i in range(len(y_test)):
        is_one = True
        for j in range(len(Strong_Classifiers)):
            if(y_preds[j][i] == 0):
                is_one = False
                break
        if(is_one == True):
            ans[i] = 1
        else:
            ans[i] = 0
    return ans


def Cascade_Classifier_predict_Img(X_test_img, y_test, Strong_Classifiers):
    y_preds = []
    for i in range((len(Strong_Classifiers))):
        y_pred = Strong_Classifiers[i].predict_img(X_test_img)
        y_preds.append(y_pred)

    ans = np.zeros(len(y_test)) 
    for i in range(len(y_test)):
        is_one = True
        for j in range(len(Strong_Classifiers)):
            if(y_preds[j][i] == 0):
                is_one = False
                break
        if(is_one == True):
            ans[i] = 1
        else:
            ans[i] = 0
    return ans


def Cascade_Classifier_predict_single_Img(X_test_img, y_test, Strong_Classifiers):
    assert(X_test_img.ndim == 2)
    X_test_simage = X_test_img
    X_test_simage.reshape(1, X_test_img.shape[0], X_test_img.shape[1])

    return Cascade_Classifier_predict_Img(X_test_simage, y_test, Strong_Classifiers)[0]


def Cascade_Classifier(X_test,y_test, Strong_Classifiers):
    y_pred = Cascade_Classifier_predict(X_test, y_test, Strong_Classifiers)
    false_positive_rate = np.sum((y_pred == 1) & (y_test == 0))/np.sum(y_test == 0)
    detection_rate = np.sum((y_pred == 1)& (y_test == 1))/np.sum(y_test == 1)
    return false_positive_rate, detection_rate



def Create_validation_sets(X_all, y_all):
    X_train, X_valid, y_train, y_valid = Split_Data(X_all, y_all)
    return X_train, X_valid, y_train, y_valid



def Train_Cascade(X_train_all, y_train_all):    
    F = [0.6]
    D = [0.9]
    f = 0.999
    d = 0.99
    F_target = 0.01
    Threshold_retention = 0.99
    
    F_new = F[-1]
    Strong_Classifiers = []
    no_outermost_loops = 0
    X_train, X_valid, y_train, y_valid = Create_validation_sets(X_train_all, y_train_all)
    No_of_samples_per_layer = []
    No_of_positive_samples_per_layer = []
    No_of_negative_samples_per_layer = []
    while F_new > F_target and no_outermost_loops < 5:
        no_outermost_loops += 1
        a = AdaBoostClassifier()
        F_new = F[-1]
        no_of_features = 0       
        No_of_samples_per_layer.append(len(X_train))
        while F_new > f*F[-1] and no_of_features < 50:          #### Hard coded
            no_of_features += 1       
            a.fit(X_train, y_train, no_of_features)
            Strong_Classifiers.append(a)
            F_new, D_new = Cascade_Classifier(X_valid, y_valid, Strong_Classifiers)

            no_innermost_iter = 0
            while(no_innermost_iter < 50):            #### Hard coded
                no_innermost_iter += 1
                a = Strong_Classifiers[-1]
                Strong_Classifiers[-1].threshold = Strong_Classifiers[-1].threshold*Threshold_retention
                F_new, D_new = Cascade_Classifier(X_valid, y_valid, Strong_Classifiers)
               # print("L1:", len(Strong_Classifiers), " L2:", no_of_features, " L3:", no_innermost_iter, " F_new: ", F_new, " F_tar:  ", f*F[-1], " D_new:", D_new, "D_expected:", d*D[-1], " Threshold:", a.threshold)
                if D_new > d*D[-1]:       #### Hard coded          
                    break     
                if a.threshold < 1e-10:   #### Hard coded
                    break
                
            Strong_Classifiers.pop()

        F.append(F_new)
        D.append(D_new)
        Strong_Classifiers.append(a)  
        y_pred = a.predict(X_train)
        # number of 1s in y_pred
        no_of_ones = np.sum(y_pred == 1)
        No_of_positive_samples_per_layer.append(no_of_ones)
        # number of 0s in y_pred
        no_of_zeros = np.sum(y_pred == 0)
        No_of_negative_samples_per_layer.append(no_of_zeros)
        X_train = X_train[y_pred == 1] 
        y_train = y_train[y_pred == 1]
        if(len(y_train) < 10):
            break
    return Strong_Classifiers, No_of_samples_per_layer,No_of_positive_samples_per_layer,No_of_negative_samples_per_layer

