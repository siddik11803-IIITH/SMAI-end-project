import features as fe
import numpy as np
from utils import *
from classifier import *
import pickle
import cascade as cs


def create_cascade(N):
    X_train_fe,y_train = Create_data(N)
    Strong_Classifiers,No_of_samples_per_layer,No_of_positive_samples_per_layer,No_of_negative_samples_per_layer = np.array(cs.Train_Cascade(X_train_fe, y_train))
    return Strong_Classifiers 

cascade = create_cascade(1000)
pickle.dump(cascade, open("cascade.pkl", "wb"))


