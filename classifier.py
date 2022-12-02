from sklearn.tree import DecisionTreeClassifier
import numpy as np
from multiprocessing.pool import Pool
import multiprocessing
from functools import partial

class AdaBoostClassifier(object):
    def __init__(self):
        self.selected_features = []
        self.aplhas = []
        self.weakclassifiers = []
        self.fitted = False
        self.x_trainp = None
        self.y_trainp = None

    def fit(self, X, y, iter,c):
        weights = self.init_weights(y)
        
        self.x_trainp = X
        self.y_trainp = y
        #errors_features, self.weakclassifiers = self.precompute(X, y)
        errors_features, self.weakclassifiers = self.precompute_parallelized_with_multiprocessing(X, y,c)
        self.selected_features, self.aplhas = self.train_helper(iter, errors_features, weights)
        self.fitted = True

    def predict(self, X_test):
        if(self.fitted == False):
            print("Model not fitted")
            return
        else:
            y_pred = np.zeros(len(X_test))
            for index in range(len(y_pred)):
                lables = []
                for feature in self.selected_features:
                    label = self.weakclassifiers[feature].predict(X_test[index][feature].reshape(1,1))
                    lables.append(label)
                if self.aplhas.dot(lables) >= (self.aplhas.sum())/2:
                    y_pred[index] = 1
                else:
                    y_pred[index] = 0
            return y_pred
        
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)/len(y_test)

    def init_weights(self, y):
        weights = np.zeros(y.shape)
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        weights[y == 1] = 1 / (2 * pos)
        weights[y == 0] = 1 / (2 * neg)
        return weights

    def precompute(self, x_train, y_train):
        n_features = x_train.shape[1]
        errors_features = [] # errors_features[j][i] = |h_j(x_i) - y_i|
        clf_features = []
        for feature in range(n_features):
            train_data = x_train[:, feature]
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(train_data.reshape(len(x_train), 1), y_train)
            errors_features.append(((np.abs(clf.predict(train_data.reshape(len(x_train), 1)) - y_train))))
            clf_features.append(clf)
        return np.array(errors_features), clf_features

    def precompute_single_clf_feature(self,feature):
        '''computes single errors and clf for a single feature(int)'''
        train_data = self.x_trainp[:, feature]
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(train_data.reshape(len(self.x_trainp), 1), self.y_trainp)
        
        return clf
    # def precompute_parallelized_with_threads(self,x_train,y_train):
    #     n_features = x_train.shape[1]
    #     errors_features = [] # errors_features[j][i] = |h_j(x_i) - y_i|
    #     clf_features = []
    #     thread = threading.Thread(target=self.precompute_single_clf_feature)
    #     thread.start()
    #     return np.array(errors_features), clf_features
    def err_c(self,clf,feature,):
        train_data = self.x_trainp[:, feature]
        erf = ((np.abs(clf.predict(train_data.reshape(len(self.x_trainp), 1)) - self.y_trainp)))
        return erf

    def precompute_parallelized_with_multiprocessing(self,x_train,y_train,c):
      
        n_features = x_train.shape[1]

        l = list(range(n_features))
        pool1 = multiprocessing.Pool(c)
        clf_features =  pool1.map(self.precompute_single_clf_feature,l)
        pool1.close()
        pool1.join()
        pool2 = multiprocessing.Pool(c)
        errors_features =  pool2.map(partial(self.err_c),clf_features,l)
        pool2.close()
        pool2.join()
        
        
        return np.array(errors_features),clf_features

    def train_helper(self, iter, errors_features, weights):
        #weight_obs = []
        selected_idx = []
        betas = []
        for _ in range(iter):
            weights /= sum(weights)
            #weight_obs.append(weights)
            round_errors = np.dot(errors_features, weights)

            min_idx = np.argmin(round_errors)
            e_t = round_errors[min_idx]
            errors = errors_features[min_idx] # contains 1 or 0.
            beta_t = (e_t)/(1 - e_t)
            
            weights = weights * (beta_t ** (1 - errors))
            
            selected_idx.append(min_idx)
            betas.append(beta_t)
        
        alphas =  -np.log(betas)
        return selected_idx, alphas 