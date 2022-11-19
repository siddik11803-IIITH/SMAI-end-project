from sklearn.tree import DecisionTreeClassifier
import numpy as np




def Adaboost_train(iter, X, y):
    weights = init_weights(y)
    errors_features, clf_features = precompute(X, y)
    selected_idx, alphas = train_helper(iter, errors_features, weights)
    return selected_idx, alphas, clf_features


def Adaboost_cassify(x, model):
    (selected_idx, alphas, classifiers) = model
    lables = [classifiers[i].predict(x[selected_idx[i]]) for i in len(selected_idx)]
    if alphas.dot(lables) >= (alphas.sum())/2:
        label = 1
    else:
        label = 0
    return label


def Adaboost_accuracy(X_test, y_test, model):
    wrong = correct = 0
    for index in range(len(X_test)):
        label = Adaboost_cassify(X_test[index], model)
        if(label == y_test[index]):
            correct += 1
        else:
            wrong += 1
    return correct/(correct + wrong)


def init_weights(y):
    weights = np.zeros(y.shape)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    weights[y == 1] = 1 / (2 * pos)
    weights[y == 0] = 1 / (2 * neg)
    return weights


def precompute(x_train, y_train):
    n_features = x_train.shape[1]
    errors_features = [] # errors_features[j][i] = |h_j(x_i) - y_i|
    clf_features = []
    for feature in range(n_features):
        train_data = x_train[:, feature]
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(train_data, y_train)
        errors_features.append(((np.abs(clf.predict(train_data) - y_train))))
        clf_features.append(clf)
    return np.array(errors_features), clf_features


def train_helper(iter, errors_features, weights):
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