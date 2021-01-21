# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: GaoSuting
mailto: 51185100004@stu.ecnu.edu.cn
"""

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

import warnings

warnings.filterwarnings("ignore")


class DAPS():
    """ Dynamic Self-paced Ensemble (DAPS)

    Parameters
    ----------
    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
    |   The base estimator to fit on self-paced under-sampled subsets of the dataset.
    |   NO need to support sample weighting.
    |   Built-in `fit()`, `predict()`, `predict_proba()` methods are required.

    n_estimators :  integer, optional (default=10)
    |   The number of base estimators in the ensemble.

    k_bins :        integer, optional (default=10)
    |   The number of hardness bins that were used to approximate hardness distribution.

    random_state :  integer / RandomState instance / None, optional (default=None)
    |   If integer, random_state is the seed used by the random number generator;
    |   If RandomState instance, random_state is the random number generator;
    |   If None, the random number generator is the RandomState instance used by
    |   `numpy.random`.

    k: integer/ optional (default = 3)
    |   The number of NearestNeighbors

    Attributes
    ----------
    base_estimator_ : estimator
    |   The base estimator from which the ensemble is grown.

    estimators_ : list of estimator
    |   The collection of fitted base estimators.


    Example:
    ```
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score
    from DAPS import DAPS

    #load data
    data_train = pd.read_csv('Synthetic_ratio30_noise4_train.csv')
    data_test = pd.read_csv('Synthetic_ratio30_noise4_test.csv')
    y_test = data_test.label.values
    X_test = data_test.drop(['label'], axis=1).values
    y_train = data_train.label.values
    X_train = data_train.drop(['label'], axis=1).values

    daps = DAPS(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=10,
            k_bins=10,
            random_state=42,
            k = 4
        ).fit(
            X_train,
            y_train,
        )
    y_pred = daps.predict(X_test)
    print('F1-score: {}'.format(f1_score(y_test, y_pred)))
    ```

    """

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 hardness_func=lambda y_true, y_pred: np.absolute(y_true - y_pred),
                 n_estimators=10,
                 k_bins=10,
                 n=1,
                 evaluate_func=f1_score,
                 random_state=None,
                 k=1):
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self.evaluate_func = evaluate_func
        self._hardness_func = hardness_func
        self._n_estimators = n_estimators
        self._k_bins = k_bins
        self._n = n
        self._random_state = random_state
        self.gamma = 0
        self.alpha = 0.25
        self.k = k

    def _fit_base_estimator(self, X, y, sample_weight):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator_).fit(X, y, sample_weight)

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        np.random.seed(self._random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])
        return X_train, y_train

    def generateSamples(self, X, num):
        if(len(X)==0):
            return None
        X_generate = []
        a = int(num/len(X))
        b = num%len(X)
        while(a>0):
            X_generate.append(X)
            a = a-1
        X_generate.append(X[np.random.choice(np.arange(0, len(X), 1), b, replace=False)])
        return np.concatenate(X_generate,axis = 0)

    def fun(self, i):
        if i >= self._n_estimators/2:
            return (self._n_estimators/2-i)/self._n_estimators
        else:
            return np.log(2*i/self._n_estimators)/np.log(1/np.exp(1))

    def weight_computed(self,pred,gamma):
        c = np.zeros((len(pred), 1))
        for i in range(len(pred)):
            if pred[i] < 0.01:
                c[i] = -((1 - 0.001) ** gamma) * (np.log(0.001))
            elif pred[i] == 1:
                c[i] = -((0.001) ** gamma) * (np.log(0.999))
            else:
                c[i] = -((1 - pred[i]) ** gamma) * (np.log(pred[i]))
        return c.flatten()

    def spe_our(self,X_maj, X_min, X_vetoed_maj, X_vetoed_min, i_estimator):
        """Private function used to perform dynamic self-paced sampling."""
        """compute hardness"""
        hardness_neg = self.predict_proba(X_maj)[:, 1]
        hardness_pos = self.predict_proba(X_min)[:, 0]
        hardness_max = hardness_neg.max() if hardness_neg.max() > hardness_pos.max() else hardness_pos.max()
        hardness_min = hardness_neg.min() if hardness_neg.min() < hardness_pos.min() else hardness_pos.min()
        if len(X_vetoed_min)!=0:
            hardness_vetoed_pos = self.predict_proba(X_vetoed_min)[:, 0]
        else:
            hardness_vetoed_pos = []
        if len(X_vetoed_maj) != 0:
            hardness_vetoed_neg = self.predict_proba(X_vetoed_maj)[:, 1]
        else:
            hardness_vetoed_neg = []

        step = (hardness_max - hardness_min) / self._k_bins

        """split instances to k_bins bins"""

        bins_neg = []
        bins_pos = []
        bins_vetoed_neg = []
        bins_vetoed_pos = []
        ave_contributions = []
        for i_bins in range(self._k_bins):
            idx_neg = (
                    (hardness_neg >= i_bins * step + hardness_min) &
                    (hardness_neg < (i_bins + 1) * step + hardness_min)
            )
            idx_pos = (
                    (hardness_pos >= i_bins * step + hardness_min) &
                    (hardness_pos < (i_bins + 1) * step + hardness_min)
            )
            idx_vetoed_pos = (
                    (hardness_vetoed_pos >= i_bins * step + hardness_min) &
                    (hardness_vetoed_pos < (i_bins + 1) * step + hardness_min)
            )
            idx_vetoed_neg = (
                    (hardness_vetoed_neg >= i_bins * step + hardness_min) &
                    (hardness_vetoed_neg < (i_bins + 1) * step + hardness_min)
            )
            # Marginal samples with highest hardness value -> kth bin
            if i_bins == (self._k_bins - 1):
                idx_neg = idx_neg | (hardness_neg == hardness_max)
                idx_pos = idx_pos | (hardness_pos == hardness_max)
                idx_vetoed_neg = idx_vetoed_neg | (hardness_vetoed_neg == hardness_max)
                idx_vetoed_pos = idx_vetoed_pos | (hardness_vetoed_pos == hardness_max)
            bins_neg.append(X_maj[idx_neg])
            bins_pos.append(X_min[idx_pos])
            bins_vetoed_neg.append(X_vetoed_maj[idx_vetoed_neg])
            bins_vetoed_pos.append(X_vetoed_min[idx_vetoed_pos])

            ave_contributions.append(
                np.concatenate((hardness_neg[idx_neg], hardness_pos[idx_pos]), axis=0).mean())  # 分箱完，计算每个bin中平均hardness

        # Caculate sampling weight
        gx = 1-i_estimator/self._n_estimators
        fx = np.exp(-np.array(ave_contributions))
        bins_weights = fx**gx
        bins_weights[np.isnan(bins_weights)] = 0
        # Caculate sample number from each bin
        sampled_num_pos = len(X_min) + len(X_vetoed_min)
        a = len(X_maj) + len(X_vetoed_maj)
        sampled_num_neg = int(a - i_estimator * (a - sampled_num_pos) / (self._n_estimators - 1))

        neg_bins_samples = sampled_num_neg * bins_weights / bins_weights.sum()
        neg_bins_samples = neg_bins_samples.astype(int)

        pos_bins_samples = sampled_num_pos * bins_weights / bins_weights.sum()
        pos_bins_samples = pos_bins_samples.astype(int)


        #compute the weight of vetoed instances
        beta = self.fun(i_estimator)
        sample_weights = np.exp(-(np.array(ave_contributions) + beta))
        sample_weights[np.isnan(sample_weights)] = 0

        j = 0
        neg_bins_samples_vetoed = []
        pos_bins_samples_vetoed = []
        
        ir_pos = len(X_vetoed_min) / (len(X_min) + len(X_vetoed_min))
        ir_neg = len(X_vetoed_maj) / (len(X_maj) + len(X_vetoed_maj))
        
        while j < self._k_bins:
            neg_bins_samples_vetoed.append(
                int(neg_bins_samples[j] * ir_neg)
                if len(bins_neg[j]) > 0 else 0)
            pos_bins_samples_vetoed.append(
                int(pos_bins_samples[j] * ir_pos)
                if len(bins_pos[j]) > 0 else 0)
            j = j + 1


        #sampling
        sampled_neg = []
        weight_neg = []
        for i_bins in range(self._k_bins):
            if min(len(bins_neg[i_bins]), neg_bins_samples[i_bins]) > 0:
                if len(bins_neg[i_bins]) >= neg_bins_samples[i_bins]:  # bin中样本数多于要采样的数量
                    idx = np.random.choice(
                        len(bins_neg[i_bins]),
                        neg_bins_samples[i_bins],
                        replace=False)
                    sampled_neg.append(bins_neg[i_bins][idx])
                    weight_neg.append([1] * neg_bins_samples[i_bins])
                else:
                    if (len(bins_neg[i_bins]) == 0):
                        continue
                    else:
                        sampled_neg.append(bins_neg[i_bins])
                        sampled_neg.append(
                            self.generateSamples(bins_neg[i_bins],
                                          neg_bins_samples[i_bins] - len(bins_neg[i_bins])))

                        weight_neg.append([1] * neg_bins_samples[i_bins])

                if min(len(bins_vetoed_neg[i_bins]), neg_bins_samples_vetoed[i_bins]) > 0:
                    np.random.seed(None)
                    idx = np.random.choice(
                        len(bins_vetoed_neg[i_bins]),
                        min(len(bins_vetoed_neg[i_bins]), neg_bins_samples_vetoed[i_bins]),
                        replace=False)
                    sampled_neg.append(bins_vetoed_neg[i_bins][idx])
                    # sample_weight
                    weight_neg.append(
                        [sample_weights[i_bins]] * min(len(bins_vetoed_neg[i_bins]),
                                                            neg_bins_samples_vetoed[i_bins]))

        weight_neg = np.concatenate(weight_neg, axis=0)
        X_train_maj = np.concatenate(sampled_neg, axis=0)
        y_train_maj = np.full(X_train_maj.shape[0], 0)

        sampled_pos = []
        weight_pos = []
        for i_bins in range(self._k_bins):
            if min(len(bins_pos[i_bins]), pos_bins_samples[i_bins]) > 0:
                if len(bins_pos[i_bins]) >= pos_bins_samples[i_bins]:  # bin中样本数多于要采样的数量
                    idx = np.random.choice(
                        len(bins_pos[i_bins]),
                        pos_bins_samples[i_bins],
                        replace=False)
                    sampled_pos.append(bins_pos[i_bins][idx])
                    weight_pos.append([1] * pos_bins_samples[i_bins])
                else:
                    if (len(bins_pos[i_bins]) == 0):
                        continue
                    else:
                        sampled_pos.append(bins_pos[i_bins])
                        sampled_pos.append(
                            self.generateSamples(bins_pos[i_bins], pos_bins_samples[i_bins] - len(bins_pos[i_bins])))

                        weight_pos.append([1] * pos_bins_samples[i_bins])

                if min(len(bins_vetoed_pos[i_bins]), pos_bins_samples_vetoed[i_bins]) > 0:
                    np.random.seed(None)
                    idx = np.random.choice(
                        len(bins_vetoed_pos[i_bins]),
                        min(len(bins_vetoed_pos[i_bins]), pos_bins_samples_vetoed[i_bins]),
                        replace=False)
                    sampled_pos.append(bins_vetoed_pos[i_bins][idx])
                    # sample_weight
                    weight_pos.append(
                        [sample_weights[i_bins]] * min(len(bins_vetoed_pos[i_bins]), pos_bins_samples_vetoed[i_bins]))

        weight_pos = np.concatenate(weight_pos, axis=0)
        X_train_min = np.concatenate(sampled_pos, axis=0)
        y_train_min = np.full(X_train_min.shape[0], 1)

        X_train = np.concatenate([X_train_maj, X_train_min])
        y_train = np.concatenate([y_train_maj, y_train_min])

        sample_weight = np.concatenate([weight_neg, weight_pos])

        return X_train, y_train, sample_weight

    def fit(self,X , y, label_maj=0, label_min=1):
        """Build a self-paced ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels).

        label_maj : int, bool or float, optional (default=0)
            The majority class label, default to be negative class.

        label_min : int, bool or float, optional (default=1)
            The minority class label, default to be positive class.

        Returns
        ------
        self : object
        """

        self.estimators_ = []
        # Initialize by spliting majority / minority set

        # Random under-sampling in the 1st round (cold start)
        # X_train, y_train = self._random_under_sampling(
        #     X_maj, y_maj, X_min, y_min)

        X_train, y_train, X_vetoed, y_vetoed = self.vetoed_identify(X, y, self.k)

        # sample_weight_neg = np.full(X[y == 0].shape[0], 1)
        # sample_weight_pos = np.full(X[y == 1].shape[0], 1)
        # # sample_weight_neg = sample_weight_neg * sum(sample_weight_pos) / sum(sample_weight_neg)
        # sample_weight = np.concatenate([sample_weight_neg, sample_weight_pos])


        sample_weight = np.full(X_train.shape[0],1)

        X_maj = X_train[y_train == label_maj];
        y_maj = y_train[y_train == label_maj]
        X_min = X_train[y_train == label_min];
        y_min = y_train[y_train == label_min]

        X_vetoed_maj = X_vetoed[y_vetoed == label_maj];
        X_vetoed_min = X_vetoed[y_vetoed == label_min]

        # X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
        # print(Counter(y_res))

        # sample_weight = [1] * len(X_train)

        self.estimators_.append(
            self._fit_base_estimator(
                X_train, y_train, sample_weight))

        # Loop start
        for i_estimator in range(1, self._n_estimators):
            X_train, y_train, sample_weight = self.spe_our(
                X_maj, X_min, X_vetoed_maj, X_vetoed_min, i_estimator, )
            self.estimators_.append(
                self._fit_base_estimator(
                    X_train, y_train, sample_weight))

        return self

    def vetoed_identify(self,X, y, k):
        X_maj = X[y == 0];
        X_min = X[y == 1];
        y_maj = y[y == 0];
        y_min = y[y == 1]
        knn = NearestNeighbors(n_neighbors=k).fit(X)
        inx = knn.kneighbors()[1][y == 0]
        index = []
        for i in range(len(X_maj)):
            if (y[inx[i]].sum() > 0):
                index.append(i)
        X_maj_temp = np.delete(X_maj, index, axis=0)
        y_maj_temp = np.delete(y_maj, index, axis=0)
        X_temp = np.concatenate((X_maj_temp, X_min), axis=0)
        y_temp = np.concatenate((y_maj_temp, y_min), axis=0)

        X_noise_maj = X_maj[index]
        y_noise_maj = y_maj[index]

        knn = NearestNeighbors(n_neighbors=k).fit(X_temp)
        inx = knn.kneighbors()[1][y_temp == 1]
        index = []
        for i in range(len(X_temp[y_temp == 1])):
            if (y_temp[inx[i]].sum() == 0):
                index.append(i)
        X_min_temp = np.delete(X_min, index, axis=0)
        y_min_temp = np.delete(y_min, index, axis=0)
        X_sam = np.concatenate((X_maj_temp, X_min_temp), axis=0)
        y_sam = np.concatenate((y_maj_temp, y_min_temp), axis=0)

        X_noise_min = X_min[index]
        y_noise_min = y_min[index]

        X_noise = np.concatenate((X_noise_maj, X_noise_min), axis=0)
        y_noise = np.concatenate((y_noise_maj, y_noise_min), axis=0)

        return X_sam, y_sam, X_noise, y_noise

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        y_pred = np.array(
            [model.predict_proba(X)[:,1] for model in self.estimators_]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_pred_binarized

    def score(self, X, y):
        """Returns the average precision score (equivalent to the area under
        the precision-recall curve) on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Average precision of self.predict_proba(X)[:, 1] wrt. y.
        """
        return sklearn.metrics.f1_score(
            y, self.predict_proba(X)[:, 1])