# DAPS
## Synthetic data
Synthetic data is divided into training set and independent test set. The number following "ratio" represents the imbalance ratio and the number following "sig" represents σ<sup>2</sup><sub>p</sub> , which control the degree of class overlap. 

## Run
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

## Paper information
Zhou F., Gao S.T., Ni L. Pavlovski M.,  Dong Q., Obradovic Z., Qian W., "Dynamic Self-paced Sampling Ensemble for Highly Imbalanced and Class-overlapped Data Classification"., Data Mining and Knowledge Discovery (DMKD-2022) , May, 2022.
