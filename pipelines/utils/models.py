import re
import gc

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from pipelines.utils.helpers import display_importances


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Remove special characters from feature names
    df.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in df.columns]

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        # used for stratified cross validation (i.e. each fold has the same proportion of target variable as the entire dataset)
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        # used to split the data into training and validation sets while evaluating a model (i.e. cross-validation) 
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # Convert relevant columns to the correct data type
        for col in train_x.columns:
            if train_x[col].dtype == 'object':
                train_x[col] = train_x[col].astype(float)
                valid_x[col] = valid_x[col].astype(float)
                test_df.loc[:, col] = test_df[col].astype(float)


        # LightGBM parameters found by Bayesian optimization
        # Model used to predict the probability of a binary outcome (0 or 1) given a set of independent variables (features) 
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            # verbose=-1, 
            )
        
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc')
        
        # Predict the probability of a binary outcome (0 or 1) given a set of independent variables (features)
        # OOF stands for Out-Of-Fold predictions. In K-Fold Cross-Validation, oof_preds are the predictions made on the validation folds. For each fold, the model is trained on the training folds and makes predictions on the validation fold. These predictions are collected and combined to form the oof_preds. They provide an estimate of the model's performance on unseen data
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        # Sub refers to submission predictions. These are the predictions made on the test set or the data for which you want to make final predictions. In a competition or real-world application, sub_preds would be the predictions you submit for evaluation or use to make decisions about the data you are working with 
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv("../data/07_model_output/test_df_predicted.csv", index= False)
    display_importances(feature_importance_df)
    return feature_importance_df