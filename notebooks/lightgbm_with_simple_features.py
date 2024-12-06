# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import os
import pickle
import re
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/application_train_test.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    # Read data and merge
    df = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    
    df = pd.concat([df, test_df]).reset_index(drop=True)
    
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()

    df.to_csv(filepath, index=False)
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/bureau_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    bureau = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    bureau_agg.to_csv(filepath, index=False)
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/prev_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    prev = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()

    prev_agg.to_csv(filepath, index=False)
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/pos_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    pos = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()

    pos_agg.to_csv(filepath, index=False)
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/ins_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    ins = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()

    ins_agg.to_csv(filepath, index=False)
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/02_intermediate/cc_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df

    cc = pd.read_csv('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/01_raw/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()

    cc_agg.to_csv(filepath, index=False)
    return cc_agg

# Fonction pour convertir les colonnes objet avec 1 à 2 valeurs uniques en booléen ou int
def convert_object_columns(df):
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # Remplacer les NaN par la valeur la plus représentée
        most_frequent_value = df[col].mode()[0]
        df[col] = df[col].fillna(most_frequent_value)

        unique_values = df[col].nunique()
        if unique_values == 1:
            df[col] = df[col].astype('bool')
        elif unique_values == 2:
            df[col] = df[col].astype('category').cat.codes.astype('int8')
    return df

def handle_data_types(df):
    df.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in df.columns]
    
    # Convertir les colonnes objet avec 1 à 2 valeurs uniques en booléen ou int
    df = convert_object_columns(df)

    inf_cols_mask = np.isinf(df).any()
    inf_cols = df.columns.to_series()[inf_cols_mask].tolist()
    # Replace inf values with max
    for col in inf_cols:
        if col in df.columns:  # Check if the column exists in the DataFrame
            max_value = df[col][df[col] != np.inf].max()  # Get the max value excluding inf
            df[col] = df[col].replace([np.inf, -np.inf], max_value)  # Replace inf values

    return df

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    auc_max = 0

    # Remove special characters from feature names
    # df.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in df.columns]

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
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
        # for col in train_x.columns:
        #     if train_x[col].dtype == 'object':
        #         # Convert to numeric and handle exceptions
        #         try:
        #             train_x[col] = pd.to_numeric(train_x[col], errors='raise')
        #         except ValueError as e:
        #             train_x[col] = train_x[col].astype(str).str.strip().astype(float)

        #         try:
        #             valid_x[col] = pd.to_numeric(valid_x[col], errors='raise')
        #         except ValueError as e:
        #             valid_x[col] = valid_x[col].astype(str).str.strip().astype(float)

        #         try:
        #             test_df.loc[:, col] = pd.to_numeric(test_df[col], errors='raise')
        #         except ValueError as e:
        #             test_df.loc[:, col] = test_df[col].astype(str).str.strip().astype(float)

        #         # Fill NaN values and convert to float
        #         train_x[col] = train_x[col].fillna(0).astype(float)
        #         valid_x[col] = valid_x[col].fillna(0).astype(float)
        #         test_df.loc[:, col] = test_df[col].fillna(0).astype(float)
                

        # Diagnose the types of the columns after conversion
        # columns_to_remove = []
        # for col in train_x.columns:
        #     train_type = train_x[col].dtype
        #     valid_type = valid_x[col].dtype
        #     test_type = test_df[col].dtype

        #     if train_type == 'object' or valid_type == 'object' or test_type == 'object':
        #         print(f"Column {col} has inconsistent types and will be removed.")
        #         columns_to_remove.append(col)

        # # Remove columns with inconsistent types from all DataFrames
        # train_df.drop(columns=columns_to_remove, inplace=True)
        # valid_x.drop(columns=columns_to_remove, inplace=True)
        # test_df.drop(columns=columns_to_remove, inplace=True)

        # print(f"Columns removed: {columns_to_remove}")

        # {
        #     'CC_NAME_CONTRACT_STATUS_Active_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Active_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Approved_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Approved_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Completed_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Completed_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Demand_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Demand_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Refused_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Refused_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Sentproposal_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Sentproposal_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Signed_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_Signed_MAX': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_nan_MIN': dtype('O'), 
        #     'CC_NAME_CONTRACT_STATUS_nan_MAX': dtype('O')
        #     }

        # Ensure that train_x, valid_x, and test_df have the same columns
        # common_columns = train_x.columns.intersection(valid_x.columns).intersection(test_df.columns)
        # train_x = train_x[common_columns]
        # valid_x = valid_x[common_columns]
        # test_df = test_df[common_columns]
                
        # LightGBM parameters found by Bayesian optimization
        clf = lightgbm.LGBMClassifier(
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
            # silent=-1,
            # verbose=-1, 
            )
        
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc',
            callbacks=[
                lightgbm.early_stopping(
                    stopping_rounds=200,
                    first_metric_only=True,  # Only use AUC metric for early stopping
                    verbose=True,
                    min_delta=0.01  # Minimum improvement needed to continue training
                )
            ]
        )

        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, auc))
        if auc > auc_max:
            with open('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/06_models/lightgbm_model.pkl', 'wb') as file:
                pickle.dump(clf, file)
            auc_max = auc
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # print(f"Model features: {[print(feat) for feat in feats]}")
    

    # Write submission file and plot feature importance
    # if not debug:
        # test_df['TARGET'] = sub_preds
        # test_df[['TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/08_reporting/lgbm_importances01.png')


def main(debug = False):
    filepath = 'C:/Users/Z478SG/Desktop/Ecole/OpenClassrooms-Projet-7/modeling/data/03_primary/df_agg.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, low_memory=False)

    else:
        num_rows = 10000 if debug else None
        df = application_train_test(num_rows)
        with timer("Process bureau and bureau_balance"):
            bureau = bureau_and_balance(num_rows)
            print("Bureau df shape:", bureau.shape)
            df = df.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with timer("Process previous_applications"):
            prev = previous_applications(num_rows)
            print("Previous applications df shape:", prev.shape)
            df = df.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with timer("Process POS-CASH balance"):
            pos = pos_cash(num_rows)
            print("Pos-cash balance df shape:", pos.shape)
            df = df.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with timer("Process installments payments"):
            ins = installments_payments(num_rows)
            print("Installments payments df shape:", ins.shape)
            df = df.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with timer("Process credit card balance"):
            cc = credit_card_balance(num_rows)
            print("Credit card balance df shape:", cc.shape)
            df = df.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()

        with timer("Process data types handling"):
            df = handle_data_types(df)

        df.to_csv(filepath, index=False)

    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()