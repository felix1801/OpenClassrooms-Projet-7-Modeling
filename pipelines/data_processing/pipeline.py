import pandas as pd

from pipelines.data_processing.feature_engineering import (
    create_final_df,
    application_train_test,
    bureau_and_balance,
    previous_applications,
    installments_payments,
    pos_cash,
    credit_card_balance
)

def run_data_processing() -> pd.DataFrame:
    df = application_train_test()
    bureau = bureau_and_balance()
    prev = previous_applications()
    ins = installments_payments()
    pos = pos_cash()
    cc = credit_card_balance()
    df = create_final_df(df, bureau, prev, pos, ins, cc)
    return df
