import pandas as pd

def merge_train_with_client(df_train: pd.DataFrame, df_client: pd.DataFrame) -> pd.DataFrame:
    df_train = df_train.copy()
    df_client = df_client.copy()
    df_train["date"] = df_train["datetime"].dt.date
    df_client["date"] = df_client["date"].dt.date
    segment_cols = ["county", "product_type", "is_business"]
    merge_cols = [*segment_cols, "county_name", "product_name", "date"]
    df_train.set_index(merge_cols, inplace=True)
    df_client.set_index(merge_cols, inplace=True)
    return df_train.join(df_client, on=merge_cols, rsuffix="_client")
