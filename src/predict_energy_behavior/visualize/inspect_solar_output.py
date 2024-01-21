from plotly.subplots import make_subplots
from IPython.display import display
import pandas as pd
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import json
import predict_energy_behavior.data.read
import predict_energy_behavior.data.constants
import predict_energy_behavior.features
from predict_energy_behavior.models.production.solar_output_regression import SolarOutputRegression, Parameter
import numpy as np
import polars as pl

def normalize(xs):
    return (xs - xs.min()) / (xs.max() - xs.min())

def inspect_solar_output_predictions(estimator: SolarOutputRegression, df_features: pd.DataFrame, q=0.75):
    
    df = df_features.copy()
    predictions = estimator.predict(df)
    
    df["preds"] = predictions
    df["difference"] = np.abs(predictions - df["target"])

    df_g_mae = df.groupby(["county_name", "product_name", "is_business"])["difference"].mean().reset_index().rename({"difference": "mae"}, axis=1)
    q_val_g = df_g_mae["mae"].quantile(q)

    low_mae_groups = df_g_mae.loc[df_g_mae["mae"] >= q_val_g]
    display(low_mae_groups)

    worst_group = low_mae_groups.sort_values("mae", ascending=False).iloc[0]

    df = df.loc[
        (df["county_name"]==worst_group["county_name"]) &
        (df["product_name"]==worst_group["product_name"]) &
        (df["is_business"]==worst_group["is_business"]) 
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=df["datetime"], y=df["preds"], opacity=1, name="predictions"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["target"], name="labels", opacity=0.5), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=normalize(df["snowfall"]),
        name="Snow"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=normalize(df["temperature"]),
        name="temperature",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=normalize(df["rain"]), 
            name="rain"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=normalize(df["windspeed_10m"]), 
            name="wind"
    ), row=2, col=1)


    fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=df["direct_solar_radiation"], 
            name="direct rad"
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=df["diffuse_radiation"], 
            name="diffuse rad"
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=df["shortwave_radiation"], 
            name="total rad"
    ), row=3, col=1)

    for cc in ["total", "high", "mid", "low"]:
        fig.add_trace(go.Scatter(
            x=df["datetime"], 
            y=df[f"cloudcover_{cc}"], 
            name=f"cloudcover {cc}",
    ), row=4, col=1)

    fig.update_layout(height=800)

    fig.show()