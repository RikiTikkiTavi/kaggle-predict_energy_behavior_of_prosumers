from IPython.display import display
import pandas as pd
from ipywidgets import widgets
import plotly.graph_objs as go


class EnergyBehaviorInteractiveEDAWidget:
    __widget_select_unit: widgets.Dropdown
    __widget_select_private: widgets.Dropdown
    __widget_figure: go.FigureWidget
    __widget_container: widgets.Widget

    __df_train: pd.DataFrame

    __business_units: pd.Series
    __private_units: pd.Series

    def __init__(
            self,
            df_train: pd.DataFrame,
            df_el_prices: pd.DataFrame,
            df_client: pd.DataFrame
    ):
        self.__df_train = df_train
        self.__df_el_prices = df_el_prices
        self.__df_client = df_client
        self.__business_units = df_train.loc[df_train["is_business"]]["prediction_unit_id"].unique()
        self.__private_units = df_train.loc[~df_train["is_business"]]["prediction_unit_id"].unique()

        self.__widget_select_unit = widgets.Dropdown(
            options=list(df_train["prediction_unit_id"].unique()),
            value=0,
            description='Unit ID:',
        )

        self.__widget_select_private = widgets.Dropdown(
            options=["private", "business"],
            value="private",
            description='Client type: '
        )

        trace_con = go.Scatter(name="consumption")
        trace_prod = go.Scatter(name="production", opacity=0.6)

        widget_container = widgets.HBox([self.__widget_select_private, self.__widget_select_unit])
        self.__widget_figure = go.FigureWidget(data=[trace_con, trace_prod])

        trace_el_prices = go.Scatter(name="electricity_prices", x=df_el_prices["forecast_date"], y=df_el_prices["euros_per_mwh"])
        self.__widget_figure_electricity_prices = go.FigureWidget(data=[trace_el_prices])
        self.__widget_figure_electricity_prices.update_layout(
            title="Electricity prices",
            yaxis_title="Price",
        )

        trace_client_capacity = go.Scatter(name="client_capacity")
        self.__widget_figure_client_capacity = go.FigureWidget(data=[trace_client_capacity])
        self.__widget_figure_client_capacity.update_layout(
            title="Client",
            yaxis_title="Installed capacity",
        )

        self.__widget_select_unit.observe(self._handle_update_figure_widget, "value")
        self.__widget_select_private.observe(self._handle_update_select_unit_values)

        self.__widget = widgets.VBox([widget_container, self.__widget_figure, self.__widget_figure_electricity_prices, self.__widget_figure_client_capacity])

    def _handle_update_select_unit_values(self, change):
        if self.__widget_select_private.value == "private":
            self.__widget_select_unit.options = self.__private_units
        elif self.__widget_select_private.value == "business":
            self.__widget_select_unit.options = self.__business_units

    def _handle_update_figure_widget(self, change):
        unit_series = self.__df_train.loc[self.__df_train["prediction_unit_id"] == self.__widget_select_unit.value]

        unit_consumption_series = unit_series.loc[unit_series["is_consumption"]]
        unit_production_series = unit_series.loc[~unit_series["is_consumption"]]

        unit_installed_capacity = self.__df_client.loc[
            (self.__df_client["county"] == unit_series.iloc[0]["county"]) &
            (self.__df_client["is_business"] == unit_series.iloc[0]["is_business"]) &
            (self.__df_client["product_type"] == unit_series.iloc[0]["product_type"])
        ]

        figure_widget = self.__widget_figure

        with figure_widget.batch_update():
            figure_widget.data[0].x = unit_consumption_series["datetime"]
            figure_widget.data[0].y = unit_consumption_series["target"]
            figure_widget.data[1].x = unit_production_series["datetime"]
            figure_widget.data[1].y = unit_production_series["target"]

        with self.__widget_figure_client_capacity.batch_update():
            self.__widget_figure_client_capacity.data[0].x = unit_installed_capacity["date"]
            self.__widget_figure_client_capacity.data[0].y = unit_installed_capacity["installed_capacity"]

    def display(self):
        display(self.__widget)
        self._handle_update_figure_widget({})