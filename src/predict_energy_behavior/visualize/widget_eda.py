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
            df_train: pd.DataFrame
    ):
        self.__df_train = df_train
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
        figure_widget = go.FigureWidget(data=[trace_con, trace_prod])
        self.__widget_figure = figure_widget

        self.__widget_select_unit.observe(self._handle_update_figure_widget, "value")
        self.__widget_select_private.observe(self._handle_update_select_unit_values)

        self.__widget = widgets.VBox([widget_container, figure_widget])

    def _handle_update_select_unit_values(self, change):
        if self.__widget_select_private.value == "private":
            self.__widget_select_unit.options = self.__private_units
        elif self.__widget_select_private.value == "business":
            self.__widget_select_unit.options = self.__business_units

    def _handle_update_figure_widget(self, change):
        unit_series = self.__df_train.loc[self.__df_train["prediction_unit_id"] == self.__widget_select_unit.value]

        unit_consumption_series = unit_series.loc[unit_series["is_consumption"]]
        unit_production_series = unit_series.loc[~unit_series["is_consumption"]]

        figure_widget = self.__widget_figure

        with figure_widget.batch_update():
            figure_widget.data[0].x = unit_consumption_series["datetime"]
            figure_widget.data[0].y = unit_consumption_series["target"]
            figure_widget.data[1].x = unit_production_series["datetime"]
            figure_widget.data[1].y = unit_production_series["target"]

    def display(self):
        display(self.__widget)
        self._handle_update_figure_widget({})