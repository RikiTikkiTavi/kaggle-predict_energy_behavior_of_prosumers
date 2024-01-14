from itertools import chain
from pathlib import Path
from typing import NamedTuple
import pandas as pd

import requests
import predict_energy_behavior.data.constants as constants
from dataclasses import dataclass
import polars as pl
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd
from geopy.geocoders import Nominatim

@dataclass
class RawDatasets:
    data: pl.DataFrame
    client: pl.DataFrame
    gas: pl.DataFrame
    electricity: pl.DataFrame
    weather_forecast: pl.DataFrame
    weather_hist: pl.DataFrame
    weather_station_to_county: pl.DataFrame


def read_datasets_from_folder(path: Path) -> RawDatasets:
    return RawDatasets(
        data=pl.read_csv(path / "train.csv", columns=constants.data_cols, try_parse_dates=True),
        client=pl.read_csv(path / "client.csv", columns=constants.client_cols, try_parse_dates=True),
        gas=pl.read_csv(path / "gas_prices.csv", columns=constants.gas_prices_cols, try_parse_dates=True),
        electricity=pl.read_csv(path / "electricity_prices.csv", columns=constants.electricity_prices_cols, try_parse_dates=True),
        weather_forecast=pl.read_csv(path / "forecast_weather.csv", columns=constants.forecast_weather_cols, try_parse_dates=True),
        weather_hist= pl.read_csv(path / "historical_weather.csv", columns=constants.historical_weather_cols, try_parse_dates=True),
        weather_station_to_county=pl.read_csv(path / "weather_station_to_county_mapping.csv", columns=constants.location_cols, try_parse_dates=True)
    )


def order_ways(list_list_points):
    #list_first_points = [list_points[0] for list_points in list_list_points]
    #list_last_points = [list_points[-1] for list_points in list_list_points]
    ordered_list_list_points = []
    first_list_point = list_list_points[0]
    list_points = list_list_points[0]
    last_point = 0
    while (last_point != first_list_point[-1]):
        ordered_list_list_points.append(list_points)
        list_list_points.remove(list_points)
        list_first_points = [list_points[0] for list_points in list_list_points]
        list_last_points = [list_points[-1] for list_points in list_list_points]
        #print(list_points[0])
        #print(list_points[-1])
        #print()
        last_point = list_points[-1]
        try:
            index_list_points = list_first_points.index(last_point)
            list_points = list_list_points[index_list_points]
        except ValueError:
            try:
                index_list_points = list_last_points.index(last_point)
                list_points = list_list_points[index_list_points]
                list_points.reverse()
            except ValueError:
                break
        last_point = list_points[-1]
    return ordered_list_list_points


def load_county_boundaries():
    # create query
    overpass_query_counties = """
    [out:json];
    area["name:en"="Estonia"]->.searchArea;
    (
    relation["admin_level"="6"](area.searchArea);
    );
    out geom;
    """


    # get Estonia boundaries from overpass
    response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query_counties)
    estonia_geojson = response.json()

    # parse geometry
    geometry = []
    names = []
    for element in estonia_geojson['elements']:
        members = element['members']
        name = element["tags"]["alt_name"]
        names.append(name)
        coords_poly = []
        for member in members:
            if member['type'] == 'way' and 'geometry' in member:
                coords = [(node['lon'], node['lat']) for node in member['geometry']]
                coords_poly.append(coords)
                #geometry.append(LineString(coords))
        coords_poly = order_ways(coords_poly)
        coords_poly = list(chain(*coords_poly))
        geometry.append(Polygon(coords_poly))

    name_series = pd.Series(names, name="County")
    gdf = gpd.GeoDataFrame(name_series, geometry=geometry)
    gdf = gdf.set_index("County")
    gdf.crs = 'EPSG:4326'
    return gdf

def load_capitals():
    geolocator = Nominatim(user_agent='myapplication')

    county_to_capital = {
        "Harjumaa": "Tallinn",
        "Hiiumaa": "Kärdla",
        "Ida-Virumaa": "Jõhvi",
        "Järvamaa": "Paide",
        "Jõgevamaa": "Jõgeva",
        "Läänemaa": "Haapsalu",
        "Lääne-Virumaa": "Rakvere",
        "Põlvamaa": "Põlva",
        "Pärnumaa": "Pärnu",
        "Raplamaa": "Rapla",
        "Saaremaa": "Kuressaare",
        "Tartumaa": "Tartu",
        "Valgamaa": "Valga",
        "Viljandimaa": "Viljandi",
        "Võrumaa": "Võru"
    }

    data = []

    for county, capital in county_to_capital.items():
        location = geolocator.geocode(capital)
        data.append(pd.Series([county, capital, location.raw["lat"], location.raw["lon"]], index=["county_name", "capital_name", "lat", "lon"]))

    return pd.DataFrame(data)