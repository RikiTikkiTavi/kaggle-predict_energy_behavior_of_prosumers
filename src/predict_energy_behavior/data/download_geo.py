from itertools import chain
import logging
import os
from pathlib import Path
from typing import NamedTuple
import hydra
import pandas as pd

import requests
import predict_energy_behavior.data.constants as constants
import predict_energy_behavior.config as config
from dataclasses import dataclass
import polars as pl
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd
from geopy.geocoders import Nominatim
import json

_logger = logging.getLogger(__name__)


def _order_ways(list_list_points):
    # list_first_points = [list_points[0] for list_points in list_list_points]
    # list_last_points = [list_points[-1] for list_points in list_list_points]
    ordered_list_list_points = []
    first_list_point = list_list_points[0]
    list_points = list_list_points[0]
    last_point = 0
    while last_point != first_list_point[-1]:
        ordered_list_list_points.append(list_points)
        list_list_points.remove(list_points)
        list_first_points = [list_points[0] for list_points in list_list_points]
        list_last_points = [list_points[-1] for list_points in list_list_points]
        # print(list_points[0])
        # print(list_points[-1])
        # print()
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
    response = requests.post(
        "https://overpass-api.de/api/interpreter", data=overpass_query_counties
    )
    estonia_geojson = response.json()

    # parse geometry
    geometry = []
    names = []
    for element in estonia_geojson["elements"]:
        members = element["members"]
        name = element["tags"]["alt_name"]
        names.append(name)
        coords_poly = []
        for member in members:
            if member["type"] == "way" and "geometry" in member:
                coords = [(node["lon"], node["lat"]) for node in member["geometry"]]
                coords_poly.append(coords)
                # geometry.append(LineString(coords))
        coords_poly = _order_ways(coords_poly)
        coords_poly = list(chain(*coords_poly))
        geometry.append(Polygon(coords_poly))

    name_series = pd.Series(names, name="county_name")
    gdf = gpd.GeoDataFrame(name_series, geometry=geometry)
    gdf = gdf.set_index("county_name")
    gdf.crs = "EPSG:4326"
    return gdf


def load_capitals():
    geolocator = Nominatim(user_agent="myapplication")

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
        "Võrumaa": "Võru",
    }

    data = []

    for county, capital in county_to_capital.items():
        location = geolocator.geocode(capital)
        data.append(
            pd.Series(
                [county, capital, location.raw["lat"], location.raw["lon"]],
                index=["county_name", "capital_name", "latitude", "longitude"],
            )
        )

    return pd.DataFrame(data)


@hydra.main(
    config_path="../../../configs", config_name="download_geo_data", version_base="1.3"
)
def download_geo_data(cfg: config.ConfigDownloadGeoData):
    path_data = Path.cwd()

    _logger.info("Loading capitals ...")
    df_cap = load_capitals()

    _logger.info("Loading county boundaries ...")
    df_reg = load_county_boundaries()

    _logger.info("Writing ...")
    df_cap.to_csv(path_data / "capitals.csv", index=False)
    df_reg.to_csv(path_data / "county_boundaries.csv")


if __name__ == "__main__":
    download_geo_data()
