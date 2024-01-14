import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd
import geopy.distance
from global_land_mask import globe

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

def locate_weather_stations(
        df_weather_hist: pd.DataFrame,
        gdf_counties: gpd.GeoSeries
    ) -> pd.DataFrame:
    weather_stations = np.unique(df_weather_hist[["latitude", "longitude"]], axis=0)

    data = []

    for lat, lon in weather_stations:
        station_counties = set()
        for county_name, county_geometry in gdf_counties.iterrows():
            county_geometry: Polygon = county_geometry["geometry"]
            if county_geometry.contains(Point(lon, lat)):
                station_counties.add(county_name)
        assert len(station_counties) <= 1
        data.append(pd.Series([lon, lat, next(iter(station_counties), None)], index=["longitude", "latitude", "county_name"]))

    return pd.DataFrame(data)

def naive_county_weather(df_weather_hist: pd.DataFrame, df_station_to_county: pd.DataFrame) -> pd.DataFrame:
    return (
        df_weather_hist
        .set_index(["latitude", "longitude"])
        .join(df_station_to_county.dropna().set_index(["latitude", "longitude"]), on=["latitude", "longitude"])
        .groupby(["county_name", "datetime"])
        .mean()
        .reset_index()
    )

def find_relevant_county_stations(
        df_capitals: pd.DataFrame, 
        df_weather_hist: pd.DataFrame,
        max_dist: float = 100.0,
        on_land_only: bool = True,
):
    data = []
    weather_stations = [Point(*r) for r in np.unique(df_weather_hist[["longitude", "latitude"]], axis=0)]
    for _, row in df_capitals.iterrows():
        capital_loc = Point(row["lon"], row["lat"])
        for station in weather_stations:
            if (dist := geopy.distance.geodesic((capital_loc.x, capital_loc.y), (station.x, station.y)).km) < max_dist:
                if on_land_only:
                    if globe.is_land(lon=station.x, lat=station.y):
                        data.append([row["county_name"], station.x, station.y, dist])
                else:
                    data.append([row["county_name"], station.x, station.y, dist])
    # 
    return pd.DataFrame(data, columns=["county_name", "lon", "lat", "dist"])