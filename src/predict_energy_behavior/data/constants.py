data_cols = ('target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id')
client_cols = ('product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date')
gas_prices_cols = ('forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh')
electricity_prices_cols = ('forecast_date', 'euros_per_mwh')
forecast_weather_cols = ('latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation')
historical_weather_cols = ('datetime', 'temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure','cloudcover_total','cloudcover_low','cloudcover_mid','cloudcover_high','windspeed_10m','winddirection_10m','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude')
location_cols = ('longitude', 'latitude', 'county')
target_cols = ('target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime')

product_type_to_name = {
    0: "Combined",
    1: "Fixed",
    2: "General service",
    3: "Spot"
}