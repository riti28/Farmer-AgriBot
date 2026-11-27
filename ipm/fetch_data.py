"""
Step 1: Download REAL daily weather from the NASA POWER API (free, no API key,
covers the whole world) and turn it into WEEKLY weather for 16 farming sites
on 5 continents (2015-2024).

Output: data/weather_raw.csv
Run:    python fetch_data.py
"""
import os
import requests
import pandas as pd

# One site = a town in a major growing area of the crop (town centre coordinates)
SITES = {
    # name                      (country,       latitude, longitude)
    "Mandya":                  ("India",        12.5223,   76.8970),
    "Cuttack":                 ("India",        20.4625,   85.8830),
    "Los Banos":               ("Philippines",  14.1699,  121.2441),
    "Can Tho":                 ("Vietnam",      10.0452,  105.7469),
    "Dharwad":                 ("India",        15.4589,   75.0078),
    "Lusaka":                  ("Zambia",      -15.3875,   28.3228),
    "Sorriso":                 ("Brazil",      -12.5425,  -55.7211),
    "Ames":                    ("USA",          42.0308,  -93.6319),
    "Raichur":                 ("India",        16.2120,   77.3439),
    "Multan":                  ("Pakistan",     30.1575,   71.5249),
    "Lubbock":                 ("USA",          33.5779, -101.8552),
    "Belagavi":                ("India",        15.8497,   74.4977),
    "Ribeirao Preto":          ("Brazil",      -21.1775,  -47.8103),
    "Bundaberg":               ("Australia",   -24.8661,  152.3489),
    "Soroti":                  ("Uganda",        1.7146,   33.6111),
    "Kathmandu":               ("Nepal",        27.7172,   85.3240),
}

# T2M = air temperature at 2 m (deg C), RH2M = relative humidity (%),
# PRECTOTCORR = rainfall (mm/day)
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch_daily(lat, lon, start, end):
    """Daily weather table for one point. start/end look like '20150101'."""
    params = {"parameters": "T2M,RH2M,PRECTOTCORR", "community": "AG",
              "latitude": lat, "longitude": lon, "start": start, "end": end, "format": "JSON"}
    reply = requests.get(URL, params=params, timeout=120)
    reply.raise_for_status()
    # the answer looks like {"T2M": {"20150101": 22.1, ...}, "RH2M": {...}, ...}
    daily = pd.DataFrame(reply.json()["properties"]["parameter"])
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    return daily.replace(-999, pd.NA).dropna()   # -999 means "missing"


def to_weekly(daily):
    """Average temperature & humidity, TOTAL rainfall, per week of the year."""
    daily = daily.copy()
    daily["year"] = daily.index.year
    # Week 1 = days 1-7 of the year, ... the last 8-9 days go in week 52
    daily["week"] = (daily.index.day_of_year - 1) // 7 + 1
    daily["week"] = daily["week"].clip(upper=52)
    return daily.groupby(["year", "week"]).agg(
        temp=("T2M", "mean"),
        humidity=("RH2M", "mean"),
        rainfall=("PRECTOTCORR", "sum"),
    ).reset_index()


if __name__ == "__main__":
    all_rows = []
    for site, (country, lat, lon) in SITES.items():
        print("Downloading", site, country, "...")
        weekly = to_weekly(fetch_daily(lat, lon, "20150101", "20241231"))
        weekly.insert(0, "site", site)
        weekly.insert(1, "country", country)
        all_rows.append(weekly)

    weather = pd.concat(all_rows).round(2)
    weather.to_csv(os.path.join(HERE, "data", "weather_raw.csv"), index=False)
    print("Saved", len(weather), "weekly rows to data/weather_raw.csv")
