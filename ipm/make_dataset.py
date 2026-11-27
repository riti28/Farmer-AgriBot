"""
Step 2: Build the labelled pest-risk dataset.

For every crop, growing site, year and week of the crop season we take the REAL
weekly weather (from fetch_data.py) and give it a pest-risk label
(Low / Medium / High) using weather thresholds reported in published papers.

IMPORTANT: these are RULE-DERIVED labels, not real field pest counts.

Output: data/pest_data.csv
Run:    python make_dataset.py
"""
import os
import random
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)  # same "random" noise every time we run

# ---------------------------------------------------------------------------
# 1. Pest thresholds (one entry per pest). A rule set to None is not used.
#    temp              = weekly mean air temperature range (deg C)
#    hum_min / hum_max = weekly mean relative humidity (%)
#    rain_min/rain_max = weekly total rainfall (mm)
#    stages            = crop stages when the crop is most attacked
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "rice": [
        {
            "pest": "Brown planthopper (Nilaparvata lugens)",
            "temp": (25, 30), "hum_min": 70, "hum_max": None,
            "rain_min": None, "rain_max": None,
            "stages": ["vegetative", "flowering"],
            "source": "Scientific Reports 11 (2021), doi:10.1038/s41598-020-80704-4 "
                      "(nymphs develop best at 25-30 C); 70-90% RH from Krishnaiah et al. (2005) "
                      "as cited in later BPH papers [VERIFY primary source]",
        },
        {
            "pest": "Yellow stem borer (Scirpophaga incertulas)",
            "temp": (24, 32), "hum_min": 75, "hum_max": None,
            "rain_min": None, "rain_max": None,
            "stages": ["vegetative", "flowering"],
            "source": "Prasannakumar, Chander & Vijay Kumar (2015) Cogent Food & Agriculture 1:995281, "
                      "doi:10.1080/23311932.2014.995281; Balaji et al. (2024) Indian J. Agric. Sci. "
                      "94(9):996-1000, doi:10.56093/ijas.v94i9.147332 (both from Mandya)",
        },
    ],
    "maize": [
        {
            "pest": "Fall armyworm (Spodoptera frugiperda)",
            "temp": (26, 30), "hum_min": None, "hum_max": None,
            "rain_min": None, "rain_max": 30,
            "stages": ["seedling", "vegetative"],
            "source": "Du Plessis, Schlemmer & Van den Berg (2020) Insects 11(4):228, "
                      "doi:10.3390/insects11040228 (larval survival highest at 26-30 C); heavy rain "
                      "kills young larvae: FAO (2018) FAW guide for farmer field schools [VERIFY]",
        },
    ],
    "cotton": [
        {
            "pest": "Pink bollworm (Pectinophora gossypiella)",
            "temp": (15, 26), "hum_min": None, "hum_max": 70,
            "rain_min": None, "rain_max": 10,
            "stages": ["flowering", "maturity"],
            "source": "Rakhesh S. et al. (2023) J. Agrometeorology 25(4), doi:10.54386/jam.v25i4.2300 "
                      "(Raichur: pest falls as min. temperature, RH and rain rise)",
        },
    ],
    "sugarcane": [
        {
            "pest": "Early shoot borer (Chilo infuscatellus)",
            "temp": (28, 40), "hum_min": None, "hum_max": 60,
            "rain_min": None, "rain_max": None,
            "stages": ["seedling", "vegetative"],
            "source": "Pandey & Kumar, Indian J. Agric. Research 48(3):227-231, "
                      "doi:10.5958/j.0976-058X.48.3.038 (incidence up with temperature, down with RH)",
        },
    ],
    "finger_millet": [
        {
            "pest": "Blast (Pyricularia grisea)",
            "temp": (15, 25), "hum_min": 85, "hum_max": None,
            "rain_min": 5, "rain_max": None,
            "stages": ["vegetative", "flowering"],
            "source": "Navya H.M. et al. (2025) Agricultural Reviews 46(3):469-473, "
                      "doi:10.18805/ag.R-2633 (15-25 C, RH > 85%, intermittent rain)",
        },
    ],
}

# ---------------------------------------------------------------------------
# 2. Simple crop calendar: for each crop, the sites that grow it (see fetch_data.py)
#    and the usual SOWING week there (weeks 1-52). Southern-hemisphere and
#    tropical sites have different seasons, so each site has its own sowing week.
# ---------------------------------------------------------------------------
CROP_CALENDAR = {
    "rice":          {"weeks_long": 20, "sites": {"Mandya": 27, "Cuttack": 26, "Los Banos": 24, "Can Tho": 48}},
    "maize":         {"weeks_long": 18, "sites": {"Dharwad": 23, "Lusaka": 47, "Sorriso": 6, "Ames": 17}},
    "cotton":        {"weeks_long": 26, "sites": {"Raichur": 26, "Multan": 18, "Lubbock": 20}},
    "sugarcane":     {"weeks_long": 30, "sites": {"Belagavi": 1, "Ribeirao Preto": 40, "Bundaberg": 36}},
    "finger_millet": {"weeks_long": 18, "sites": {"Mandya": 31, "Soroti": 12, "Kathmandu": 26}},
}

STAGES = ["seedling", "vegetative", "flowering", "maturity"]


def get_stage(weeks_after_sowing, weeks_long):
    """Split the crop season into 4 equal parts = 4 crop stages."""
    return STAGES[int(weeks_after_sowing / weeks_long * 4)]


def pest_risk(rule, temp, humidity, rainfall, stage):
    """Check each condition. All met -> High, one missed -> Medium, else Low."""
    checks = [rule["temp"][0] <= temp <= rule["temp"][1], stage in rule["stages"]]
    if rule["hum_min"] is not None:
        checks.append(humidity >= rule["hum_min"])
    if rule["hum_max"] is not None:
        checks.append(humidity <= rule["hum_max"])
    if rule["rain_min"] is not None:
        checks.append(rainfall >= rule["rain_min"])
    if rule["rain_max"] is not None:
        checks.append(rainfall <= rule["rain_max"])

    missed = checks.count(False)
    if missed == 0:
        return "High"
    if missed == 1:
        return "Medium"
    return "Low"


def risk_for_crop(crop, temp, humidity, rainfall, stage):
    """Rice has 2 pests, so we keep the WORST (highest) risk."""
    order = {"Low": 0, "Medium": 1, "High": 2}
    risks = [pest_risk(rule, temp, humidity, rainfall, stage) for rule in THRESHOLDS[crop]]
    return max(risks, key=lambda r: order[r])


if __name__ == "__main__":
    weather = pd.read_csv(os.path.join(HERE, "data", "weather_raw.csv"))
    weather = weather.set_index(["site", "year", "week"])

    rows = []
    for crop, info in CROP_CALENDAR.items():
        for site, sowing_week in info["sites"].items():
            for year in range(2015, 2025):
                for i in range(info["weeks_long"]):
                    week = (sowing_week - 1 + i) % 52 + 1   # wraps 52 -> 1 (same year's weather, simplification)
                    w = weather.loc[(site, year, week)]
                    stage = get_stage(i, info["weeks_long"])
                    rows.append({
                        "crop": crop, "site": site, "country": w["country"], "year": year, "week": week,
                        "temp": w["temp"], "humidity": w["humidity"], "rainfall": w["rainfall"],
                        "crop_stage": stage,
                        "pest_risk": risk_for_crop(crop, w["temp"], w["humidity"], w["rainfall"], stage),
                    })
    data = pd.DataFrame(rows)

    # 3. Add 10% label noise (real field data is never perfect)
    for i in data.sample(frac=0.10, random_state=42).index:
        other_labels = [label for label in ["Low", "Medium", "High"] if label != data.at[i, "pest_risk"]]
        data.at[i, "pest_risk"] = random.choice(other_labels)

    data.to_csv(os.path.join(HERE, "data", "pest_data.csv"), index=False)
    print("Saved", len(data), "rows to data/pest_data.csv")
    print(pd.crosstab(data["crop"], data["pest_risk"], margins=True).to_string())
