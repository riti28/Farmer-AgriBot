# Data dictionary

## `weather_raw.csv` (made by `fetch_data.py`) - 8,320 rows

| Column | Meaning | Unit | Source |
|---|---|---|---|
| site | Town in a major growing area (weather taken at its centre point) | text | Our choice of 16 sites |
| country | Country of the site | text | - |
| year | Calendar year, 2015-2024 | year | NASA POWER |
| week | Week of the year (days 1-7 = week 1; the last 8-9 days of the year go into week 52) | 1-52 | Computed from the date |
| temp | Weekly mean of daily air temperature at 2 m (`T2M`) | °C | NASA POWER daily API, community AG |
| humidity | Weekly mean of daily relative humidity at 2 m (`RH2M`) | % | NASA POWER daily API |
| rainfall | Weekly total of daily corrected precipitation (`PRECTOTCORR`) | mm/week | NASA POWER daily API |

NASA POWER API: https://power.larc.nasa.gov/ (free, no key, global coverage).

| Site | Country | Lat, Lon | Crop(s) and sowing week |
|---|---|---|---|
| Mandya | India | 12.52, 76.90 | rice (27), finger millet (31) |
| Cuttack | India | 20.46, 85.88 | rice (26) |
| Los Banos | Philippines | 14.17, 121.24 | rice (24) |
| Can Tho | Vietnam | 10.05, 105.75 | rice (48) |
| Dharwad | India | 15.46, 75.01 | maize (23) |
| Lusaka | Zambia | -15.39, 28.32 | maize (47) |
| Sorriso | Brazil | -12.54, -55.72 | maize (6) |
| Ames | USA | 42.03, -93.63 | maize (17) |
| Raichur | India | 16.21, 77.34 | cotton (26) |
| Multan | Pakistan | 30.16, 71.52 | cotton (18) |
| Lubbock | USA | 33.58, -101.86 | cotton (20) |
| Belagavi | India | 15.85, 74.50 | sugarcane (1) |
| Ribeirao Preto | Brazil | -21.18, -47.81 | sugarcane (40) |
| Bundaberg | Australia | -24.87, 152.35 | sugarcane (36) |
| Soroti | Uganda | 1.71, 33.61 | finger millet (12) |
| Kathmandu | Nepal | 27.72, 85.32 | finger millet (26) |

Sowing weeks are typical main-season values, rounded; real dates vary by year and farmer.

## `pest_data.csv` (made by `make_dataset.py`) - 3,740 rows

| Column | Meaning | Unit / values | Source |
|---|---|---|---|
| crop | Crop | rice, maize, cotton, sugarcane, finger_millet | Our crop calendar |
| site, country | Where the weather comes from | text | `weather_raw.csv` |
| year | Year of the weather | 2015-2024 | NASA POWER |
| week | Week of the year | 1-52 | NASA POWER date |
| temp, humidity, rainfall | Same as in `weather_raw.csv` | °C, %, mm/week | NASA POWER |
| crop_stage | Crop season (rice 20, maize 18, cotton 26, sugarcane 30, finger millet 18 weeks) split into 4 equal parts | seedling, vegetative, flowering, maturity | Rule in `get_stage()` |
| **pest_risk** (label) | Weekly pest risk | Low / Medium / High | Thresholds below + 10% random label noise |

**Model inputs** (`train.py`): crop, crop_stage, temp, humidity, rainfall. Site, country, year and week
are *not* used, so the model can be applied to any place in the world.

### How the label is made
For each pest of the crop we check: temperature in range, humidity rule, rainfall rule, and susceptible crop stage.
All checks met → **High**; one check missed → **Medium**; two or more missed → **Low**.
Rice has two pests; the higher risk is kept. Then 10% of rows get a randomly chosen *different* label (noise).

### Threshold table

| Crop | Pest | Temp (°C, weekly mean) | Humidity (%) | Rain (mm/week) | Stages | Citation |
|---|---|---|---|---|---|---|
| Rice | Brown planthopper | 25-30 | ≥ 70 | - | vegetative, flowering | Scientific Reports 11 (2021), doi:10.1038/s41598-020-80704-4 (nymph development best at 25-30 °C); 70-90% RH from Krishnaiah et al. (2005), cited in later BPH papers **[VERIFY primary source]** |
| Rice | Yellow stem borer | 24-32 | ≥ 75 | - | vegetative, flowering | Prasannakumar, Chander & Vijay Kumar (2015) *Cogent Food & Agriculture* 1:995281, doi:10.1080/23311932.2014.995281; Balaji et al. (2024) *Indian J. Agric. Sci.* 94(9):996-1000, doi:10.56093/ijas.v94i9.147332 |
| Maize | Fall armyworm | 26-30 | - | ≤ 30 | seedling, vegetative | Du Plessis, Schlemmer & Van den Berg (2020) *Insects* 11(4):228, doi:10.3390/insects11040228; rain effect from FAO (2018) *Integrated management of the fall armyworm on maize* (I8665EN) **[VERIFY]** |
| Cotton | Pink bollworm | 15-26 | ≤ 70 | ≤ 10 | flowering, maturity | Rakhesh S., Hanchinal S.G., Bheemanna M., Hosamani A.K., Nidagundi J.M., Prabhulinga T. (2023) *J. Agrometeorology* 25(4), doi:10.54386/jam.v25i4.2300 (catches fall as min. temp, RH and rain rise) |
| Sugarcane | Early shoot borer | 28-40 | ≤ 60 | - | seedling, vegetative | Pandey S.K. & Kumar R., *Indian J. Agric. Research* 48(3):227-231, doi:10.5958/j.0976-058X.48.3.038 (incidence up with temperature, down with RH) |
| Finger millet | Blast | 15-25 | ≥ 85 | ≥ 5 | vegetative, flowering | Navya H.M. et al. (2025) *Agricultural Reviews* 46(3):469-473, doi:10.18805/ag.R-2633 (15-25 °C, RH > 85%, intermittent rain) |

The papers report *correlations* or *favourable ranges*; we turned them into simple cut-offs.
The exact cut-off numbers (e.g. "≤ 10 mm rain") are our simplification, not values copied from the papers.

## Limitations
- **Rule-derived labels, not field pest counts.** The model learns our threshold rules, not real outbreaks.
- **Thresholds come mostly from Indian studies** (plus one South African lab study). Pest populations,
  varieties and climates differ between countries, so the same rule may not hold everywhere.
- **One point per site.** NASA POWER has a ~0.5° grid; a whole region is represented by one town.
- **No soil, irrigation, variety or pesticide data**, and no natural-enemy counts.
- Weekly mean temperature is used even where a paper talks about minimum or maximum temperature.
- Crop stages are an equal 4-way split from one fixed sowing week per site, not real sowing dates.
  Seasons that cross New Year use the same calendar year's weather for weeks 1-n (a simplification).
- 10% label noise is artificial; it means ~90% is the best accuracy any model can reach.
