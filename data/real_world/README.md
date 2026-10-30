# SelfHealDS Real-World Case Study CSVs

Place the researcher-supplied CSVs in this folder. The Streamlit application never downloads these datasets at runtime.

Required six case-study files:
1. `01_nasa_cmapss.csv` — NASA C-MAPSS aerospace data. This is an analogous public NASA dataset for the documented 1999 Mars Climate Orbiter unit-contract failure; it must NOT be described as the original MCO telemetry.
2. `02_hospital.csv` — HoloClean Hospital dirty benchmark CSV.
3. `03_flights.csv` — HoloClean Flight benchmark CSV.
4. `04_chicago_food_inspections.csv` — Chicago Food Inspections CSV snapshot.
5. `05_air_quality.csv` — UCI Air Quality CSV (semicolon-delimited source should be converted to a normal CSV before upload).
6. `06_adult_census.csv` — UCI Adult/Census Income CSV.

Optional ground-truth files for independent cell-level scoring:
- `02_hospital_clean.csv`
- `03_flights_clean.csv`

Important: the NASA case is a documented historical incident plus an analogous public dataset, not a claim that C-MAPSS contains the Mars Climate Orbiter telemetry failure.
