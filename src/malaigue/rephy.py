"""Load IFREMER REPHY in-situ series, filtered to the Thau lagoon.

The REPHY metropolitan dataset (SEANOE DOI 10.17882/47248) ships two CSVs;
the Mediterranean one (REPHY_Med_1987-2022.csv) covers Thau. Format is
semicolon-separated, latin-1, with verbose Quadrige column names. Thau is grouped
under the classement entity "104 - Etang de Thau"; the long-term station is
"Bouzigues (a)". Parameter labels: "Chlorophylle a", "Oxygène dissous", "Turbidité FNU".
"""
import pandas as pd

COL_DATE = "Passage : Date"
COL_ENTITE = "Lieu de surveillance : Entité de classement : Libellé"
COL_STATION = "Lieu de surveillance : Libellé"
COL_PARAM = "Résultat : Libellé paramètre"
COL_VALUE = "Résultat : Valeur de la mesure"
THAU_KEY = "Thau"


def thau_series(csv_path, params, start, end):
    """Tidy long-form Thau series for the requested parameters and date range.

    Returns columns: date, station, param, value.
    """
    usecols = [COL_DATE, COL_ENTITE, COL_STATION, COL_PARAM, COL_VALUE]
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1", usecols=usecols, low_memory=False)
    df = df.rename(columns={
        COL_DATE: "date", COL_ENTITE: "entite", COL_STATION: "station",
        COL_PARAM: "param", COL_VALUE: "value",
    })
    df = df[df["entite"].astype(str).str.contains(THAU_KEY, case=False, na=False)]
    df = df[df["param"].isin(params)]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)[["date", "station", "param", "value"]]
