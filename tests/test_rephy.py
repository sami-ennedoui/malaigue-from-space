import pandas as pd
from malaigue import rephy


def test_thau_series_filters(tmp_path):
    csv = tmp_path / "rephy.csv"
    pd.DataFrame({
        "Passage : Date": ["05/07/2018", "05/07/2018", "01/01/2016"],
        "Lieu de surveillance : Entité de classement : Libellé":
            ["104 - Etang de Thau", "999 - Bassin Arcachon", "104 - Etang de Thau"],
        "Lieu de surveillance : Libellé": ["Bouzigues (a)", "Arcachon (a)", "Marseillan (a)"],
        "Résultat : Libellé paramètre": ["Chlorophylle a", "Chlorophylle a", "Oxygène dissous"],
        "Résultat : Valeur de la mesure": ["42,0", "1.0", "3,0"],
    }).to_csv(csv, sep=";", index=False, encoding="latin-1")
    out = rephy.thau_series(
        str(csv), params=["Chlorophylle a", "Oxygène dissous"],
        start="2018-01-01", end="2018-12-31",
    )
    # Arcachon dropped (not Thau); 2016 row dropped (out of range)
    assert set(out["station"]) == {"Bouzigues (a)"}
    assert len(out) == 1
    assert out["param"].iloc[0] == "Chlorophylle a"
    assert abs(out["value"].iloc[0] - 42.0) < 1e-9  # French comma decimal parsed
