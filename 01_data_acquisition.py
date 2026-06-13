"""
YPerf — Point 1 : Acquisition & Préparation des données
========================================================
Sources recommandées :
  - Kaggle "120 years of Olympic history" : rgriffin/olympic-history
  - Kaggle "Olympic Games (Paris 2024 incl.)" : piterfm/paris-2024-olympic-summer-games
  - IOC Results & Medals : https://olympics.com/en/olympic-games

Usage :
  pip install pandas numpy kaggle pyarrow
  kaggle datasets download -d rgriffin/olympic-history -p data/raw/
  python 01_data_acquisition.py
"""

import os
import zipfile
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. CHARGEMENT
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """
    Charge le dataset historique principal.
    Le fichier athlete_events.csv couvre 1896-2016.
    """
    path = os.path.join(RAW_DIR, "athlete_events.csv")
    if not os.path.exists(path):
        # Extraction automatique si le zip est présent
        zip_path = os.path.join(RAW_DIR, "olympic-history.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
        else:
            raise FileNotFoundError(
                "Télécharge le dataset depuis Kaggle :\n"
                "  kaggle datasets download -d rgriffin/olympic-history -p data/raw/"
            )
    df = pd.read_csv(path)
    print(f"[LOAD] {len(df):,} lignes chargées — colonnes : {list(df.columns)}")
    return df


def load_paris2024() -> pd.DataFrame:
    """
    Optionnel : enrichit avec les données Paris 2024.
    """
    path = os.path.join(RAW_DIR, "medals_total.csv")
    if not os.path.exists(path):
        print("[SKIP] Données Paris 2024 non trouvées, ignorées.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Year"] = 2024
    df["Season"] = "Summer"
    return df


# ---------------------------------------------------------------------------
# 2. NETTOYAGE
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage, normalisation et ingénierie des variables.
    """

    # Garder uniquement les JO d'été
    df = df[df["Season"] == "Summer"].copy()

    # Suppression des doublons (même athlète, même épreuve, même année)
    df = df.drop_duplicates(subset=["Name", "Sport", "Event", "Year", "NOC"])

    # Normalisation des médailles : Gold=3, Silver=2, Bronze=1, None=0
    medal_map = {"Gold": 3, "Silver": 2, "Bronze": 1}
    df["Medal_Score"] = df["Medal"].map(medal_map).fillna(0).astype(int)
    df["Has_Medal"] = (df["Medal_Score"] > 0).astype(int)

    # Nettoyage âge : suppression des valeurs aberrantes
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= 14) | (df["Age"].isna())]
    df = df[(df["Age"] <= 80) | (df["Age"].isna())]

    # Nettoyage poids / taille
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")

    # Standardisation des noms de pays (quelques corrections connues)
    country_fixes = {
        "URS": "Russia",
        "EUA": "Germany",
        "FRG": "Germany",
        "GDR": "Germany",
        "TCH": "Czech Republic",
        "YUG": "Yugoslavia",
    }
    df["Team"] = df["Team"].replace(country_fixes)

    # Calcul de la génération (décennie de naissance)
    df["Birth_Year"] = df["Year"] - df["Age"]
    df["Generation"] = (df["Birth_Year"] // 10 * 10).astype("Int64")

    # Variable binaire de participation
    df["Is_Summer"] = 1

    print(f"[CLEAN] {len(df):,} lignes après nettoyage")
    print(f"  Années couvertes : {sorted(df['Year'].unique())}")
    print(f"  Sports : {df['Sport'].nunique()}")
    print(f"  Pays (NOC) : {df['NOC'].nunique()}")
    print(f"  Taux de médaillés : {df['Has_Medal'].mean():.1%}")

    return df


# ---------------------------------------------------------------------------
# 3. AGRÉGATIONS
# ---------------------------------------------------------------------------

def build_country_year_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tableau de bord par pays et par année :
    nombre de médailles Gold/Silver/Bronze, total, score pondéré.
    """
    agg = (
        df.groupby(["NOC", "Team", "Year"])
        .agg(
            Gold=("Medal", lambda x: (x == "Gold").sum()),
            Silver=("Medal", lambda x: (x == "Silver").sum()),
            Bronze=("Medal", lambda x: (x == "Bronze").sum()),
            Total_Medals=("Has_Medal", "sum"),
            Medal_Score=("Medal_Score", "sum"),
            Athletes=("Name", "nunique"),
            Sports=("Sport", "nunique"),
        )
        .reset_index()
    )
    agg["Gold_Rate"] = agg["Gold"] / agg["Athletes"].clip(lower=1)
    return agg


def build_sport_year_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques par sport et par année.
    """
    agg = (
        df.groupby(["Sport", "Year"])
        .agg(
            Total_Athletes=("Name", "nunique"),
            Total_Countries=("NOC", "nunique"),
            Total_Events=("Event", "nunique"),
            Total_Medals=("Has_Medal", "sum"),
            Avg_Age=("Age", "mean"),
        )
        .reset_index()
    )
    return agg


def build_athlete_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Palmarès agrégé par athlète sur toute sa carrière.
    """
    agg = (
        df.groupby(["Name", "NOC", "Team", "Sport", "Sex"])
        .agg(
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
            Participations=("Year", "nunique"),
            Gold=("Medal", lambda x: (x == "Gold").sum()),
            Silver=("Medal", lambda x: (x == "Silver").sum()),
            Bronze=("Medal", lambda x: (x == "Bronze").sum()),
            Total_Medals=("Has_Medal", "sum"),
            Medal_Score=("Medal_Score", "sum"),
        )
        .reset_index()
    )
    agg["Career_Span"] = agg["Last_Year"] - agg["First_Year"]
    return agg


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING POUR LA PRÉDICTION
# ---------------------------------------------------------------------------

def build_prediction_features(country_year: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features pour les modèles prédictifs.
    Fenêtre glissante sur 3 éditions précédentes.
    """
    df = country_year.sort_values(["NOC", "Year"]).copy()

    for col in ["Gold", "Total_Medals", "Medal_Score", "Athletes"]:
        # Moyenne mobile sur les 3 dernières éditions
        df[f"{col}_ma3"] = (
            df.groupby("NOC")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        # Tendance : différence avec édition précédente
        df[f"{col}_trend"] = df.groupby("NOC")[col].diff()
        # Taux de croissance
        df[f"{col}_growth"] = df.groupby("NOC")[col].pct_change().clip(-2, 2)

    # Rang mondial à chaque édition
    df["World_Rank"] = df.groupby("Year")["Medal_Score"].rank(ascending=False, method="min")

    # Nombre d'éditions consécutives avec au moins une médaille
    df["Streak"] = (
        df.groupby("NOC")["Total_Medals"]
        .transform(lambda x: x.gt(0).groupby((x == 0).cumsum()).cumcount())
    )

    df = df.dropna(subset=["Gold_ma3", "Total_Medals_ma3"])
    print(f"[FEATURES] Dataset prédiction : {len(df):,} lignes, {df.shape[1]} colonnes")
    return df


# ---------------------------------------------------------------------------
# 5. EXPORT
# ---------------------------------------------------------------------------

def export_datasets(
    df_clean: pd.DataFrame,
    country_year: pd.DataFrame,
    sport_year: pd.DataFrame,
    athletes: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Exporte tous les datasets au format Parquet (compact et rapide)."""
    exports = {
        "olympics_clean": df_clean,
        "country_year_stats": country_year,
        "sport_year_stats": sport_year,
        "athlete_stats": athletes,
        "prediction_features": features,
    }
    for name, data in exports.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        data.to_parquet(path, index=False)
        print(f"[EXPORT] {path} — {len(data):,} lignes, {data.shape[1]} colonnes")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("YPerf — Pipeline Acquisition & Préparation")
    print("=" * 60)

    # 1. Chargement
    df_raw = load_raw_data()
    df_paris = load_paris2024()

    # 2. Nettoyage
    df_clean = clean_data(df_raw)

    # 3. Agrégations
    country_year = build_country_year_stats(df_clean)
    sport_year = build_sport_year_stats(df_clean)
    athletes = build_athlete_stats(df_clean)

    # 4. Features prédiction
    features = build_prediction_features(country_year)

    # 5. Export
    export_datasets(df_clean, country_year, sport_year, athletes, features)

    print("\n[OK] Pipeline terminé. Données prêtes pour la modélisation.")
