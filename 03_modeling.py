"""
YPerf — Point 3 : Modélisation Prédictive JO 2028
==================================================
Modèles implémentés :
  1. Régression (Random Forest + Ridge) — prédiction du nombre de médailles par pays
  2. Clustering (K-Means) — segmentation des pays par profil de performance
  3. Scoring "côte" athlète — score composite basé sur la progression récente

Usage :
  pip install scikit-learn joblib matplotlib seaborn
  python 03_modeling.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET_YEAR = 2028
LAST_KNOWN_YEAR = 2021  # Tokyo (2020 reportés à 2021)
MIN_ACTIVE_YEAR = 1992  # On ignore les pays disparus avant 1992

# Pays défunts à exclure explicitement des prédictions
DEFUNCT_COUNTRIES = {
    "URS", "EUA", "FRG", "GDR", "TCH", "YUG", "SCG",
    "ANZ", "BOH", "RU1", "NFL", "MAL", "WIF",
}


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DES FEATURES
# ---------------------------------------------------------------------------

def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, "prediction_features.parquet"))
    athletes = pd.read_parquet(os.path.join(PROCESSED_DIR, "athlete_stats.parquet"))
    country_year = pd.read_parquet(os.path.join(PROCESSED_DIR, "country_year_stats.parquet"))
    return features, athletes, country_year


# ---------------------------------------------------------------------------
# 2. RÉGRESSION — PRÉDICTION DES MÉDAILLES PAR PAYS
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Gold_ma3", "Total_Medals_ma3", "Medal_Score_ma3", "Athletes_ma3",
    "Total_Medals_trend", "Medal_Score_trend", "Gold_growth",
    "World_Rank", "Streak", "Athletes",
]
TARGET_COL = "Total_Medals"


def train_regression_models(features: pd.DataFrame) -> dict:
    """
    Entraîne deux modèles (Random Forest + Ridge) avec validation croisée
    par série temporelle (TimeSeriesSplit).
    """
    df_train = features[features["Year"] <= LAST_KNOWN_YEAR].dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df_train[FEATURE_COLS]
    y = df_train[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)),
        ]),
    }

    results = {}
    best_score = -np.inf
    best_name = None

    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2")
        mae_scores = -cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        pipe.fit(X, y)
        results[name] = {
            "pipeline": pipe,
            "r2_mean": scores.mean(),
            "r2_std": scores.std(),
            "mae_mean": mae_scores.mean(),
        }
        print(f"[REGRESSION] {name:25s} R²={scores.mean():.3f} ± {scores.std():.3f}  MAE={mae_scores.mean():.2f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name

    print(f"[REGRESSION] Meilleur modèle : {best_name} (R²={best_score:.3f})")

    # Sauvegarde du meilleur modèle
    joblib.dump(results[best_name]["pipeline"], os.path.join(MODELS_DIR, "regression_medals.pkl"))

    # Feature importance (Random Forest)
    rf_pipe = results["random_forest"]["pipeline"]
    importances = rf_pipe.named_steps["model"].feature_importances_
    feat_imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}).sort_values("importance", ascending=False)
    _plot_feature_importance(feat_imp)

    return results, best_name


def _plot_feature_importance(feat_imp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_imp, x="importance", y="feature", ax=ax, palette="Blues_d")
    ax.set_title("Importance des variables — Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"[PLOT] Feature importance sauvegardé")


def predict_2028(features: pd.DataFrame, model_results: dict, best_name: str) -> pd.DataFrame:
    """
    Projette les performances pour 2028 en utilisant les données les plus récentes
    (dernier quadriennal connu) comme features.
    Exclut les pays défunts et ceux inactifs depuis 1992.
    """
    # Garder uniquement les pays actifs (dernière participation >= 1992)
    last_year_by_noc = features.groupby("NOC")["Year"].max()
    active_nocs = last_year_by_noc[last_year_by_noc >= MIN_ACTIVE_YEAR].index

    df_active = features[
        features["NOC"].isin(active_nocs) &
        ~features["NOC"].isin(DEFUNCT_COUNTRIES)
    ].copy()

    # Prendre le snapshot le plus récent disponible par pays
    latest = (
        df_active.sort_values("Year")
        .groupby("NOC")
        .last()
        .reset_index()
        .dropna(subset=FEATURE_COLS)
    )

    pipe = model_results[best_name]["pipeline"]
    X_pred = latest[FEATURE_COLS]
    latest["Pred_Medals_2028"] = pipe.predict(X_pred).clip(0).round().astype(int)

    # Intervalle de confiance approximatif (±1 MAE)
    mae = model_results[best_name]["mae_mean"]
    latest["Pred_Medals_Low"] = (latest["Pred_Medals_2028"] - mae).clip(0).round().astype(int)
    latest["Pred_Medals_High"] = (latest["Pred_Medals_2028"] + mae).round().astype(int)

    # Tendance : différence entre prédiction et dernière performance réelle
    latest["Delta_vs_Last"] = latest["Pred_Medals_2028"] - latest["Total_Medals"]
    latest["Trend_Label"] = pd.cut(
        latest["Delta_vs_Last"],
        bins=[-999, -3, -1, 1, 3, 999],
        labels=["En recul fort", "En recul", "Stable", "En hausse", "En forte hausse"],
    )

    pred_df = latest[["NOC", "Team", "Total_Medals", "Pred_Medals_2028",
                       "Pred_Medals_Low", "Pred_Medals_High", "Delta_vs_Last", "Trend_Label"]]
    pred_df = pred_df.sort_values("Pred_Medals_2028", ascending=False).reset_index(drop=True)
    pred_df["Rank_2028"] = pred_df.index + 1

    path = os.path.join(PROCESSED_DIR, "predictions_2028.parquet")
    pred_df.to_parquet(path, index=False)
    print(f"[PREDICT] Top 10 pays prédits JO 2028 :")
    print(pred_df.head(10)[["Rank_2028", "Team", "Total_Medals", "Pred_Medals_2028", "Trend_Label"]].to_string(index=False))
    return pred_df


# ---------------------------------------------------------------------------
# 3. CLUSTERING — SEGMENTATION DES PAYS
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "Total_Medals_ma3",
    "Medal_Score_ma3",
    "Athletes_ma3",
    "Gold_Rate",
    "Streak",
    "World_Rank",
]
N_CLUSTERS = 5
CLUSTER_LABELS = {
    0: "Dominants historiques",
    1: "Puissances émergentes",
    2: "Pays spécialisés",
    3: "Participants réguliers",
    4: "Nouveaux entrants",
}


def train_clustering(features: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means sur les pays actifs : identifie 5 profils de nations.
    """
    # Même filtre que pour les prédictions : pays actifs uniquement
    last_year_by_noc = features.groupby("NOC")["Year"].max()
    active_nocs = last_year_by_noc[last_year_by_noc >= MIN_ACTIVE_YEAR].index

    df_latest = (
        features[
            features["NOC"].isin(active_nocs) &
            ~features["NOC"].isin(DEFUNCT_COUNTRIES)
        ]
        .sort_values("Year")
        .groupby("NOC")
        .last()
        .reset_index()
        .dropna(subset=CLUSTER_FEATURES)
    )

    # Calcul du Gold_Rate si absent
    if "Gold_Rate" not in df_latest.columns:
        df_latest["Gold_Rate"] = df_latest["Gold_ma3"] / df_latest["Athletes_ma3"].clip(lower=1)

    X = df_latest[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choix du K via inertie (elbow)
    inertias = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    _plot_elbow(inertias)

    # Modèle final
    km_final = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    df_latest["Cluster"] = km_final.fit_predict(X_scaled)

    # Attribution des labels : trier les clusters par score médian (plus robuste que la moyenne)
    # pour éviter qu'un outlier tire tout un cluster vers le haut
    cluster_medians = df_latest.groupby("Cluster")["Medal_Score_ma3"].median().sort_values(ascending=False)
    rank_to_label = {
        0: "Dominants historiques",
        1: "Puissances émergentes",
        2: "Pays spécialisés",
        3: "Participants réguliers",
        4: "Nouveaux entrants",
    }
    cluster_rename = {c: rank_to_label[i] for i, c in enumerate(cluster_medians.index)}
    df_latest["Cluster_Label"] = df_latest["Cluster"].map(cluster_rename)

    # Sauvegarde modèle et scaler
    pipeline_cluster = {"kmeans": km_final, "scaler": scaler, "feature_cols": CLUSTER_FEATURES}
    joblib.dump(pipeline_cluster, os.path.join(MODELS_DIR, "clustering_countries.pkl"))

    result = df_latest[["NOC", "Team", "Cluster", "Cluster_Label"] + CLUSTER_FEATURES]
    result.to_parquet(os.path.join(PROCESSED_DIR, "country_clusters.parquet"), index=False)

    print("\n[CLUSTERING] Distribution des clusters :")
    print(result["Cluster_Label"].value_counts().to_string())
    _plot_cluster_scatter(df_latest)
    return result


def _plot_elbow(inertias: list) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(2, 10), inertias, marker="o")
    ax.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"K={N_CLUSTERS} retenu")
    ax.set_title("Méthode du coude — K-Means")
    ax.set_xlabel("Nombre de clusters")
    ax.set_ylabel("Inertie")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "kmeans_elbow.png"), dpi=150)
    plt.close(fig)


def _plot_cluster_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", N_CLUSTERS)
    for i, (label, grp) in enumerate(df.groupby("Cluster_Label")):
        ax.scatter(grp["Athletes_ma3"], grp["Medal_Score_ma3"],
                   label=label, alpha=0.7, s=60, color=palette[i])
        # Annoter les top pays
        for _, row in grp.nlargest(3, "Medal_Score_ma3").iterrows():
            ax.annotate(row["NOC"], (row["Athletes_ma3"], row["Medal_Score_ma3"]),
                        fontsize=7, alpha=0.9)
    ax.set_xlabel("Nombre moyen d'athlètes (ma3)")
    ax.set_ylabel("Score médailles pondéré (ma3)")
    ax.set_title("Segmentation des pays olympiques — K-Means")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "cluster_scatter.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. SCORE "CÔTE" ATHLÈTE
# ---------------------------------------------------------------------------

def compute_athlete_scores(athletes: pd.DataFrame) -> pd.DataFrame:
    """
    Score composite pour chaque athlète — utilisé dans le dashboard.
    Score = (Gold × 3 + Silver × 2 + Bronze × 1) × (1 + progression)
              × facteur_longévité × facteur_récence
    """
    df = athletes.copy()

    # Score brut
    df["Raw_Score"] = df["Gold"] * 3 + df["Silver"] * 2 + df["Bronze"]

    # Facteur longévité : plus d'éditions = plus fiable
    df["Longevity_Factor"] = np.log1p(df["Participations"]) / np.log1p(5)

    # Facteur récence : favorise les athlètes actifs récemment
    df["Recency_Factor"] = np.where(df["Last_Year"] >= 2016, 1.2,
                           np.where(df["Last_Year"] >= 2008, 1.0, 0.7))

    # Score final normalisé 0-100
    df["Athlete_Score"] = df["Raw_Score"] * df["Longevity_Factor"] * df["Recency_Factor"]
    max_score = df["Athlete_Score"].max()
    if max_score > 0:
        df["Athlete_Score_Norm"] = (df["Athlete_Score"] / max_score * 100).round(1)
    else:
        df["Athlete_Score_Norm"] = 0

    # Catégorie de côte
    df["Cote_Label"] = pd.cut(
        df["Athlete_Score_Norm"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["Outsider", "Compétiteur", "Favori", "Médaillable", "Légende"],
        include_lowest=True,
    )

    result = df.sort_values("Athlete_Score_Norm", ascending=False).reset_index(drop=True)
    result.to_parquet(os.path.join(PROCESSED_DIR, "athlete_scores.parquet"), index=False)

    print(f"\n[SCORES] Top 10 athlètes :")
    print(result.head(10)[["Name", "NOC", "Sport", "Athlete_Score_Norm", "Cote_Label"]].to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 5. EXPORT RAPPORT JSON (pour le dashboard)
# ---------------------------------------------------------------------------

def export_model_report(model_results: dict, best_name: str) -> None:
    """Exporte les métriques pour affichage dans le dashboard."""
    report = {}
    for name, res in model_results.items():
        report[name] = {
            "r2_mean": round(float(res["r2_mean"]), 4),
            "r2_std": round(float(res["r2_std"]), 4),
            "mae_mean": round(float(res["mae_mean"]), 4),
        }
    report["best_model"] = best_name
    report["target_year"] = TARGET_YEAR
    report["feature_cols"] = FEATURE_COLS

    path = os.path.join(REPORTS_DIR, "model_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Métriques exportées → {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("YPerf — Pipeline Modélisation Prédictive JO 2028")
    print("=" * 60)

    features, athletes, country_year = load_features()

    print("\n--- Régression : prédiction des médailles ---")
    model_results, best_name = train_regression_models(features)
    pred_2028 = predict_2028(features, model_results, best_name)

    print("\n--- Clustering : segmentation des pays ---")
    clusters = train_clustering(features)

    print("\n--- Scoring des athlètes ---")
    athlete_scores = compute_athlete_scores(athletes)

    export_model_report(model_results, best_name)

    print("\n[OK] Modélisation terminée.")
    print(f"  Modèles sauvegardés dans : {MODELS_DIR}/")
    print(f"  Prédictions : {PROCESSED_DIR}/predictions_2028.parquet")
    print(f"  Clusters    : {PROCESSED_DIR}/country_clusters.parquet")
    print(f"  Scores      : {PROCESSED_DIR}/athlete_scores.parquet")
