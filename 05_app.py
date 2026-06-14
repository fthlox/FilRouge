"""
YPerf — Point 5 : Application Streamlit (déploiement local)
============================================================
Lance avec :
  streamlit run 05_app.py
"""

import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="YPerf — JO 2028",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"
LAST_REAL_YEAR = 2024          # Paris 2024 = référence

# ---------------------------------------------------------------------------
# CHARGEMENT
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> dict:
    data, missing = {}, []
    files = {
        "predictions":  "predictions_2028.parquet",
        "clusters":     "country_clusters.parquet",
        "athletes":     "athlete_scores.parquet",
        "country_year": "country_year_stats.parquet",
        "sport_year":   "sport_year_stats.parquet",
        "clean":        "olympics_clean.parquet",
    }
    for key, fname in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
        else:
            missing.append(fname)
    if missing:
        st.warning(f"Fichiers manquants : {missing}")
    return data

@st.cache_data
def load_report() -> dict:
    path = os.path.join(REPORTS_DIR, "model_report.json")
    return json.load(open(path)) if os.path.exists(path) else {}

# ---------------------------------------------------------------------------
# SIDEBAR — retourne les filtres actifs
# ---------------------------------------------------------------------------

def build_sidebar(data: dict) -> dict:
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=250)
    else:
        st.sidebar.markdown("## 🏅 YPerf")

    st.sidebar.markdown("---")
    st.sidebar.title("Filtres")

    f = {}

    # ── Sports (utile pour Côtes athlètes)
    all_sports = []
    if "athletes" in data and "Sport" in data["athletes"].columns:
        all_sports = sorted(data["athletes"]["Sport"].dropna().unique())
    f["sports"] = st.sidebar.multiselect(
        "Sport(s)", all_sports,
        default=[],
        placeholder="Tous les sports"
    )

    # ── Profil de pays (utile pour Prédictions + Segmentation)
    cluster_options = ["Tous"]
    if "clusters" in data and not data["clusters"].empty:
        cluster_options += sorted(data["clusters"]["Cluster_Label"].dropna().unique())
    f["cluster"] = st.sidebar.selectbox("Profil de pays", cluster_options)

    # ── Genre (utile pour Côtes athlètes)
    f["sex"] = st.sidebar.radio("Genre", ["Tous", "M", "F"])

    # ── Top N
    f["top_n"] = st.sidebar.slider("Top N à afficher", 5, 50, 20)

    st.sidebar.markdown("---")
    st.sidebar.caption("YPerf · JO 2028 · Los Angeles")
    return f

# ---------------------------------------------------------------------------
# PAGE 1 — Prédictions 2028
# ---------------------------------------------------------------------------

def page_predictions(data: dict, filters: dict, report: dict) -> None:
    st.header("🎯 Prédictions JO 2028 — Los Angeles")

    if "predictions" not in data or data["predictions"].empty:
        st.info("Lance 03_modeling.py pour générer les prédictions.")
        return

    pred = data["predictions"].copy()
    top_n = filters["top_n"]

    # ── Filtre profil de pays
    cluster_filter = filters["cluster"]
    if cluster_filter != "Tous" and "clusters" in data:
        nocs_in_cluster = data["clusters"].loc[
            data["clusters"]["Cluster_Label"] == cluster_filter, "NOC"
        ]
        pred = pred[pred["NOC"].isin(nocs_in_cluster)]

    # ── KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pays analysés", len(pred))
    best_name = report.get("best_model", "")
    c2.metric("Modèle", best_name.replace("_", " ").title() if best_name else "—")
    if best_name and best_name in report:
        c3.metric("R²", f"{report[best_name]['r2_mean']:.3f}")
        c4.metric("MAE", f"{report[best_name]['mae_mean']:.1f} médailles")

    # ── Tableau
    ref_label = f"Paris 2024 (réel)" if (pred.get("Total_Medals") is not None) else "Dernière édition"
    st.subheader(f"Top {top_n} nations — médailles prédites pour 2028")

    disp = pred.head(top_n)[[
        "Rank_2028", "Team", "Total_Medals", "Pred_Medals_2028",
        "Pred_Medals_Low", "Pred_Medals_High", "Delta_vs_Last", "Trend_Label"
    ]].rename(columns={
        "Total_Medals":      "Paris 2024 (réel)",
        "Pred_Medals_2028":  "Prédiction 2028",
        "Pred_Medals_Low":   "Borne basse",
        "Pred_Medals_High":  "Borne haute",
        "Delta_vs_Last":     "Δ vs Paris 2024",
        "Trend_Label":       "Tendance",
    })

    def color_trend(val):
        return {
            "En forte hausse": "background-color:#1a3d2b; color:#4ade80",
            "En hausse":       "background-color:#1a3020; color:#86efac",
            "Stable":          "",
            "En recul":        "background-color:#3d2a0a; color:#fbbf24",
            "En recul fort":   "background-color:#3d1010; color:#f87171",
        }.get(str(val), "")

    st.dataframe(
        disp.style.map(color_trend, subset=["Tendance"]),
        use_container_width=True, height=500,
    )

    # ── Graphique
    st.subheader("Comparaison Paris 2024 (réel) vs 2028 (prédit)")
    chart = pred.head(15).set_index("Team")[["Total_Medals", "Pred_Medals_2028"]]
    chart.columns = ["Paris 2024 (réel)", "2028 (prédit)"]
    st.bar_chart(chart)

# ---------------------------------------------------------------------------
# PAGE 2 — Segmentation pays
# ---------------------------------------------------------------------------

def page_clusters(data: dict, filters: dict) -> None:
    st.header("🗺️ Segmentation des pays")

    if "clusters" not in data or data["clusters"].empty:
        st.info("Lance 03_modeling.py pour générer les clusters.")
        return

    df = data["clusters"].copy()

    # ── Filtre profil
    cluster_filter = filters["cluster"]
    if cluster_filter != "Tous":
        df = df[df["Cluster_Label"] == cluster_filter]

    # ── Distribution (toujours sur tout le dataset pour le graphique)
    st.subheader("Distribution par profil (tous pays)")
    dist = data["clusters"]["Cluster_Label"].value_counts().reset_index()
    dist.columns = ["Profil", "Nombre de pays"]
    st.bar_chart(dist.set_index("Profil"))

    # ── Tableau filtré
    st.subheader(f"Pays — profil : {cluster_filter}")
    cols = ["NOC", "Team", "Cluster_Label", "Medal_Score_ma3", "Athletes_ma3", "Streak"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols].rename(columns={
            "Cluster_Label":   "Profil",
            "Medal_Score_ma3": "Score pondéré (moy.)",
            "Athletes_ma3":    "Athlètes (moy.)",
            "Streak":          "Série médailles",
        }).sort_values("Score pondéré (moy.)", ascending=False),
        use_container_width=True, height=500,
    )

    scatter = os.path.join(REPORTS_DIR, "cluster_scatter.png")
    if os.path.exists(scatter):
        st.image(scatter, caption="Scatter plot des clusters", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3 — Côtes athlètes
# ---------------------------------------------------------------------------

def page_athletes(data: dict, filters: dict) -> None:
    st.header("🏃 Côtes des athlètes")

    if "athletes" not in data or data["athletes"].empty:
        st.info("Lance 03_modeling.py pour générer les scores athlètes.")
        return

    df = data["athletes"].copy()
    top_n = filters["top_n"]

    # ── Filtre genre
    sex = filters["sex"]
    if sex != "Tous" and "Sex" in df.columns:
        df = df[df["Sex"] == sex]

    # ── Filtre sport(s)
    sports = filters["sports"]
    if sports and "Sport" in df.columns:
        df = df[df["Sport"].isin(sports)]

    st.subheader(f"Top {top_n} athlètes — score YPerf")

    if df.empty:
        st.warning("Aucun athlète ne correspond aux filtres sélectionnés.")
        return

    cols = ["Name", "NOC", "Sport", "Sex", "Gold", "Silver", "Bronze",
            "Participations", "Last_Year", "Athlete_Score_Norm", "Cote_Label"]
    cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df.head(top_n)[cols].rename(columns={
            "Athlete_Score_Norm": "Score YPerf (/100)",
            "Cote_Label":         "Côte",
            "Last_Year":          "Dernière participation",
            "Participations":     "Éditions",
            "Sex":                "Genre",
        }),
        use_container_width=True, height=520,
    )

    # ── Distribution des côtes (sur la sélection filtrée)
    st.subheader("Distribution des côtes (sélection filtrée)")
    if "Cote_Label" in df.columns:
        cote_dist = df.head(top_n)["Cote_Label"].value_counts().reset_index()
        cote_dist.columns = ["Côte", "Nb athlètes"]
        st.bar_chart(cote_dist.set_index("Côte"))

    # ── Top sports de la sélection
    if "Sport" in df.columns and not sports:
        st.subheader("Répartition par sport (top 10)")
        sport_dist = df["Sport"].value_counts().head(10).reset_index()
        sport_dist.columns = ["Sport", "Nb athlètes"]
        st.bar_chart(sport_dist.set_index("Sport"))

# ---------------------------------------------------------------------------
# PAGE 4 — Métriques modèles
# ---------------------------------------------------------------------------

def page_metrics(report: dict) -> None:
    st.header("🔬 Performance des modèles")

    if not report:
        st.info("Lance 03_modeling.py pour générer le rapport.")
        return

    best = report.get("best_model", "")
    st.success(f"Meilleur modèle : **{best.replace('_', ' ').title()}**")

    rows = [
        {
            "Modèle":      n.replace("_", " ").title(),
            "R² moyen":    f"{m['r2_mean']:.4f}",
            "R² std":      f"± {m['r2_std']:.4f}",
            "MAE moyenne": f"{m['mae_mean']:.2f}",
            "✓":           "✅" if n == best else "",
        }
        for n, m in report.items()
        if isinstance(m, dict) and "r2_mean" in m
    ]
    if rows:
        st.table(pd.DataFrame(rows))

    st.caption(f"Features : {', '.join(report.get('feature_cols', []))}")
    st.caption("Validation : TimeSeriesSplit 5 folds (ordre chronologique respecté).")

    c1, c2 = st.columns(2)
    fi = os.path.join(REPORTS_DIR, "feature_importance.png")
    el = os.path.join(REPORTS_DIR, "kmeans_elbow.png")
    if os.path.exists(fi):
        c1.image(fi, caption="Importance des variables")
    if os.path.exists(el):
        c2.image(el, caption="Méthode du coude K-Means")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    data    = load_data()
    report  = load_report()
    filters = build_sidebar(data)

    pages = {
        "🎯 Prédictions 2028":  lambda: page_predictions(data, filters, report),
        "🗺️ Segmentation pays": lambda: page_clusters(data, filters),
        "🏃 Côtes athlètes":    lambda: page_athletes(data, filters),
        "🔬 Métriques modèles": lambda: page_metrics(report),
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
