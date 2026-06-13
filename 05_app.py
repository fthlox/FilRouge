"""
YPerf — Point 5 : Application Streamlit (déploiement local)
============================================================
Lance avec :
  streamlit run 05_app.py

Ce fichier constitue le point d'entrée du déploiement local.
Il s'intègre avec les dashboards existants (points 2 & 4) en chargeant
les données produites par les pipelines 01 et 03.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="YPerf — JO 2028",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


# ---------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES (mise en cache)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> dict:
    """Charge tous les fichiers Parquet produits par les pipelines 01 et 03."""
    data = {}
    files = {
        "predictions": "predictions_2028.parquet",
        "clusters": "country_clusters.parquet",
        "athletes": "athlete_scores.parquet",
        "country_year": "country_year_stats.parquet",
        "sport_year": "sport_year_stats.parquet",
    }
    missing = []
    for key, fname in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
        else:
            missing.append(fname)
    if missing:
        st.warning(f"Fichiers manquants (lancez les pipelines 01 et 03 d'abord) : {missing}")
    return data


@st.cache_data
def load_model_report() -> dict:
    path = os.path.join(REPORTS_DIR, "model_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def sidebar(data: dict) -> dict:
    st.sidebar.image("Yperf.png", width=250)
    st.sidebar.title("Filtres")

    filters = {}

    # Filtre sport
    if "sport_year" in data and not data["sport_year"].empty:
        sports = sorted(data["sport_year"]["Sport"].unique())
        filters["sport"] = st.sidebar.multiselect("Sport(s)", sports, default=sports[:5])

    # Filtre pays / cluster
    if "clusters" in data and not data["clusters"].empty:
        clusters = sorted(data["clusters"]["Cluster_Label"].dropna().unique())
        filters["cluster"] = st.sidebar.selectbox("Profil de pays", ["Tous"] + list(clusters))

    # Filtre genre (si présent)
    if "athletes" in data and "Sex" in data["athletes"].columns:
        filters["sex"] = st.sidebar.radio("Genre", ["Tous", "M", "F"])

    # Nombre de pays à afficher
    filters["top_n"] = st.sidebar.slider("Top N pays", 5, 50, 20)

    return filters


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def page_predictions(data: dict, filters: dict, report: dict) -> None:
    st.header("🎯 Prédictions JO 2028 — Los Angeles")

    if "predictions" not in data or data["predictions"].empty:
        st.info("Lance le pipeline 03_modeling.py pour générer les prédictions.")
        return

    pred = data["predictions"].copy()
    top_n = filters.get("top_n", 20)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pays analysés", len(pred))
    col2.metric("Meilleur modèle", report.get("best_model", "N/A").replace("_", " ").title())
    best_model_name = report.get("best_model", "")
    if best_model_name in report:
        col3.metric("R² (validation)", f"{report[best_model_name]['r2_mean']:.3f}")
        col4.metric("MAE moyenne", f"{report[best_model_name]['mae_mean']:.1f} médailles")

    st.subheader(f"Top {top_n} nations — médailles prédites")
    display = pred.head(top_n)[
        ["Rank_2028", "Team", "Total_Medals", "Pred_Medals_2028",
         "Pred_Medals_Low", "Pred_Medals_High", "Delta_vs_Last", "Trend_Label"]
    ].rename(columns={
        "Total_Medals": "Médailles Tokyo 2020",
        "Pred_Medals_2028": "Prédiction 2028",
        "Pred_Medals_Low": "Borne basse",
        "Pred_Medals_High": "Borne haute",
        "Delta_vs_Last": "Δ vs Tokyo",
        "Trend_Label": "Tendance",
    })

    def color_trend(val):
        colors = {
            "En forte hausse": "background-color: #0b7511",
            "En hausse": "background-color: #42b337",
            "Stable": "",
            "En recul": "background-color: #f5674c",
            "En recul fort": "background-color: #cf0808",
        }
        return colors.get(str(val), "")

    st.dataframe(
        display.style.applymap(color_trend, subset=["Tendance"]),
        use_container_width=True,
        height=550,
    )

    # Graphique barres
    st.subheader("Comparaison prédiction vs réalité Tokyo")
    chart_data = pred.head(15).set_index("Team")[["Total_Medals", "Pred_Medals_2028"]]
    chart_data.columns = ["Tokyo 2020 (réel)", "2028 (prédit)"]
    st.bar_chart(chart_data)


def page_clusters(data: dict, filters: dict) -> None:
    st.header("🗺️ Segmentation des pays")

    if "clusters" not in data or data["clusters"].empty:
        st.info("Lance le pipeline 03_modeling.py pour générer les clusters.")
        return

    df = data["clusters"].copy()
    cluster_filter = filters.get("cluster", "Tous")
    if cluster_filter != "Tous":
        df = df[df["Cluster_Label"] == cluster_filter]

    st.subheader("Distribution par profil")
    dist = df["Cluster_Label"].value_counts().reset_index()
    dist.columns = ["Profil", "Nombre de pays"]
    st.bar_chart(dist.set_index("Profil"))

    st.subheader("Détail par pays")
    st.dataframe(
        df[["NOC", "Team", "Cluster_Label", "Medal_Score_ma3", "Athletes_ma3", "Streak"]]
        .rename(columns={
            "Cluster_Label": "Profil",
            "Medal_Score_ma3": "Score pondéré (moy.)",
            "Athletes_ma3": "Athlètes (moy.)",
            "Streak": "Série médailles",
        })
        .sort_values("Score pondéré (moy.)", ascending=False),
        use_container_width=True,
        height=500,
    )

    if os.path.exists(os.path.join(REPORTS_DIR, "cluster_scatter.png")):
        st.image(os.path.join(REPORTS_DIR, "cluster_scatter.png"), caption="Scatter plot des clusters", use_column_width=True)


def page_athletes(data: dict, filters: dict) -> None:
    st.header("🏃 Côtes des athlètes")

    if "athletes" not in data or data["athletes"].empty:
        st.info("Lance le pipeline 03_modeling.py pour générer les scores athlètes.")
        return

    df = data["athletes"].copy()

    # Filtre genre
    sex_filter = filters.get("sex", "Tous")
    if sex_filter != "Tous" and "Sex" in df.columns:
        df = df[df["Sex"] == sex_filter]

    top_n = filters.get("top_n", 20)

    st.subheader(f"Top {top_n} athlètes — score YPerf")
    cols_display = ["Name", "NOC", "Sport", "Gold", "Silver", "Bronze",
                    "Participations", "Last_Year", "Athlete_Score_Norm", "Cote_Label"]
    cols_display = [c for c in cols_display if c in df.columns]

    st.dataframe(
        df.head(top_n)[cols_display].rename(columns={
            "Athlete_Score_Norm": "Score YPerf (/100)",
            "Cote_Label": "Côte",
            "Last_Year": "Dernière participation",
            "Participations": "Éditions",
        }),
        use_container_width=True,
        height=550,
    )

    st.subheader("Distribution des côtes")
    if "Cote_Label" in df.columns:
        cote_dist = df["Cote_Label"].value_counts().reset_index()
        cote_dist.columns = ["Côte", "Nb athlètes"]
        st.bar_chart(cote_dist.set_index("Côte"))


def page_model_metrics(report: dict) -> None:
    st.header("🔬 Performance des modèles")

    if not report:
        st.info("Lance le pipeline 03_modeling.py pour générer le rapport de métriques.")
        return

    best = report.get("best_model", "")
    st.success(f"Meilleur modèle sélectionné : **{best.replace('_', ' ').title()}**")

    rows = []
    for name, metrics in report.items():
        if isinstance(metrics, dict) and "r2_mean" in metrics:
            rows.append({
                "Modèle": name.replace("_", " ").title(),
                "R² moyen": f"{metrics['r2_mean']:.4f}",
                "R² std": f"± {metrics['r2_std']:.4f}",
                "MAE moyenne": f"{metrics['mae_mean']:.2f}",
                "Sélectionné": "✅" if name == best else "",
            })
    if rows:
        st.table(pd.DataFrame(rows))

    st.caption(f"Variables d'entrée : {', '.join(report.get('feature_cols', []))}")
    st.caption("Validation croisée temporelle (TimeSeriesSplit, 5 folds).")

    col1, col2 = st.columns(2)
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    el_path = os.path.join(REPORTS_DIR, "kmeans_elbow.png")
    if os.path.exists(fi_path):
        col1.image(fi_path, caption="Importance des variables")
    if os.path.exists(el_path):
        col2.image(el_path, caption="Méthode du coude (K-Means)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_data()
    report = load_model_report()
    filters = sidebar(data)

    pages = {
        "🎯 Prédictions 2028": lambda: page_predictions(data, filters, report),
        "🗺️ Segmentation pays": lambda: page_clusters(data, filters),
        "🏃 Côtes athlètes": lambda: page_athletes(data, filters),
        "🔬 Métriques modèles": lambda: page_model_metrics(report),
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()

    st.sidebar.markdown("---")
    st.sidebar.caption("YPerf · OuzLey · JO 2028 LA")


if __name__ == "__main__":
    main()
