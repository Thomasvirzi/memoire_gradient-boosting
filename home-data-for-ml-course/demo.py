import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

# ── Palette couleurs ────────────────────────────────────────────────────────
C_BLUE   = "#457b9d"
C_RED    = "#e63946"
C_DARK   = "#1d3557"
C_GREEN  = "#2d6a4f"
C_GRAY   = "#6b7280"
C_LIGHT  = "#f1faee"

st.set_page_config(
    page_title="Démo – Gradient Boosting",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #f8fafc; }
    h1 { color: #1d3557; }
    h2 { color: #1d3557; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 5px solid #457b9d;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 12px;
        color: #1d3557;
    }
    .metric-label { font-size: 0.78rem; color: #4b5563; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1d3557; }
    .metric-sub   { font-size: 0.82rem; color: #2563a8; }
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #457b9d;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #1d3557;
    }
    .info-box b { color: #1d3557; }
    .highlight-box {
        background: #fff3e0;
        border-left: 4px solid #f97316;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #7c2d12;
    }
    .highlight-box b { color: #7c2d12; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSES FROM SCRATCH  (identiques au notebook)
# ══════════════════════════════════════════════════════════════════════════════

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold   = None
        self.left_value  = None
        self.right_value = None

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_mse = np.inf
        for fi in range(n_features):
            for threshold in np.unique(X[:, fi]):
                lm = X[:, fi] <= threshold
                rm = ~lm
                if lm.sum() == 0 or rm.sum() == 0:
                    continue
                yl, yr = y[lm], y[rm]
                mse_s = len(yl)/n_samples*self._mse(yl) + len(yr)/n_samples*self._mse(yr)
                if mse_s < best_mse:
                    best_mse         = mse_s
                    self.feature_idx = fi
                    self.threshold   = threshold
                    self.left_value  = np.mean(yl)
                    self.right_value = np.mean(yr)

    def predict(self, X):
        lm = X[:, self.feature_idx] <= self.threshold
        return np.where(lm, self.left_value, self.right_value)


class GradientBoostingScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.stumps        = []
        self.init_value    = None
        self.train_mse_history = []
        self.test_mse_history  = []

    def fit(self, X_tr, y_tr, X_te=None, y_te=None):
        self.init_value = np.mean(y_tr)
        pred_tr = np.full(len(y_tr), self.init_value)
        pred_te = np.full(len(y_te), self.init_value) if X_te is not None else None
        for _ in range(self.n_estimators):
            residus = y_tr - pred_tr
            s = DecisionStump()
            s.fit(X_tr, residus)
            self.stumps.append(s)
            pred_tr = pred_tr + self.learning_rate * s.predict(X_tr)
            self.train_mse_history.append(np.mean((pred_tr - y_tr)**2))
            if X_te is not None:
                pred_te = pred_te + self.learning_rate * s.predict(X_te)
                self.test_mse_history.append(np.mean((pred_te - y_te)**2))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_value)
        for s in self.stumps:
            pred += self.learning_rate * s.predict(X)
        return pred


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION DES DONNÉES  (caché pour vitesse)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Chargement et préparation des données…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    FEATURES = [
        "GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt",
        "GarageCars", "FullBath", "LotArea", "YearRemodAdd", "Neighborhood",
    ]
    TARGET = "SalePrice"
    df_model = df[FEATURES + [TARGET]].copy()
    for col in df_model.select_dtypes(include="number").columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    df_model["Neighborhood"] = df_model["Neighborhood"].fillna(
        df_model["Neighborhood"].mode()[0])
    df_model = pd.get_dummies(df_model, columns=["Neighborhood"], drop_first=True)
    df_model[TARGET] = np.log1p(df_model[TARGET])
    feature_cols = [c for c in df_model.columns if c != TARGET]
    X = df_model[feature_cols].values
    y = df_model[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_cols


@st.cache_data(show_spinner="Entraînement du Gradient Boosting from scratch…")
def train_gb_scratch(n_estimators, learning_rate):
    X_train, X_test, y_train, y_test, _ = load_data()
    gb = GradientBoostingScratch(n_estimators=n_estimators, learning_rate=learning_rate)
    gb.fit(X_train, y_train, X_test, y_test)
    return gb.train_mse_history, gb.test_mse_history, gb.predict(X_test), y_test


@st.cache_data(show_spinner="Calcul des benchmarks sklearn…")
def compute_benchmarks():
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    # Régression linéaire (toutes features)
    lr = LinearRegression().fit(X_train, y_train)
    mse_lr = mean_squared_error(y_test, lr.predict(X_test))

    # Decision stump seul
    stump = DecisionStump()
    stump.fit(X_train, y_train)
    mse_stump = mean_squared_error(y_test, stump.predict(X_test))

    # GB sklearn iso-paramètres (depth=1, 100 arbres, lr=0.1)
    gb_iso = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
    gb_iso.fit(X_train, y_train)
    mse_gb_iso = mean_squared_error(y_test, gb_iso.predict(X_test))

    # GB sklearn optimal (depth=3, 200 arbres, lr=0.05)
    gb_opt = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    gb_opt.fit(X_train, y_train)
    mse_gb_opt = mean_squared_error(y_test, gb_opt.predict(X_test))
    fi_gb = gb_opt.feature_importances_

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    mse_rf = mean_squared_error(y_test, rf.predict(X_test))

    return {
        "mse_lr": mse_lr,
        "mse_stump": mse_stump,
        "mse_gb_iso": mse_gb_iso,
        "mse_gb_opt": mse_gb_opt,
        "mse_rf": mse_rf,
        "fi_gb": fi_gb,
        "feature_cols": feature_cols,
        "gb_opt": gb_opt,
        "X_test": X_test,
        "y_test": y_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🌲 Gradient Boosting — Application interactive")
st.markdown(
    "**Mémoire M1 Data Science** · Démonstration des concepts étudiés sur le dataset "
    "*Ames Housing (Kaggle House Prices)*  \n"
    "Implémentation from-scratch vs scikit-learn · Comparaison des modèles · Feature importance"
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs([
    "📈  Onglet 1 — Convergence du Gradient Boosting",
    "📊  Onglet 2 — Comparaison des modèles",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLET 1 — CONVERGENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    st.subheader("Visualisation de la convergence du Gradient Boosting from scratch")
    st.markdown(
        "Utilisez les curseurs pour explorer comment le nombre d'arbres et le learning rate "
        "influencent la convergence — et l'apparition du surapprentissage."
    )

    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### ⚙️ Paramètres")
        n_estimators = st.slider(
            "Nombre d'arbres (stumps)", min_value=5, max_value=300,
            value=100, step=5,
            help="Chaque arbre corrige les résidus du précédent."
        )
        learning_rate = st.select_slider(
            "Learning rate (η)",
            options=[0.01, 0.05, 0.10, 0.20, 0.50],
            value=0.10,
            help="Poids donné à chaque arbre. Faible = apprentissage lent mais stable."
        )
        st.markdown(
            f"""<div class="info-box">
            <b>Convergence attendue :</b><br>
            Avec η={learning_rate} et {n_estimators} arbres, 
            {'le modèle est probablement en <b>sous-apprentissage</b> — augmentez les arbres.' 
             if n_estimators < 30 else
             'la convergence est en cours.' 
             if n_estimators < 150 else
             'le <b>surapprentissage</b> peut apparaître — regardez l\'écart train/test.'}
            </div>""",
            unsafe_allow_html=True
        )

    with col_info:
        st.markdown("#### 💡 Principe")
        st.markdown("""
        Le **Gradient Boosting** construit des arbres *séquentiellement* :  
        chaque arbre apprend à corriger les **résidus** (erreurs) du modèle précédent.

        - **Learning rate faible** → corrections prudentes, plus d'arbres nécessaires  
        - **Trop d'arbres** → le modèle mémorise le bruit → surapprentissage (test MSE remonte)  
        - **Point optimal** = là où le MSE test est minimal (early stopping)
        """)

    # Entraînement
    with st.spinner("Entraînement en cours…"):
        train_hist, test_hist, y_pred_scratch, y_test_ref = train_gb_scratch(
            n_estimators, learning_rate)

    # Métriques rapides
    best_iter = int(np.argmin(test_hist)) + 1
    best_mse  = float(min(test_hist))
    final_mse = float(test_hist[-1])
    gap       = final_mse - best_mse

    m1, m2, m3, m4 = st.columns(4)
    def metric_card(col, label, value, sub=""):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>',
            unsafe_allow_html=True)

    metric_card(m1, "MSE test final", f"{final_mse:.5f}", f"après {n_estimators} arbres")
    metric_card(m2, "MSE test optimal", f"{best_mse:.5f}", f"arbre n°{best_iter}")
    metric_card(m3, "Surapprentissage", f"+{gap:.5f}",
                "0 = pas de sur-fit" if gap < 0.002 else "⚠️ écart train/test croissant")
    metric_card(m4, "RMSE test", f"{np.sqrt(final_mse):.4f}", "en log(SalePrice)")

    # ── Graphiques ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    iters = range(1, n_estimators + 1)

    # — Courbe de convergence ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(iters, train_hist, color=C_BLUE, lw=2, label="MSE Train")
    ax.plot(iters, test_hist,  color=C_RED,  lw=2, label="MSE Test")
    ax.axvline(best_iter, color=C_GREEN, lw=2, linestyle=":", label=f"Optimal (n°{best_iter})")
    ax.axhline(0.0238, color=C_GRAY, lw=1.2, linestyle="--", alpha=0.7,
               label="Régression linéaire (réf.)")

    # zone surapprentissage
    if best_iter < n_estimators:
        ax.axvspan(best_iter, n_estimators, alpha=0.08, color=C_RED, label="Zone surapprentissage")

    ax.set_title("MSE en fonction du nombre d'arbres", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Nombre d'arbres", fontsize=11)
    ax.set_ylabel("MSE (log-SalePrice²)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#fafafa")

    # — Prédictions vs réalité ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(y_test_ref, y_pred_scratch, alpha=0.45, s=18,
                color=C_BLUE, edgecolors="none", label="Prédictions GB scratch")
    lims = [min(y_test_ref.min(), y_pred_scratch.min()) - 0.05,
            max(y_test_ref.max(), y_pred_scratch.max()) + 0.05]
    ax2.plot(lims, lims, color=C_RED, lw=2, linestyle="--", label="Prédiction parfaite")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title(f"Prédictions vs Valeurs réelles ({n_estimators} arbres)",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("log(SalePrice) réel", fontsize=11)
    ax2.set_ylabel("log(SalePrice) prédit", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#fafafa")

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    # ── Distribution des résidus ─────────────────────────────────────────
    st.markdown("#### Distribution des résidus (jeu de test)")
    residus = y_pred_scratch - y_test_ref
    fig2, ax3 = plt.subplots(figsize=(10, 3.5), facecolor="white")
    ax3.hist(residus, bins=50, color=C_BLUE, edgecolor="white", alpha=0.85)
    ax3.axvline(0, color=C_RED, lw=2.5, linestyle="--", label="Zéro (erreur nulle)")
    ax3.axvline(np.mean(residus), color=C_GREEN, lw=2, linestyle="-.",
                label=f"Moyenne = {np.mean(residus):.4f}")
    ax3.set_title("Distribution des résidus — Gradient Boosting from scratch", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Résidu (ŷ − y)", fontsize=10)
    ax3.set_ylabel("Fréquence", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown(
        f"""<div class="highlight-box">
        <b>📌 À retenir :</b> Avec {n_estimators} arbres et η={learning_rate}, 
        le MSE optimal est <b>{best_mse:.5f}</b> (atteint à l'arbre n°{best_iter}).
        {"Le modèle est encore en apprentissage — pas de surapprentissage détecté." 
         if gap < 0.001 else
         f"Un surapprentissage de +{gap:.5f} est observé après le point optimal."}
        La régression linéaire de référence atteint MSE ≈ 0.024 avec 9 features.
        </div>""",
        unsafe_allow_html=True
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLET 2 — COMPARAISON DES MODÈLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.subheader("Comparaison des modèles sur le jeu de test")
    st.markdown(
        "Performances des quatre modèles étudiés dans le mémoire, "
        "entraînés sur le jeu Ames Housing (log-SalePrice, 80% train / 20% test)."
    )

    bench = compute_benchmarks()

    # Recalcul propre pour le tableau
    models_data = [
        ("Decision Stump (depth=1)", bench["mse_stump"], np.sqrt(bench["mse_stump"]), "From scratch"),
        ("Régression Linéaire (9 features)", bench["mse_lr"], np.sqrt(bench["mse_lr"]), "sklearn"),
        ("Random Forest (200 arbres)", bench["mse_rf"], np.sqrt(bench["mse_rf"]), "sklearn"),
        ("Gradient Boosting iso-param.\n(100 stumps, depth=1, lr=0.1)", bench["mse_gb_iso"], np.sqrt(bench["mse_gb_iso"]), "sklearn"),
        ("Gradient Boosting optimal\n(200 arbres, depth=3, lr=0.05)", bench["mse_gb_opt"], np.sqrt(bench["mse_gb_opt"]), "sklearn ★ meilleur"),
    ]

    # ── Tableau de performances ───────────────────────────────────────────
    st.markdown("#### 📋 Tableau des performances")
    df_results = pd.DataFrame(models_data, columns=["Modèle", "MSE test", "RMSE test", "Type"])
    df_results = df_results.sort_values("MSE test")
    df_results["MSE test"]  = df_results["MSE test"].map("{:.6f}".format)
    df_results["RMSE test"] = df_results["RMSE test"].map("{:.5f}".format)

    st.dataframe(
        df_results.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modèle":    st.column_config.TextColumn("Modèle", width="large"),
            "MSE test":  st.column_config.TextColumn("MSE test ↓"),
            "RMSE test": st.column_config.TextColumn("RMSE test ↓"),
            "Type":      st.column_config.TextColumn("Implémentation"),
        }
    )

    # ── Graphiques ───────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 📊 Comparaison MSE (jeu de test)")
        labels = [
            "Decision Stump",
            "Régression\nLinéaire",
            "Random\nForest",
            "GB iso-param.\n(depth=1)",
            "GB optimal\n(depth=3)",
        ]
        values = [
            bench["mse_stump"],
            bench["mse_lr"],
            bench["mse_rf"],
            bench["mse_gb_iso"],
            bench["mse_gb_opt"],
        ]
        colors = [C_RED, C_GRAY, C_BLUE, C_BLUE, C_GREEN]
        alphas = [0.9, 0.7, 0.75, 0.85, 1.0]

        fig3, ax4 = plt.subplots(figsize=(7, 5), facecolor="white")
        bars = ax4.barh(labels, values, color=colors, edgecolor="white", height=0.55)

        for bar, val, a in zip(bars, values, alphas):
            bar.set_alpha(a)
            ax4.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                     f"{val:.5f}", va="center", fontsize=9.5, fontweight="bold")

        ax4.axvline(bench["mse_gb_opt"], color=C_GREEN, lw=1.5, linestyle="--", alpha=0.6)
        ax4.set_xlabel("MSE (log-SalePrice²) — plus bas = meilleur", fontsize=10)
        ax4.set_title("MSE sur le jeu de test", fontsize=12, fontweight="bold", pad=10)
        ax4.grid(True, axis="x", alpha=0.3)
        ax4.set_facecolor("#fafafa")
        ax4.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_g2:
        st.markdown("#### 🔍 Feature Importance — GB Optimal (depth=3)")
        fcols  = bench["feature_cols"]

        # Top 12 features
        fi_arr  = bench["fi_gb"]
        fi_idx  = np.argsort(fi_arr)[::-1][:12]
        fi_vals = fi_arr[fi_idx]
        fi_names = [fcols[i] if len(fcols[i]) <= 18 else fcols[i][:16]+"…" for i in fi_idx]

        fig4, ax5 = plt.subplots(figsize=(7, 5), facecolor="white")
        bar_colors = [C_DARK if v == fi_vals.max() else C_BLUE for v in fi_vals]
        bars2 = ax5.barh(fi_names[::-1], fi_vals[::-1],
                         color=bar_colors[::-1], edgecolor="white", height=0.6)
        for bar in bars2:
            bar.set_alpha(0.85)

        ax5.set_xlabel("Importance (gain)", fontsize=10)
        ax5.set_title("Top 12 features — Gradient Boosting\n(200 arbres, depth=3, lr=0.05)",
                      fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, axis="x", alpha=0.3)
        ax5.set_facecolor("#fafafa")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # ── Prédictions vs réalité — 4 modèles côte à côte ──────────────────
    st.markdown("#### 🔬 Prédictions vs Valeurs réelles — comparaison par modèle")

    X_tr, X_te, y_tr, y_te, fcols = load_data()

    stump_model = DecisionStump()
    stump_model.fit(X_tr, y_tr)

    lr_model = LinearRegression().fit(X_tr, y_tr)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    gb_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42).fit(X_tr, y_tr)

    preds = {
        "Decision Stump": stump_model.predict(X_te),
        "Régression Linéaire": lr_model.predict(X_te),
        "Random Forest": rf_model.predict(X_te),
        "GB sklearn optimal": gb_model.predict(X_te),
    }
    violin_colors = [C_RED, C_GRAY, C_BLUE, C_GREEN]

    # ── Violin : distribution des résidus ─────────────────────────────────
    fig_violin = go.Figure()
    for (name, ypred), color in zip(preds.items(), violin_colors):
        fig_violin.add_trace(go.Violin(
            y=(ypred - y_te).tolist(),
            name=name,
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color,
            opacity=0.55,
            points="outliers",
            marker=dict(size=3, opacity=0.4),
        ))
    fig_violin.add_hline(y=0, line_dash="dash", line_color="#1d3557", line_width=1.5,
                         annotation_text="Résidu = 0", annotation_position="top right",
                         annotation_font_color="#1d3557")
    fig_violin.update_layout(
        title=dict(text="Distribution des résidus (ŷ − y) — plus la boîte est étroite, meilleur est le modèle",
                   font=dict(size=13, color="#1d3557")),
        yaxis=dict(title="Résidu (ŷ − y)", gridcolor="#e5e7eb", zerolinecolor="#1d3557",
                   tickfont=dict(color="#1d3557"), titlefont=dict(color="#1d3557")),
        xaxis=dict(tickfont=dict(color="#1d3557")),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(color="#1d3557"),
        showlegend=False,
        violingap=0.25,
        margin=dict(t=55, b=30, l=60, r=20),
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    # ── Scatter 2×2 colorés par |résidu| ──────────────────────────────────
    model_list = list(preds.items())
    fig_sc = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"<b>{name}</b>  —  MSE = {mean_squared_error(y_te, yp):.5f}"
            for name, yp in model_list
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    global_cmax = max(float(np.abs(yp - y_te).max()) for _, yp in model_list)

    for (name, ypred), (row, col) in zip(model_list, positions):
        abs_res = np.abs(ypred - y_te)
        lims = [min(float(y_te.min()), float(ypred.min())) - 0.05,
                max(float(y_te.max()), float(ypred.max())) + 0.05]
        show_scale = (row == 2 and col == 2)

        fig_sc.add_trace(go.Scatter(
            x=y_te.tolist(), y=ypred.tolist(),
            mode="markers",
            marker=dict(
                size=5,
                color=abs_res.tolist(),
                colorscale="RdYlGn_r",
                cmin=0, cmax=global_cmax,
                showscale=show_scale,
                colorbar=dict(title="|Résidu|", thickness=12, len=0.45,
                              y=0.25, tickfont=dict(color="#1d3557"),
                              titlefont=dict(color="#1d3557")) if show_scale else {},
                opacity=0.75,
            ),
            showlegend=False,
            hovertemplate="Réel: %{x:.3f}<br>Prédit: %{y:.3f}<br>|Résidu|: %{marker.color:.3f}<extra></extra>",
        ), row=row, col=col)

        fig_sc.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#1d3557", dash="dash", width=1.5),
            showlegend=False,
        ), row=row, col=col)

        fig_sc.update_xaxes(range=lims, title_text="log(SalePrice) réel",
                            gridcolor="#e5e7eb", tickfont=dict(color="#1d3557"),
                            titlefont=dict(color="#1d3557"), row=row, col=col)
        fig_sc.update_yaxes(range=lims, title_text="log(SalePrice) prédit",
                            gridcolor="#e5e7eb", tickfont=dict(color="#1d3557"),
                            titlefont=dict(color="#1d3557"), row=row, col=col)

    fig_sc.update_layout(
        height=660,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(color="#1d3557", size=11),
        margin=dict(t=60, b=40, l=60, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Interprétation ───────────────────────────────────────────────────
    st.markdown("#### 💬 Interprétation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="info-box">
            <b>Pourquoi le GB surpasse la régression linéaire ?</b><br>
            La régression linéaire suppose une relation <i>linéaire additive</i> entre features et cible.
            Le Gradient Boosting, via des arbres de décision, capture les <b>interactions non-linéaires</b>
            entre features (ex. OverallQual × GrLivArea) — impossible à modéliser linéairement.
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="info-box">
            <b>Rôle de la profondeur (depth) :</b><br>
            Un stump (depth=1) ne peut capturer qu'<i>une seule interaction à la fois</i>.
            Avec depth=3, chaque arbre peut modéliser jusqu'à 2³=8 régions — permettant de 
            capturer des <b>interactions multi-features</b> et de réduire le biais structurel.
            MSE : {bench['mse_gb_iso']:.5f} (depth=1) → {bench['mse_gb_opt']:.5f} (depth=3).
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        f"""<div class="highlight-box">
        <b>🏆 Résultat clé du mémoire :</b> Le GB sklearn optimal (depth=3, 200 arbres, lr=0.05) 
        atteint MSE = <b>{bench['mse_gb_opt']:.5f}</b>, soit une réduction de 
        <b>{(1 - bench['mse_gb_opt']/bench['mse_lr'])*100:.1f}%</b> par rapport à la régression linéaire 
        (MSE = {bench['mse_lr']:.5f}). La feature la plus importante est <b>OverallQual</b> 
        (qualité générale), confirmant l'intuition humaine sur le marché immobilier.
        </div>""",
        unsafe_allow_html=True
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Mémoire M1 Data Science · Gradient Boosting · Dataset : Ames Housing (Kaggle) · "
    "Références : Chen & Guestrin (2016), Friedman (2001)</small>",
    unsafe_allow_html=True
)