import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree as sk_plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#e2e8f0",
    "axes.labelcolor":   "#334155",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#0f172a",
    "grid.color":        "#e2e8f0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.55,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#e2e8f0",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_DARK   = "#0f172a"
C_GREEN  = "#16a34a"
C_GRAY   = "#64748b"
C_PURPLE = "#7c3aed"

st.set_page_config(
    page_title="Gradient Boosting — Mémoire M1",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --c-primary:    #2563eb;
    --c-success:    #16a34a;
    --c-danger:     #dc2626;
    --c-warning:    #d97706;
    --c-purple:     #7c3aed;
    --c-dark:       #0f172a;
    --c-dark-800:   #1e293b;
    --c-text:       #334155;
    --c-muted:      #64748b;
    --c-border:     #e2e8f0;
    --c-surface:    #ffffff;
    --c-bg:         #f1f5f9;
    --r-md: 10px; --r-lg: 14px; --r-xl: 20px;
    --s-sm: 0 1px 3px rgba(15,23,42,.07);
    --s-md: 0 4px 14px rgba(15,23,42,.09);
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
}

*, html, body { font-family: var(--font) !important; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: var(--c-bg) !important; }
[data-testid="stHeader"] { background: var(--c-surface); border-bottom: 1px solid var(--c-border); }

h1, h2, h3 {
    color: var(--c-dark) !important; font-weight: 700 !important;
    letter-spacing: -.022em; font-family: var(--font) !important;
}
h4, h5 { color: var(--c-dark-800) !important; font-weight: 600 !important; }
p, li   { color: var(--c-text) !important; line-height: 1.65; }
label   { color: var(--c-dark-800) !important; font-weight: 500; }
.stMarkdown p, .stMarkdown li { color: var(--c-text) !important; }
[data-testid="stHeadingWithActionElements"] * { color: var(--c-dark) !important; }

/* ── Hero ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1a3561 55%, #1d4ed8 100%);
    border-radius: var(--r-xl); padding: 46px 42px; margin-bottom: 26px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse at 80% 15%, rgba(99,102,241,.32) 0%, transparent 55%),
        radial-gradient(ellipse at 18% 85%, rgba(37,99,235,.22) 0%, transparent 50%);
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,.25); border: 1px solid rgba(165,180,252,.38);
    color: #a5b4fc !important; font-size: .73rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 14px;
}
.hero-title {
    font-size: 2.75rem; font-weight: 800 !important;
    color: #fff !important; letter-spacing: -.04em; line-height: 1.1;
    margin: 0 0 10px; font-family: var(--font) !important;
}
.hero-subtitle {
    color: #fff !important; font-size: 1.02rem;
    line-height: 1.65; margin: 0 0 30px; max-width: 560px;
}
.hero-stats { display: flex; gap: 36px; flex-wrap: wrap; }
.hero-stat  { display: flex; flex-direction: column; gap: 2px; }
.hero-stat-num {
    font-size: 1.88rem; font-weight: 700; color: #fff !important;
    letter-spacing: -.025em; font-family: var(--mono) !important;
}
.hero-stat-label {
    font-size: .73rem; color: rgba(255,255,255,.52) !important;
    font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
}

/* ── Metric cards ─────────────────────────────────────────────── */
.metric-card {
    background: var(--c-surface); border-radius: var(--r-lg);
    padding: 18px 20px; border: 1px solid var(--c-border);
    border-left: 4px solid var(--c-primary); box-shadow: var(--s-sm);
    margin-bottom: 12px; transition: box-shadow .18s, transform .18s;
}
.metric-card:hover { box-shadow: var(--s-md); transform: translateY(-1px); }
.metric-label {
    font-size: .67rem; color: var(--c-muted); font-weight: 700;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem; font-weight: 700; color: var(--c-dark);
    letter-spacing: -.02em; font-family: var(--mono) !important;
}
.metric-sub { font-size: .79rem; color: var(--c-primary); font-weight: 500; margin-top: 2px; }

/* ── Feature cards ────────────────────────────────────────────── */
.feature-card {
    background: var(--c-surface); border-radius: var(--r-lg); padding: 20px;
    border: 1px solid var(--c-border); box-shadow: var(--s-sm); height: 100%;
}
.feature-card-icon { font-size: 1.6rem; margin-bottom: 10px; display: block; }
.feature-card-title { font-size: .92rem; font-weight: 700; color: var(--c-dark) !important; margin-bottom: 6px; }
.feature-card-body  { font-size: .84rem; color: var(--c-muted) !important; line-height: 1.6; }

/* ── Step boxes ───────────────────────────────────────────────── */
.step-box {
    background: var(--c-surface); border-radius: var(--r-lg); padding: 14px 18px;
    border: 1px solid var(--c-border); box-shadow: var(--s-sm);
    display: flex; align-items: flex-start; gap: 14px; margin-bottom: 10px;
}
.step-num {
    min-width: 32px; height: 32px; border-radius: 50%; background: var(--c-primary);
    color: #fff !important; font-weight: 700; font-size: .9rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-title { font-size: .9rem; font-weight: 700; color: var(--c-dark) !important; margin-bottom: 4px; }
.step-desc  { font-size: .83rem; color: var(--c-muted) !important; line-height: 1.55; }

/* ── Pseudocode ───────────────────────────────────────────────── */
.pseudo-code {
    background: #0f172a; border-radius: var(--r-lg); padding: 22px 24px;
    font-family: var(--mono) !important; font-size: .86rem; line-height: 1.9;
    color: #e2e8f0 !important; margin: 14px 0; border: 1px solid #1e293b; overflow-x: auto;
}
.pc-kw  { color: #93c5fd !important; font-weight: 600; }
.pc-fn  { color: #6ee7b7 !important; }
.pc-cm  { color: #475569 !important; font-style: italic; }
.pc-nm  { color: #fcd34d !important; }
.pc-op  { color: #f87171 !important; }

/* ── Info / highlight boxes ───────────────────────────────────── */
.info-box {
    background: #eff6ff; border-left: 3px solid #3b82f6;
    padding: 13px 15px; border-radius: 0 var(--r-md) var(--r-md) 0;
    margin: 10px 0; font-size: .88rem; color: #1e3a8a !important; line-height: 1.65;
}
.info-box b, .info-box i { color: #1e3a8a !important; }
.highlight-box {
    background: #fefce8; border-left: 3px solid #fbbf24;
    padding: 13px 15px; border-radius: 0 var(--r-md) var(--r-md) 0;
    margin: 10px 0; font-size: .88rem; color: #78350f !important; line-height: 1.65;
}
.highlight-box b { color: #78350f !important; }
.success-box {
    background: #f0fdf4; border-left: 3px solid #22c55e;
    padding: 13px 15px; border-radius: 0 var(--r-md) var(--r-md) 0;
    margin: 10px 0; font-size: .88rem; color: #14532d !important; line-height: 1.65;
}
.success-box b { color: #14532d !important; }

/* ── Badges ───────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 4px;
    font-size: .71rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
}
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-purple { background: #ede9fe; color: #5b21b6; }
.badge-green  { background: #dcfce7; color: #15803d; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e2e8f0; padding: 5px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px; font-size: .9rem; font-weight: 600; border-radius: 8px;
    color: #475569; background: transparent; padding: 0 18px;
    font-family: var(--font) !important; transition: all .15s ease;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(15,23,42,.10);
}

/* ── Widgets — renforcement lisibilité ────────────────────────── */

/* Font globale sur les labels de widgets */
[data-testid="stSlider"] label,
[data-testid="stSelectSlider"] label,
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label,
[data-testid="stWidgetLabel"] p {
    font-family: var(--font) !important;
    font-weight: 600;
}

/* Tooltip d'aide */
[data-testid="stTooltipContent"] p { color: #334155 !important; }

/* ── Footer ───────────────────────────────────────────────────── */
.footer {
    text-align: center; padding: 22px 0 6px;
    border-top: 1px solid var(--c-border); margin-top: 32px;
    font-size: .8rem; color: var(--c-muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSES FROM SCRATCH
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
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.trees             = []
        self.init_value        = None
        self.train_mse_history = []
        self.test_mse_history  = []

    def _make_tree(self):
        if self.max_depth == 1:
            return DecisionStump()
        return DecisionTreeRegressor(max_depth=self.max_depth)

    def fit(self, X_tr, y_tr, X_te=None, y_te=None):
        self.init_value = np.mean(y_tr)
        pred_tr = np.full(len(y_tr), self.init_value)
        pred_te = np.full(len(y_te), self.init_value) if X_te is not None else None
        for _ in range(self.n_estimators):
            residus = y_tr - pred_tr
            t = self._make_tree()
            t.fit(X_tr, residus)
            self.trees.append(t)
            pred_tr = pred_tr + self.learning_rate * t.predict(X_tr)
            self.train_mse_history.append(np.mean((pred_tr - y_tr)**2))
            if X_te is not None:
                pred_te = pred_te + self.learning_rate * t.predict(X_te)
                self.test_mse_history.append(np.mean((pred_te - y_te)**2))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_value)
        for t in self.trees:
            pred += self.learning_rate * t.predict(X)
        return pred


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION DES DONNÉES
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
def train_gb_scratch(n_estimators, learning_rate, max_depth):
    X_train, X_test, y_train, y_test, _ = load_data()
    gb = GradientBoostingScratch(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    gb.fit(X_train, y_train, X_test, y_test)
    return gb.train_mse_history, gb.test_mse_history, gb.predict(X_test), y_test


@st.cache_data(show_spinner="Calcul des benchmarks sklearn…")
def compute_benchmarks():
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    lr = LinearRegression().fit(X_train, y_train)
    mse_lr = mean_squared_error(y_test, lr.predict(X_test))

    stump = DecisionStump()
    stump.fit(X_train, y_train)
    mse_stump = mean_squared_error(y_test, stump.predict(X_test))

    gb_iso = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
    gb_iso.fit(X_train, y_train)
    mse_gb_iso = mean_squared_error(y_test, gb_iso.predict(X_test))

    gb_opt = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    gb_opt.fit(X_train, y_train)
    mse_gb_opt = mean_squared_error(y_test, gb_opt.predict(X_test))
    fi_gb = gb_opt.feature_importances_

    gb_scratch = GradientBoostingScratch(n_estimators=100, learning_rate=0.1)
    gb_scratch.fit(X_train, y_train)
    pred_gb_scratch = gb_scratch.predict(X_test)
    mse_gb_scratch = mean_squared_error(y_test, pred_gb_scratch)

    return {
        "mse_lr":         mse_lr,
        "mse_stump":      mse_stump,
        "mse_gb_iso":     mse_gb_iso,
        "mse_gb_opt":     mse_gb_opt,
        "mse_gb_scratch": mse_gb_scratch,
        "pred_gb_scratch": pred_gb_scratch,
        "fi_gb":          fi_gb,
        "feature_cols":   feature_cols,
        "gb_opt":         gb_opt,
        "X_test":         X_test,
        "y_test":         y_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(col, label, value, sub="", accent=C_BLUE):
    style = f"border-left-color:{accent};"
    col.markdown(
        f'<div class="metric-card" style="{style}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-badge">Mémoire M1 Data Science · 2024–2025</div>
  <div class="hero-title">🌲 Gradient Boosting</div>
  <div class="hero-subtitle">
    Implémentation from-scratch &amp; comparaison avec scikit-learn<br>
    Dataset Ames Housing · log(SalePrice) · 80 % train / 20 % test
  </div>
  <div class="hero-stats">
    <div class="hero-stat">
      <span class="hero-stat-num">1 460</span>
      <span class="hero-stat-label">maisons analysées</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-num">4</span>
      <span class="hero-stat-label">modèles comparés</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-num">9</span>
      <span class="hero-stat-label">features sélectionnées</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-num">200</span>
      <span class="hero-stat-label">arbres — GB optimal</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🧠  Algorithme",
    "📈  Convergence interactive",
    "📊  Comparaison des modèles",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLET 1 — ALGORITHME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Principe — descente de gradient fonctionnel")
        st.markdown("""
        Le **Gradient Boosting** construit des modèles faibles de façon **séquentielle** :
        chaque arbre apprend à corriger le **résidu** (erreur) laissé par le modèle précédent,
        en suivant le gradient d'une fonction de perte (MSE ici).
        """)

        for num, title, desc in [
            (0, "Initialisation", "F₀(x) = ȳ — prédiction constante égale à la moyenne de y."),
            (1, "Calcul des résidus", "rᵢ = yᵢ − Fₜ₋₁(xᵢ) — gradient négatif de la MSE par rapport à F."),
            (2, "Apprenant faible", "hₜ = DecisionStump.fit(X, r) — apprend à corriger les résidus."),
            (3, "Mise à jour additive", "Fₜ(x) = Fₜ₋₁(x) + η · hₜ(x) — η freine la contribution de chaque arbre."),
        ]:
            st.markdown(f"""
            <div class="step-box">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Illustration — Arbre de décision (apprenant faible)")
        st.markdown(
            "À chaque itération, le GB entraîne un **arbre peu profond** sur les résidus courants. "
            "Voici un exemple de Decision Stump (depth=1) ajusté sur des résidus 1D :"
        )

        np.random.seed(42)
        _X_demo = np.sort(np.random.uniform(0, 10, 80)).reshape(-1, 1)
        _y_demo = np.sin(_X_demo.ravel()) + 0.35 * np.random.randn(80)
        _init   = np.mean(_y_demo)
        _resid  = _y_demo - _init

        _stump = DecisionTreeRegressor(max_depth=1, random_state=42)
        _stump.fit(_X_demo, _resid)

        fig_tree, axes_tree = plt.subplots(
            1, 2, figsize=(8, 4.2), facecolor="white",
            gridspec_kw={"width_ratios": [1.6, 1]}
        )

        sk_plot_tree(
            _stump,
            feature_names=["x"],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=axes_tree[0],
            impurity=False,
            precision=3,
        )
        axes_tree[0].set_title("Decision Stump sur les résidus", fontweight="bold", fontsize=11)
        axes_tree[0].set_facecolor("white")

        _x_line  = np.linspace(0, 10, 300).reshape(-1, 1)
        _thr     = _stump.tree_.threshold[0]
        _lv      = _stump.tree_.value[1][0][0]
        _rv      = _stump.tree_.value[2][0][0]

        axes_tree[1].scatter(_X_demo.ravel(), _resid, s=18, alpha=0.55,
                             color=C_GRAY, edgecolors="none", label="Résidus")
        axes_tree[1].axvline(_thr, color=C_BLUE, lw=1.8, linestyle="--",
                             label=f"Seuil x ≤ {_thr:.2f}")
        axes_tree[1].hlines(_lv, 0, _thr, colors=C_RED, lw=2.5,
                            label=f"Préd. gauche = {_lv:.3f}")
        axes_tree[1].hlines(_rv, _thr, 10, colors=C_GREEN, lw=2.5,
                            label=f"Préd. droite = {_rv:.3f}")
        axes_tree[1].set_xlabel("x", fontsize=10)
        axes_tree[1].set_ylabel("Résidu", fontsize=10)
        axes_tree[1].set_title("Régions prédites", fontweight="bold", fontsize=11)
        axes_tree[1].legend(fontsize=7.5, loc="upper right")
        axes_tree[1].set_facecolor("#fafafa")

        plt.tight_layout()
        st.pyplot(fig_tree)
        plt.close()

        st.markdown("""
        <div class="info-box">
        <b>Comment lire :</b> le stump choisit <i>un seul seuil</i> sur <i>x</i>.
        Les points à gauche reçoivent la valeur moyenne des résidus gauche,
        ceux à droite la valeur droite. Ce correctif est ajouté au modèle courant
        pondéré par η, puis l'opération se répète.
        </div>""", unsafe_allow_html=True)


    st.divider()

    # Concepts clés
    st.markdown("#### Concepts clés")
    k1, k2, k3 = st.columns(3)
    for col, icon, title, body in [
        (k1, "⚡", "Apprenant faible (Weak Learner)",
         "Un Decision Stump (depth = 1) — une seule coupure binaire. Individuellement très limité, mais puissant en ensemble séquentiel."),
        (k2, "🎯", "Learning Rate η",
         "Pondère chaque arbre. η faible → corrections prudentes, plus d'arbres nécessaires, meilleure généralisation. η fort → risque de surapprentissage."),
        (k3, "📉", "Résidus = Gradient négatif",
         "Pour la MSE, rᵢ = yᵢ − F(xᵢ) est exactement le gradient négatif. C'est ce qui fait du GB une descente de gradient dans l'espace fonctionnel."),
    ]:
        col.markdown(f"""
        <div class="feature-card">
          <span class="feature-card-icon">{icon}</span>
          <div class="feature-card-title">{title}</div>
          <div class="feature-card-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Visualisation sur données synthétiques
    st.markdown("#### Visualisation sur données synthétiques — sin(x) + bruit")
    st.markdown(
        "Observation du Gradient Boosting en action sur une fonction 1D. "
        "Chaque panneau montre la prédiction après T arbres."
    )

    np.random.seed(42)
    X_toy  = np.sort(np.random.uniform(0, 10, 100)).reshape(-1, 1)
    y_toy  = np.sin(X_toy.ravel()) + 0.35 * np.random.randn(100)
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X_plot.ravel())

    fig_toy, axes_toy = plt.subplots(1, 4, figsize=(16, 4.2), facecolor="white")
    for ax, n_trees in zip(axes_toy, [1, 5, 20, 100]):
        gb_toy = GradientBoostingRegressor(
            n_estimators=n_trees, learning_rate=0.3, max_depth=2, random_state=42)
        gb_toy.fit(X_toy, y_toy)
        mse_t = mean_squared_error(y_toy, gb_toy.predict(X_toy))

        ax.scatter(X_toy.ravel(), y_toy, s=15, alpha=0.45, color=C_GRAY,
                   edgecolors="none", label="Données")
        ax.plot(X_plot.ravel(), y_true, color=C_GREEN, lw=1.5,
                linestyle="--", alpha=0.8, label="Signal réel")
        ax.plot(X_plot.ravel(), gb_toy.predict(X_plot), color=C_BLUE,
                lw=2.2, label=f"GB (T={n_trees})")
        ax.set_title(f"T = {n_trees} arbre{'s' if n_trees > 1 else ''}\nMSE train = {mse_t:.4f}",
                     fontsize=11)
        ax.set_xlabel("x", fontsize=10)
        if n_trees == 1:
            ax.set_ylabel("y", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_facecolor("#fafafa")

    plt.suptitle("Convergence du Gradient Boosting — données synthétiques",
                 fontsize=12, y=1.03, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_toy)
    plt.close()

    st.markdown("""
    <div class="info-box">
    <b>Lecture :</b> T=1 → biais élevé, la forme globale n'est pas capturée.
    T=5 → la tendance générale emerge. T=20 → la courbe sinusoïde est bien approchée.
    T=100 → le modèle converge vers la vraie fonction sans sur-fitter grâce à η=0.3.
    </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLET 2 — CONVERGENCE INTERACTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.subheader("Convergence du Gradient Boosting — Ames Housing")
    st.markdown(
        "Explorez l'effet du nombre d'arbres et du learning rate sur la convergence "
        "— et l'apparition du surapprentissage."
    )

    col_ctrl, col_info = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### ⚙️ Paramètres")
        n_estimators = st.slider(
            "Nombre d'arbres", min_value=5, max_value=300, value=100, step=5,
            help="Chaque arbre corrige les résidus du précédent.")
        learning_rate = st.select_slider(
            "Learning rate (η)",
            options=[0.01, 0.05, 0.10, 0.20, 0.50], value=0.10,
            help="Poids donné à chaque arbre.")
        max_depth = st.radio(
            "Profondeur des arbres",
            options=[1, 2, 3], index=0, horizontal=True,
            help="depth=1 = stump · depth=2/3 = arbre sklearn")

        depth_labels = {
            1: "stump — 1 coupure",
            2: "2 niveaux — 4 feuilles",
            3: "3 niveaux — 8 feuilles",
        }
        status = (
            "⚠️ <b>Sous-apprentissage</b> probable — augmentez le nombre d'arbres."
            if n_estimators < 30 else
            "⚠️ <b>Surapprentissage</b> possible — observez l'écart train/test."
            if n_estimators >= 150 else
            "Convergence en cours."
        )
        depth_note = (
            " Avec depth&gt;1, le modèle capte plus d'interactions mais risque de sur-fitter plus vite."
            if max_depth > 1 else ""
        )
        st.markdown(
            f'<div class="info-box"><b>Configuration :</b> {n_estimators} arbres · '
            f'η={learning_rate} · {depth_labels[max_depth]}<br>{status}{depth_note}</div>',
            unsafe_allow_html=True)

    with col_info:
        st.markdown("#### 💡 Principe")
        st.markdown("""
        Le **Gradient Boosting** construit des arbres *séquentiellement* :
        chaque arbre apprend à corriger les **résidus** du modèle précédent.

        - **Learning rate faible** → corrections prudentes, plus d'arbres nécessaires
        - **Trop d'arbres** → mémorisation du bruit → surapprentissage (MSE test remonte)
        - **Point optimal** = arbre où le MSE test est minimal (early stopping)
        """)

    with st.spinner("Entraînement en cours…"):
        train_hist, test_hist, y_pred_scratch, y_test_ref = train_gb_scratch(
            n_estimators, learning_rate, max_depth)

    best_iter = int(np.argmin(test_hist)) + 1
    best_mse  = float(min(test_hist))
    final_mse = float(test_hist[-1])
    gap       = final_mse - best_mse

    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "MSE test final",   f"{final_mse:.5f}", f"après {n_estimators} arbres")
    metric_card(m2, "MSE test optimal", f"{best_mse:.5f}",  f"arbre n°{best_iter}", C_GREEN)
    metric_card(m3, "Surapprentissage", f"+{gap:.5f}",
                "0 = pas de sur-fit" if gap < 0.002 else "⚠ écart train/test croissant",
                C_RED if gap >= 0.002 else C_GREEN)
    metric_card(m4, "RMSE test", f"{np.sqrt(final_mse):.4f}", "en log(SalePrice)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    iters = range(1, n_estimators + 1)

    ax = axes[0]
    ax.plot(iters, train_hist, color=C_BLUE, lw=2, label="MSE Train")
    ax.plot(iters, test_hist,  color=C_RED,  lw=2, label="MSE Test")
    ax.axvline(best_iter, color=C_GREEN, lw=2, linestyle=":", label=f"Optimal (n°{best_iter})")
    ax.axhline(0.0238, color=C_GRAY, lw=1.2, linestyle="--", alpha=0.7,
               label="Régression linéaire (réf.)")
    if best_iter < n_estimators:
        ax.axvspan(best_iter, n_estimators, alpha=0.07, color=C_RED, label="Zone surapprentissage")
    ax.set_title("MSE en fonction du nombre d'arbres")
    ax.set_xlabel("Nombre d'arbres")
    ax.set_ylabel("MSE (log-SalePrice²)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_facecolor("#fafafa")

    ax2 = axes[1]
    ax2.scatter(y_test_ref, y_pred_scratch, alpha=0.42, s=16,
                color=C_BLUE, edgecolors="none", label="Prédictions GB scratch")
    lims = [min(y_test_ref.min(), y_pred_scratch.min()) - 0.05,
            max(y_test_ref.max(), y_pred_scratch.max()) + 0.05]
    ax2.plot(lims, lims, color=C_RED, lw=2, linestyle="--", label="Prédiction parfaite")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_title(f"Prédictions vs Valeurs réelles ({n_estimators} arbres)")
    ax2.set_xlabel("log(SalePrice) réel")
    ax2.set_ylabel("log(SalePrice) prédit")
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#fafafa")

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Distribution des résidus (jeu de test)")
    residus = y_pred_scratch - y_test_ref
    fig2, ax3 = plt.subplots(figsize=(10, 3.5), facecolor="white")
    ax3.hist(residus, bins=50, color=C_BLUE, edgecolor="white", alpha=0.82)
    ax3.axvline(0, color=C_RED, lw=2.5, linestyle="--", label="Zéro (erreur nulle)")
    ax3.axvline(np.mean(residus), color=C_GREEN, lw=2, linestyle="-.",
                label=f"Moyenne = {np.mean(residus):.4f}")
    ax3.set_title("Distribution des résidus — Gradient Boosting from scratch")
    ax3.set_xlabel("Résidu (ŷ − y)")
    ax3.set_ylabel("Fréquence")
    ax3.legend(fontsize=9)
    ax3.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown(
        f'<div class="highlight-box">'
        f'<b>📌 À retenir :</b> {n_estimators} arbres · η={learning_rate} · depth={max_depth} — '
        f'MSE optimal <b>{best_mse:.5f}</b> (arbre n°{best_iter}). '
        f'{"Pas de surapprentissage détecté." if gap < 0.001 else f"Surapprentissage de +{gap:.5f} après le point optimal."} '
        f'La régression linéaire de référence atteint MSE ≈ 0.024.</div>',
        unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLET 3 — COMPARAISON DES MODÈLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.subheader("Comparaison des modèles sur le jeu de test")
    st.markdown(
        "Performances des quatre modèles étudiés dans le mémoire, "
        "entraînés sur le jeu Ames Housing (log-SalePrice, 80 % train / 20 % test)."
    )

    bench = compute_benchmarks()

    models_data = [
        ("Decision Stump (depth=1)",
         bench["mse_stump"], np.sqrt(bench["mse_stump"]),
         '<span class="badge badge-purple">From scratch</span>'),
        ("Régression Linéaire (9 features)",
         bench["mse_lr"], np.sqrt(bench["mse_lr"]),
         '<span class="badge badge-blue">sklearn</span>'),
        ("GB from scratch (100 stumps, lr=0.1)",
         bench["mse_gb_scratch"], np.sqrt(bench["mse_gb_scratch"]),
         '<span class="badge badge-purple">From scratch</span>'),
        ("Gradient Boosting iso-param.\n(depth=1, 100 arbres, lr=0.1)",
         bench["mse_gb_iso"], np.sqrt(bench["mse_gb_iso"]),
         '<span class="badge badge-blue">sklearn</span>'),
        ("Gradient Boosting optimal\n(depth=3, 200 arbres, lr=0.05)",
         bench["mse_gb_opt"], np.sqrt(bench["mse_gb_opt"]),
         '<span class="badge badge-green">sklearn ★</span>'),
    ]

    st.markdown("#### Tableau des performances")
    df_results = pd.DataFrame(
        [(m[0], f"{m[1]:.6f}", f"{m[2]:.5f}") for m in models_data],
        columns=["Modèle", "MSE test", "RMSE test"]
    )
    df_results = df_results.sort_values("MSE test")
    st.dataframe(
        df_results.reset_index(drop=True),
        use_container_width=True, hide_index=True,
        column_config={
            "Modèle":    st.column_config.TextColumn("Modèle", width="large"),
            "MSE test":  st.column_config.TextColumn("MSE test ↓"),
            "RMSE test": st.column_config.TextColumn("RMSE test ↓"),
        }
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### Comparaison MSE (jeu de test)")
        labels = [
            "Decision Stump",
            "Régression\nLinéaire",
            "GB scratch\n(100 stumps)",
            "GB sklearn\n(depth=1)",
            "GB optimal\n(depth=3)",
        ]
        values = [
            bench["mse_stump"], bench["mse_lr"], bench["mse_gb_scratch"],
            bench["mse_gb_iso"], bench["mse_gb_opt"],
        ]
        colors = [C_RED, C_GRAY, C_PURPLE, C_BLUE, C_GREEN]

        fig3, ax4 = plt.subplots(figsize=(7, 5), facecolor="white")
        bars = ax4.barh(labels, values, color=colors, edgecolor="white", height=0.55)
        for bar, val in zip(bars, values):
            bar.set_alpha(0.88)
            ax4.text(val + 0.0004, bar.get_y() + bar.get_height()/2,
                     f"{val:.5f}", va="center", fontsize=9.5, fontweight="bold", color=C_DARK)
        ax4.axvline(bench["mse_gb_opt"], color=C_GREEN, lw=1.5, linestyle="--", alpha=0.6)
        ax4.set_xlabel("MSE (log-SalePrice²) — plus bas = meilleur")
        ax4.set_title("MSE sur le jeu de test")
        ax4.set_facecolor("#fafafa")
        ax4.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_g2:
        st.markdown("#### Feature Importance — GB Optimal (depth=3)")
        fcols  = bench["feature_cols"]
        fi_arr = bench["fi_gb"]
        fi_idx = np.argsort(fi_arr)[::-1][:12]
        fi_vals  = fi_arr[fi_idx]
        fi_names = [fcols[i] if len(fcols[i]) <= 18 else fcols[i][:16]+"…" for i in fi_idx]

        fig4, ax5 = plt.subplots(figsize=(7, 5), facecolor="white")
        bar_colors = [C_DARK if v == fi_vals.max() else C_BLUE for v in fi_vals]
        bars2 = ax5.barh(fi_names[::-1], fi_vals[::-1],
                         color=bar_colors[::-1], edgecolor="white", height=0.6)
        for bar in bars2:
            bar.set_alpha(0.85)
        ax5.set_xlabel("Importance (gain)")
        ax5.set_title("Top 12 features — Gradient Boosting\n(200 arbres, depth=3, lr=0.05)")
        ax5.set_facecolor("#fafafa")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("#### Distribution des résidus par modèle (violin plot)")
    X_tr, X_te, y_tr, y_te, _ = load_data()
    stump_m = DecisionStump(); stump_m.fit(X_tr, y_tr)
    lr_m    = LinearRegression().fit(X_tr, y_tr)
    gb_m    = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42).fit(X_tr, y_tr)

    preds = {
        "Decision Stump":    stump_m.predict(X_te),
        "Régr. Linéaire":    lr_m.predict(X_te),
        "GB from scratch":   bench["pred_gb_scratch"],
        "GB optimal sklearn": gb_m.predict(X_te),
    }
    violin_colors = [C_RED, C_GRAY, C_PURPLE, C_GREEN]

    fig_violin = go.Figure()
    for (name, ypred), color in zip(preds.items(), violin_colors):
        fig_violin.add_trace(go.Violin(
            y=(ypred - y_te).tolist(), name=name,
            box_visible=True, meanline_visible=True,
            line_color=color, fillcolor=color, opacity=0.52,
            points="outliers", marker=dict(size=3, opacity=0.4),
        ))
    fig_violin.add_hline(y=0, line_dash="dash", line_color="#1d3557", line_width=1.5,
                         annotation_text="Résidu = 0", annotation_position="top right",
                         annotation_font_color="#1d3557")
    fig_violin.update_layout(
        title=dict(
            text="Distribution des résidus (ŷ − y) — plus la boîte est étroite, meilleur est le modèle",
            font=dict(size=13, color="#1d3557")),
        yaxis=dict(title="Résidu (ŷ − y)", gridcolor="#e5e7eb",
                   tickfont=dict(color="#1d3557"), titlefont=dict(color="#1d3557")),
        xaxis=dict(tickfont=dict(color="#1d3557")),
        height=420, paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(color="#1d3557"), showlegend=False,
        violingap=0.25, margin=dict(t=55, b=30, l=60, r=20),
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("#### Prédictions vs Valeurs réelles — comparaison 2×2")
    model_list = list(preds.items())
    fig_sc = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"<b>{name}</b>  —  MSE = {mean_squared_error(y_te, yp):.5f}"
            for name, yp in model_list
        ],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    positions   = [(1, 1), (1, 2), (2, 1), (2, 2)]
    global_cmax = max(float(np.abs(yp - y_te).max()) for _, yp in model_list)

    for (name, ypred), (row, col) in zip(model_list, positions):
        abs_res = np.abs(ypred - y_te)
        lims = [min(float(y_te.min()), float(ypred.min())) - 0.05,
                max(float(y_te.max()), float(ypred.max())) + 0.05]
        show_scale = (row == 2 and col == 2)

        fig_sc.add_trace(go.Scatter(
            x=y_te.tolist(), y=ypred.tolist(), mode="markers",
            marker=dict(size=5, color=abs_res.tolist(),
                        colorscale="RdYlGn_r", cmin=0, cmax=global_cmax,
                        showscale=show_scale,
                        colorbar=dict(title="|Résidu|", thickness=12, len=0.45,
                                      y=0.25, tickfont=dict(color="#1d3557"),
                                      titlefont=dict(color="#1d3557")) if show_scale else {},
                        opacity=0.75),
            showlegend=False,
            hovertemplate="Réel: %{x:.3f}<br>Prédit: %{y:.3f}<br>|Résidu|: %{marker.color:.3f}<extra></extra>",
        ), row=row, col=col)
        fig_sc.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#1d3557", dash="dash", width=1.5), showlegend=False,
        ), row=row, col=col)
        fig_sc.update_xaxes(range=lims, title_text="log(SalePrice) réel",
                            gridcolor="#e5e7eb", tickfont=dict(color="#1d3557"),
                            titlefont=dict(color="#1d3557"), row=row, col=col)
        fig_sc.update_yaxes(range=lims, title_text="log(SalePrice) prédit",
                            gridcolor="#e5e7eb", tickfont=dict(color="#1d3557"),
                            titlefont=dict(color="#1d3557"), row=row, col=col)

    fig_sc.update_layout(
        height=660, paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(color="#1d3557", size=11),
        margin=dict(t=60, b=40, l=60, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Interprétation")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown(
            '<div class="info-box"><b>Pourquoi le GB surpasse la régression linéaire ?</b><br>'
            'La régression suppose une relation <i>linéaire additive</i> entre features et cible. '
            'Le GB, via des arbres, capture les <b>interactions non-linéaires</b> '
            '(ex. OverallQual × GrLivArea) — impossibles à modéliser linéairement.</div>',
            unsafe_allow_html=True)
    with ci2:
        st.markdown(
            f'<div class="info-box"><b>Rôle de la profondeur (depth) :</b><br>'
            f'Un stump (depth=1) ne capte qu\'<i>une seule interaction à la fois</i>. '
            f'Avec depth=3, chaque arbre modélise jusqu\'à 8 régions — des '
            f'<b>interactions multi-features</b> et un biais structurel réduit. '
            f'MSE : {bench["mse_gb_iso"]:.5f} (depth=1) → {bench["mse_gb_opt"]:.5f} (depth=3).</div>',
            unsafe_allow_html=True)

    st.markdown(
        f'<div class="highlight-box">'
        f'<b>🏆 Résultat clé du mémoire :</b> Le GB sklearn optimal (depth=3, 200 arbres, lr=0.05) '
        f'atteint MSE = <b>{bench["mse_gb_opt"]:.5f}</b>, soit une réduction de '
        f'<b>{(1 - bench["mse_gb_opt"]/bench["mse_lr"])*100:.1f} %</b> par rapport à la régression '
        f'linéaire (MSE = {bench["mse_lr"]:.5f}). '
        f'La feature la plus importante est <b>OverallQual</b> — qualité générale du bien.</div>',
        unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Mémoire M1 Data Science · Gradient Boosting · Dataset : Ames Housing (Kaggle) · '
    'Références : Friedman (2001) · Chen &amp; Guestrin (2016)'
    '</div>',
    unsafe_allow_html=True)
