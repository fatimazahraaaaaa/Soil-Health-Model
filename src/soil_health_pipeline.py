"""
soil_health_pipeline.py
=======================
Explainable Long-Term Soil Health Forecasting Framework
Stages 2-8: EDA → Imputation → SHI → Features → Modelling → SHAP → Scenarios

Author : ZaHra
Date   : 2026-06-18
"""

from __future__ import annotations
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────────────────────────────────────
CHEMISTRY_COLS  = ['pH_H2O', 'pH_CaCl2', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K', 'Ox_Al', 'Ox_Fe']
PHYSICAL_COLS   = ['BD_0_20', 'Clay', 'Sand', 'Silt', 'Coarse']
SPATIAL_COLS    = ['TH_LAT', 'TH_LONG', 'Elev']
CATEGORICAL_COLS= ['NUTS_0', 'LC', 'LU', 'Depth']
LOG_COLS        = ['EC', 'OC', 'P', 'K', 'CaCO3']   # right-skewed

# SHI weights (must sum to 1)
SHI_WEIGHTS = {
    'pH_H2O': 0.25,
    'OC':     0.25,
    'N':      0.15,
    'EC':     0.15,
    'P':      0.10,
    'K':      0.10,
}

SHI_RISK_LABELS = {
    0: 'Critical  (<0.25)',
    1: 'High      (0.25-0.50)',
    2: 'Moderate  (0.50-0.75)',
    3: 'Low       (>0.75)',
}

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Run full EDA suite and display/save all plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log.info('=== Stage 2: EDA ===')
    plot_missing_values(df, save_dir)
    plot_chemistry_distributions(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_spatial_distribution(df, save_dir)
    plot_land_cover_stats(df, save_dir)
    log.info('EDA complete. Figures saved to %s', save_dir)


def plot_missing_values(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Horizontal bar chart: % missing per column (only columns with >0 missing)."""
    miss = (df.isna().mean() * 100).sort_values(ascending=True)
    miss = miss[miss > 0]
    if miss.empty:
        log.info('No missing values.')
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.28)))
    colors = ['#d62728' if v > 50 else '#ff7f0e' if v > 20 else '#1f77b4' for v in miss]
    miss.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Missing (%)', fontsize=12)
    ax.set_title('Missing Value Profile — LUCAS 2018 Master Dataset', fontsize=14, pad=12)
    for patch, val in zip(ax.patches, miss):
        ax.text(val + 0.5, patch.get_y() + patch.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_missing_values.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_chemistry_distributions(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Grid of histograms + KDE for all chemistry variables."""
    cols = [c for c in CHEMISTRY_COLS if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)
        try:
            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 300)
            axes[i].plot(x, kde(x), 'r-', lw=1.5)
        except Exception:
            pass
        axes[i].set_title(f'{col}  (n={len(data):,}, skew={data.skew():.2f})', fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('LUCAS 2018 — Chemistry Variable Distributions', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_chemistry_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Pearson correlation heatmap for numeric soil variables."""
    num_cols = [c for c in CHEMISTRY_COLS + PHYSICAL_COLS if c in df.columns]
    corr = df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.4, annot_kws={'size': 8},
    )
    ax.set_title('Pearson Correlation — Soil Chemistry & Physical Variables', fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_spatial_distribution(df: pd.DataFrame, save_dir: str = '.', color_col: str = 'OC') -> None:
    """Scatter map of Europe coloured by a soil variable."""
    if 'TH_LAT' not in df.columns or 'TH_LONG' not in df.columns:
        log.warning('Spatial columns not found.')
        return
    if color_col not in df.columns:
        color_col = 'pH_H2O'

    data = df.dropna(subset=['TH_LAT', 'TH_LONG', color_col])
    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(
        data['TH_LONG'], data['TH_LAT'],
        c=data[color_col], cmap='RdYlGn', s=3, alpha=0.6,
        vmin=data[color_col].quantile(0.02),
        vmax=data[color_col].quantile(0.98),
    )
    plt.colorbar(sc, ax=ax, label=color_col, shrink=0.7)
    ax.set(xlabel='Longitude', ylabel='Latitude',
           title=f'LUCAS 2018 — Spatial Distribution of {color_col}')
    ax.set_xlim(-12, 35); ax.set_ylim(34, 72)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_spatial_{color_col}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_land_cover_stats(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Point count and median OC by land cover description."""
    col = 'LC0_Desc' if 'LC0_Desc' in df.columns else ('LC' if 'LC' in df.columns else None)
    if col is None:
        return

    top_lc = df[col].value_counts().head(10).index
    sub = df[df[col].isin(top_lc)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Point counts
    sub[col].value_counts().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Sample Count by Land Cover', fontsize=12)
    axes[0].set_xlabel('Count')

    # Median OC by land cover
    if 'OC' in df.columns:
        oc_by_lc = sub.groupby(col)['OC'].median().sort_values()
        oc_by_lc.plot(kind='barh', ax=axes[1], color='forestgreen')
        axes[1].set_title('Median OC (g/kg) by Land Cover', fontsize=12)
        axes[1].set_xlabel('Median OC (g/kg)')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_land_cover.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — IMPUTATION & SCALING
# ═════════════════════════════════════════════════════════════════════════════

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median imputation for numeric columns with missing values.
    Adds a binary <col>_MISSING indicator for columns imputed.
    Columns with >80% missing (Ox_Al, Ox_Fe, OC_20_30, CaCO3_20_30, BD_20_30)
    are dropped to avoid noise.
    """
    df = df.copy()

    # Drop very-high-missingness columns
    high_miss = [c for c in df.columns
                 if df[c].isna().mean() > 0.80 and df[c].dtype in ['float64', 'Float64']]
    if high_miss:
        log.info('Dropping high-missingness columns (>80%%): %s', high_miss)
        df = df.drop(columns=high_miss)

    # Median imputation for remaining numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[f'{col}_MISSING'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(median_val)

    log.info('Imputation complete. Shape: %s', df.shape)
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transform to right-skewed chemistry columns.
    Creates <col>_log columns and keeps originals.
    """
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            log.info('  log1p applied to %s', col)
    return df


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    StandardScaler on numeric features.
    Returns (X_scaled DataFrame, fitted scaler).
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled, scaler


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SUSTAINABLE HEALTH INDEX (SHI)
# ═════════════════════════════════════════════════════════════════════════════

def _score_ph(s: pd.Series) -> pd.Series:
    """
    Triangular scoring: optimal range 6.0–7.0 (score=1).
    Score=0 at pH≤3.5 or pH≥9.5.
    Based on FAO/EU soil quality guidelines for European agricultural soils.
    """
    score = pd.Series(np.nan, index=s.index)
    # Below optimum
    mask_low = s < 6.0
    score[mask_low] = ((s[mask_low] - 3.5) / (6.0 - 3.5)).clip(0, 1)
    # Optimum zone
    mask_opt = (s >= 6.0) & (s <= 7.0)
    score[mask_opt] = 1.0
    # Above optimum
    mask_high = s > 7.0
    score[mask_high] = ((9.5 - s[mask_high]) / (9.5 - 7.0)).clip(0, 1)
    return score


def _score_oc(s: pd.Series) -> pd.Series:
    """
    Saturating curve: score = min(OC/40, 1).
    40 g/kg is considered 'high' organic carbon for European topsoils.
    """
    return (s / 40.0).clip(0, 1)


def _score_n(s: pd.Series) -> pd.Series:
    """Linear 0–4 g/kg range; capped at 1."""
    return (s / 4.0).clip(0, 1)


def _score_p(s: pd.Series) -> pd.Series:
    """
    Triangular: low P = nutrient deficiency, high P = eutrophication risk.
    Optimal 20–60 mg/kg; score=0 at P=0 or P≥300.
    """
    score = pd.Series(np.nan, index=s.index)
    mask_low  = s < 20
    mask_opt  = (s >= 20) & (s <= 60)
    mask_high = s > 60
    score[mask_low]  = (s[mask_low] / 20.0).clip(0, 1)
    score[mask_opt]  = 1.0
    score[mask_high] = ((300 - s[mask_high]) / (300 - 60)).clip(0, 1)
    return score


def _score_k(s: pd.Series) -> pd.Series:
    """Diminishing returns: score = min(K/400, 1)."""
    return (s / 400.0).clip(0, 1)


def _score_ec(s: pd.Series) -> pd.Series:
    """
    Inverse scoring: low EC (salinity) = high score.
    EC > 400 mS/m is considered severely saline.
    """
    return (1.0 - s / 400.0).clip(0, 1)


_SCORE_FNS = {
    'pH_H2O': _score_ph,
    'OC':     _score_oc,
    'N':      _score_n,
    'P':      _score_p,
    'K':      _score_k,
    'EC':     _score_ec,
}


def compute_shi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Sustainable Health Index (SHI) for every point.

    SHI = Σ (weight_i × score_i)  where scores ∈ [0,1]

    Adds columns:
      score_pH, score_OC, score_N, score_P, score_K, score_EC
      SHI          — continuous [0,1]
      SHI_CLASS    — 0=Critical, 1=High risk, 2=Moderate, 3=Low risk
      SHI_LABEL    — human-readable class label

    Points with all 6 indicators missing receive SHI=NaN.
    Points with ≥1 available indicator use available-indicator-weighted mean.
    """
    df = df.copy()
    weighted_sum  = pd.Series(0.0, index=df.index)
    weight_total  = pd.Series(0.0, index=df.index)

    for col, fn in _SCORE_FNS.items():
        w = SHI_WEIGHTS[col]
        if col not in df.columns:
            log.warning('SHI: %s not found — skipping.', col)
            continue
        score = fn(df[col])
        df[f'score_{col}'] = score
        available = score.notna()
        weighted_sum  += score.fillna(0) * w * available
        weight_total  += w * available

    df['SHI'] = (weighted_sum / weight_total.replace(0, np.nan)).round(4)

    # Risk classification
    bins   = [-0.001, 0.25, 0.50, 0.75, 1.001]
    labels = [0, 1, 2, 3]
    df['SHI_CLASS'] = pd.cut(df['SHI'], bins=bins, labels=labels).astype('Int8')
    df['SHI_LABEL'] = df['SHI_CLASS'].map(SHI_RISK_LABELS)

    # Summary
    for cls, lbl in SHI_RISK_LABELS.items():
        n = (df['SHI_CLASS'] == cls).sum()
        log.info('  %s : %d points (%.1f%%)', lbl, n, n / len(df) * 100)
    log.info('  SHI range: %.3f – %.3f  |  mean: %.3f',
             df['SHI'].min(), df['SHI'].max(), df['SHI'].mean())

    return df


def plot_shi_distribution(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Plot SHI histogram + risk class breakdown."""
    if 'SHI' not in df.columns:
        log.warning('SHI not found. Run compute_shi() first.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SHI histogram
    df['SHI'].dropna().hist(bins=60, ax=axes[0], color='teal', edgecolor='white')
    axes[0].axvline(0.25, color='red',    ls='--', lw=1.5, label='Critical')
    axes[0].axvline(0.50, color='orange', ls='--', lw=1.5, label='High')
    axes[0].axvline(0.75, color='green',  ls='--', lw=1.5, label='Moderate')
    axes[0].set(xlabel='SHI', ylabel='Count',
                title='Sustainable Health Index Distribution')
    axes[0].legend(fontsize=9)

    # Risk class pie
    counts = df['SHI_CLASS'].value_counts().sort_index()
    labels = [SHI_RISK_LABELS.get(int(k), str(k)) for k in counts.index]
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c']
    axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 9})
    axes[1].set_title('SHI Risk Class Distribution', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/shi_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_shi_spatial(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Map of Europe coloured by SHI."""
    plot_spatial_distribution(df, save_dir, color_col='SHI')


def plot_shi_by_land_cover(df: pd.DataFrame, save_dir: str = '.') -> None:
    """Boxplot of SHI by land cover category."""
    if 'SHI' not in df.columns or 'LC0_Desc' not in df.columns:
        return
    top = df['LC0_Desc'].value_counts().head(8).index
    sub = df[df['LC0_Desc'].isin(top)].copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    order = sub.groupby('LC0_Desc')['SHI'].median().sort_values(ascending=False).index
    sns.boxplot(data=sub, x='LC0_Desc', y='SHI', order=order,
                palette='RdYlGn', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.axhline(0.50, color='red', ls='--', lw=1, label='Risk threshold')
    ax.set(title='SHI Distribution by Land Cover', ylabel='SHI', xlabel='')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shi_by_land_cover.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns. Fills NaN with 'Unknown' first."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


def build_feature_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the final feature matrix X, target y (SHI), and feature name list.

    Feature groups included:
    - Chemistry (raw):  pH_H2O, pH_CaCl2, OC, N, P, K, EC, CaCO3, BD_0_20
    - Log-transformed:  OC_log, P_log, K_log, EC_log, CaCO3_log
    - Physical:         Clay, Sand, Silt, Coarse
    - Spatial:          TH_LAT, TH_LONG, Elev
    - Categorical:      NUTS_0, LC, LU, Depth (label-encoded)
    - Temporal proxy:   TEXTURE_YEAR
    - Missing flags:    <col>_MISSING (wherever imputed)

    Target:
    - y = SHI (continuous)

    Drops rows where SHI is NaN.
    """
    if 'SHI' not in df.columns:
        raise ValueError('Run compute_shi(df) before build_feature_set().')

    df = df.copy()

    # Collect available feature columns
    chem    = [c for c in ['pH_H2O','pH_CaCl2','OC','N','P','K','EC','CaCO3','BD_0_20'] if c in df.columns]
    log_t   = [c for c in ['OC_log','P_log','K_log','EC_log','CaCO3_log'] if c in df.columns]
    phys    = [c for c in PHYSICAL_COLS if c in df.columns]
    spat    = [c for c in SPATIAL_COLS if c in df.columns]
    cat     = [c for c in CATEGORICAL_COLS if c in df.columns]
    temp    = [c for c in ['TEXTURE_YEAR'] if c in df.columns]
    miss    = [c for c in df.columns if c.endswith('_MISSING')]

    feature_cols = list(dict.fromkeys(chem + log_t + phys + spat + cat + temp + miss))
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Filter to rows with a valid SHI
    sub = df.dropna(subset=['SHI'])
    X = sub[feature_cols].copy()
    y = sub['SHI'].copy()

    # Convert Int types to float
    for col in X.columns:
        if str(X[col].dtype).startswith('Int') or str(X[col].dtype).startswith('bool'):
            X[col] = X[col].astype(float)

    log.info('Feature set: %d rows × %d features', len(X), len(feature_cols))
    log.info('Target (SHI): mean=%.3f  std=%.3f', y.mean(), y.std())
    return X, y, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 — MODELLING
# ═════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
    random_state: int = 42,
    run_hyperparam_search: bool = False,
) -> dict:
    """
    Train XGBoost and Random Forest on the SHI regression target.
    Returns a results dict with models, CV scores, test metrics, and predictions.

    Parameters
    ----------
    X : feature matrix (after imputation + scaling)
    y : SHI target
    cv : cross-validation folds
    run_hyperparam_search : if True, runs RandomizedSearchCV (slower)
    """
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    log.info('=== Stage 6: Model Training ===')

    # Train / test split  (80/20, stratified by NUTS_0 country if available)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state,
    )
    log.info('Train: %d  |  Test: %d', len(X_train), len(X_test))

    # ── Model definitions ──────────────────────────────────────────────────
    xgb_params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state, n_jobs=-1,
        eval_metric='rmse',
    )
    rf_params = dict(
        n_estimators=300, max_depth=12,
        min_samples_leaf=5, max_features=0.6,
        random_state=random_state, n_jobs=-1,
    )

    if run_hyperparam_search:
        log.info('Running hyperparameter search (this may take a few minutes)…')
        xgb_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9],
        }
        rf_grid = {
            'n_estimators': [200, 400],
            'max_depth': [8, 12, None],
            'min_samples_leaf': [3, 5, 10],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
            xgb_grid, n_iter=20, cv=3, scoring='r2',
            random_state=random_state, n_jobs=-1, verbose=0,
        )
        xgb_search.fit(X_train, y_train)
        xgb_model = xgb_search.best_estimator_
        log.info('XGBoost best params: %s', xgb_search.best_params_)

        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            rf_grid, n_iter=10, cv=3, scoring='r2',
            random_state=random_state, n_jobs=-1, verbose=0,
        )
        rf_search.fit(X_train, y_train)
        rf_model = rf_search.best_estimator_
        log.info('RF best params: %s', rf_search.best_params_)
    else:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)

    models = {'XGBoost': xgb_model, 'RandomForest': rf_model}

    # ── Evaluation ─────────────────────────────────────────────────────────
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        metrics = {
            'R2_test':  r2_score(y_test, y_pred),
            'MAE_test': mean_absolute_error(y_test, y_pred),
            'RMSE_test': mean_squared_error(y_test, y_pred) ** 0.5,
            'R2_cv_mean': cv_scores.mean(),
            'R2_cv_std':  cv_scores.std(),
        }
        results[name] = {
            'model':   model,
            'metrics': metrics,
            'y_pred':  y_pred,
            'y_test':  y_test,
            'X_test':  X_test,
            'X_train': X_train,
            'y_train': y_train,
        }
        log.info('%s | R²=%.4f  MAE=%.4f  RMSE=%.4f  CV-R²=%.4f±%.4f',
                 name, metrics['R2_test'], metrics['MAE_test'],
                 metrics['RMSE_test'], metrics['R2_cv_mean'], metrics['R2_cv_std'])

    return results


def plot_model_performance(results: dict, save_dir: str = '.') -> None:
    """Actual vs predicted + residual plots for all models."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (name, res) in enumerate(results.items()):
        y_test = res['y_test']
        y_pred = res['y_pred']
        m = res['metrics']

        # Actual vs Predicted
        axes[0, j].scatter(y_test, y_pred, alpha=0.25, s=6, color='steelblue')
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        axes[0, j].plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        axes[0, j].set(
            xlabel='Actual SHI', ylabel='Predicted SHI',
            title=f'{name}\nR²={m["R2_test"]:.4f}  RMSE={m["RMSE_test"]:.4f}',
        )

        # Residuals
        residuals = y_test - y_pred
        axes[1, j].scatter(y_pred, residuals, alpha=0.25, s=6, color='darkorange')
        axes[1, j].axhline(0, color='red', lw=1.5, ls='--')
        axes[1, j].set(
            xlabel='Predicted SHI', ylabel='Residual',
            title=f'{name} — Residual Plot',
        )

    plt.suptitle('Model Performance — SHI Prediction', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(results: dict, feature_names: list, save_dir: str = '.', top_n: int = 20) -> None:
    """Bar chart of built-in feature importances for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        model = res['model']
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            imp.nlargest(top_n).sort_values().plot(kind='barh', ax=ax, color='steelblue')
            ax.set(title=f'{name} — Feature Importance (top {top_n})', xlabel='Importance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7 — SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(
    results: dict,
    feature_names: list,
    save_dir: str = '.',
    n_background: int = 200,
) -> dict:
    """
    Compute SHAP values for all trained models.
    Produces: summary beeswarm, bar importance, top-5 dependence plots.
    Returns dict of {model_name: shap_values array}.
    """
    import shap
    log.info('=== Stage 7: SHAP Analysis ===')
    shap_results = {}

    for name, res in results.items():
        model   = res['model']
        X_train = res['X_train']
        X_test  = res['X_test']

        log.info('Computing SHAP for %s …', name)
        # Use TreeExplainer (fast for tree-based models)
        background = X_train.sample(min(n_background, len(X_train)), random_state=42)
        explainer  = shap.TreeExplainer(model, data=background)
        shap_vals  = explainer(X_test)

        shap_results[name] = shap_vals

        # ── Beeswarm summary ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_vals, max_display=20, show=False)
        plt.title(f'{name} — SHAP Beeswarm (Global Feature Impact)', pad=10)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_{name}_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ── Bar importance ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.bar(shap_vals, max_display=20, show=False)
        plt.title(f'{name} — SHAP Feature Importance', pad=10)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_{name}_bar.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ── Dependence plots for top-5 features ──────────────────────────
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        top5_idx = np.argsort(mean_abs)[::-1][:5]

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        for ax, idx in zip(axes, top5_idx):
            shap.plots.scatter(shap_vals[:, idx], ax=ax, show=False)
            ax.set_title(feature_names[idx], fontsize=10)
        fig.suptitle(f'{name} — SHAP Dependence (Top 5 Features)', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_{name}_dependence.png', dpi=150, bbox_inches='tight')
        plt.show()

    return shap_results


def generate_diagnostic_report(
    model,
    point: pd.Series,
    X_background: pd.DataFrame,
    feature_names: list,
    point_id: str = 'Point',
    save_dir: str = '.',
) -> dict:
    """
    Generate a per-point SHAP diagnostic report.

    Returns a dict with:
    - shi_predicted: model's SHI prediction for this point
    - top_positive_drivers: features pushing SHI UP (healthy drivers)
    - top_negative_drivers: features pushing SHI DOWN (degradation drivers)
    - text_report: human-readable diagnostic string
    - waterfall_path: path to saved waterfall plot
    """
    import shap

    X_point = point[feature_names].values.reshape(1, -1)
    X_bg_arr = X_background[feature_names].sample(
        min(100, len(X_background)), random_state=42
    )

    explainer = shap.TreeExplainer(model, data=X_bg_arr)
    shap_exp  = explainer(pd.DataFrame(X_point, columns=feature_names))

    shi_pred = float(model.predict(X_point)[0])
    shap_vals = shap_exp.values[0]

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_exp[0], max_display=15, show=False)
    plt.title(f'Diagnostic Report — {point_id}  |  Predicted SHI = {shi_pred:.3f}', fontsize=12)
    plt.tight_layout()
    wf_path = f'{save_dir}/diagnostic_{point_id}_waterfall.png'
    plt.savefig(wf_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Rank drivers
    driver_df = pd.DataFrame({
        'feature': feature_names,
        'shap':    shap_vals,
        'value':   point[feature_names].values,
    }).sort_values('shap')

    pos_drivers = driver_df[driver_df['shap'] > 0].tail(5)
    neg_drivers = driver_df[driver_df['shap'] < 0].head(5)

    # Risk class
    if shi_pred >= 0.75:   risk = 'Low risk'
    elif shi_pred >= 0.50: risk = 'Moderate risk'
    elif shi_pred >= 0.25: risk = 'High risk'
    else:                  risk = 'CRITICAL'

    # Text report
    lines = [
        f'═══ Soil Health Diagnostic Report — {point_id} ═══',
        f'Predicted SHI : {shi_pred:.3f}  ({risk})',
        '',
        '▼ Top DEGRADATION drivers (pulling SHI down):',
    ]
    for _, row in neg_drivers.iterrows():
        lines.append(f'  • {row["feature"]:20s}  value={row["value"]:.3g}  SHAP={row["shap"]:+.4f}')
    lines += ['', '▲ Top POSITIVE drivers (supporting soil health):']
    for _, row in pos_drivers[::-1].iterrows():
        lines.append(f'  • {row["feature"]:20s}  value={row["value"]:.3g}  SHAP={row["shap"]:+.4f}')

    report_text = '\n'.join(lines)
    print(report_text)

    return {
        'shi_predicted':       shi_pred,
        'risk_class':          risk,
        'top_positive_drivers': pos_drivers,
        'top_negative_drivers': neg_drivers,
        'text_report':         report_text,
        'waterfall_path':      wf_path,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SCENARIO / WHAT-IF ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

# Pre-defined management intervention scenarios
SCENARIOS = {
    'Liming': {
        'pH_H2O':   +1.5,   # Apply agricultural lime → raise pH
        'pH_CaCl2': +1.2,
        'CaCO3':    +50.0,  # g/kg
    },
    'Compost_addition': {
        'OC': +15.0,         # g/kg organic matter addition
        'N':  +0.8,
        'P':  +10.0,
    },
    'NPK_fertiliser': {
        'N': +1.5,
        'P': +30.0,
        'K': +80.0,
    },
    'Combined_management': {
        'pH_H2O':   +1.0,
        'OC':       +10.0,
        'N':        +1.0,
        'P':        +15.0,
        'K':        +40.0,
        'CaCO3':    +30.0,
    },
}


def simulate_scenario(
    model,
    point: pd.Series,
    intervention: dict,
    feature_names: list,
    scaler=None,
) -> tuple[float, float]:
    """
    Apply an intervention to a single soil point and predict the new SHI.

    Parameters
    ----------
    model       : trained model (XGBoost or RF)
    point       : pd.Series — a single row from the feature matrix
    intervention: dict of {feature: delta} — additive changes
    feature_names: list of feature column names
    scaler      : if the model was trained on scaled data, pass the fitted scaler

    Returns (original_shi, scenario_shi)
    """
    X_orig = point[feature_names].values.reshape(1, -1)

    # Apply intervention on original (unscaled) values
    point_mod = point[feature_names].copy()
    for feat, delta in intervention.items():
        if feat in point_mod.index:
            point_mod[feat] = max(0, point_mod[feat] + delta)

    X_mod = point_mod.values.reshape(1, -1)

    if scaler is not None:
        X_orig = scaler.transform(X_orig)
        X_mod  = scaler.transform(X_mod)

    shi_orig = float(model.predict(X_orig)[0])
    shi_mod  = float(model.predict(X_mod)[0])
    return shi_orig, shi_mod


def run_all_scenarios(
    model,
    df_features: pd.DataFrame,
    feature_names: list,
    scaler=None,
    n_sample: int = 500,
    save_dir: str = '.',
) -> pd.DataFrame:
    """
    Run all pre-defined scenarios on a sample of points.
    Returns a DataFrame with original SHI + delta SHI for each scenario.
    """
    log.info('=== Stage 8: Scenario Analysis ===')
    sample = df_features.sample(min(n_sample, len(df_features)), random_state=42)

    rows = []
    for idx, point in sample.iterrows():
        row = {'point_idx': idx}
        X = point[feature_names].values.reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        row['SHI_baseline'] = float(model.predict(X)[0])

        for sc_name, intervention in SCENARIOS.items():
            _, shi_sc = simulate_scenario(model, point, intervention, feature_names, scaler)
            row[f'SHI_{sc_name}']       = shi_sc
            row[f'delta_{sc_name}']     = shi_sc - row['SHI_baseline']
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index('point_idx')
    _plot_scenario_summary(results_df, save_dir)
    return results_df


def _plot_scenario_summary(scenario_df: pd.DataFrame, save_dir: str = '.') -> None:
    """Violin plot comparing SHI distributions under each scenario."""
    delta_cols = [c for c in scenario_df.columns if c.startswith('delta_')]
    labels     = [c.replace('delta_', '').replace('_', ' ') for c in delta_cols]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Delta SHI distributions
    data = [scenario_df[c].values for c in delta_cols]
    vp = axes[0].violinplot(data, showmedians=True)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.6)
    axes[0].axhline(0, color='red', ls='--', lw=1.5)
    axes[0].set_xticks(range(1, len(labels) + 1))
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set(ylabel='ΔSHI', title='SHI Change Under Each Management Scenario')

    # Mean delta bars
    mean_deltas = [scenario_df[c].mean() for c in delta_cols]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in mean_deltas]
    axes[1].barh(labels, mean_deltas, color=colors, edgecolor='white')
    axes[1].axvline(0, color='black', lw=0.8)
    for i, v in enumerate(mean_deltas):
        axes[1].text(v + 0.001 if v >= 0 else v - 0.001, i,
                     f'{v:+.4f}', va='center', fontsize=9,
                     ha='left' if v >= 0 else 'right')
    axes[1].set(xlabel='Mean ΔSHI', title='Mean SHI Change by Scenario')

    plt.suptitle('What-If Scenario Analysis — Impact on Soil Health Index', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scenario_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_single_scenario(
    model,
    point: pd.Series,
    feature_names: list,
    scaler=None,
    save_dir: str = '.',
    point_id: str = 'Point',
) -> None:
    """Bar chart comparing SHI across all scenarios for a single point."""
    baseline, _ = simulate_scenario(model, point, {}, feature_names, scaler)
    scenario_results = {'Baseline': baseline}
    for sc_name, intervention in SCENARIOS.items():
        _, shi_sc = simulate_scenario(model, point, intervention, feature_names, scaler)
        scenario_results[sc_name.replace('_', ' ')] = shi_sc

    labels = list(scenario_results.keys())
    values = list(scenario_results.values())
    colors = ['#7f7f7f'] + [
        '#2ca02c' if v >= baseline else '#d62728'
        for v in values[1:]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    ax.axhline(0.50, color='orange', ls='--', lw=1.5, label='Moderate risk threshold')
    ax.axhline(0.75, color='green',  ls='--', lw=1.5, label='Low risk threshold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set(ylim=(0, 1.05), ylabel='Predicted SHI',
           title=f'Scenario Comparison — {point_id}  (Baseline SHI={baseline:.3f})')
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scenario_point_{point_id}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6b — HYPERPARAMETERS, EXTENDED METRICS & CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def display_hyperparameters(results: dict) -> pd.DataFrame:
    """
    Print and return a tidy DataFrame of hyperparameters used by each model.
    Works for both regression and classification results dicts.
    """
    rows = []
    for name, res in results.items():
        model = res['model']
        params = model.get_params()
        for param, value in sorted(params.items()):
            rows.append({'model': name, 'hyperparameter': param, 'value': str(value)})
    df_params = pd.DataFrame(rows)

    # Pretty-print
    print('╔══════════════════════════════════════════════════════╗')
    print('║           MODEL HYPERPARAMETERS                      ║')
    print('╚══════════════════════════════════════════════════════╝')
    for name in df_params['model'].unique():
        print(f'\n── {name} ──────────────────────────────')
        sub = df_params[df_params['model'] == name]
        for _, row in sub.iterrows():
            print(f'  {row["hyperparameter"]:35s} = {row["value"]}')

    return df_params


def extended_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Return extended regression metrics beyond basic R²/MAE/RMSE.

    Metrics
    -------
    R²          : coefficient of determination
    Adj-R²      : adjusted R² (penalises excess features; requires n_features arg)
    MAE         : mean absolute error
    RMSE        : root mean squared error
    MAPE        : mean absolute percentage error (%)
    Max-Error   : maximum absolute residual
    Explained-Var : explained variance score
    """
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error,
        explained_variance_score, max_error,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # MAPE — skip zeros to avoid division error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {
        'R²':             round(r2, 5),
        'MAE':            round(mae, 5),
        'RMSE':           round(rmse, 5),
        'MAPE (%)':       round(mape, 3) if not np.isnan(mape) else None,
        'Max Error':      round(float(max_error(y_true, y_pred)), 5),
        'Explained Var':  round(float(explained_variance_score(y_true, y_pred)), 5),
    }


def print_extended_metrics(results: dict) -> pd.DataFrame:
    """Print and return extended regression metrics for all models."""
    rows = []
    for name, res in results.items():
        m = extended_regression_metrics(
            np.asarray(res['y_test']), np.asarray(res['y_pred'])
        )
        m['Model'] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index('Model')

    print('\n╔══════════════════════════════════════════════════════╗')
    print('║         EXTENDED REGRESSION METRICS                  ║')
    print('╚══════════════════════════════════════════════════════╝')
    print(df.to_string())
    return df


# ── Classification models ──────────────────────────────────────────────────

CLASS_LABELS = {0: 'Critical', 1: 'High', 2: 'Moderate', 3: 'Low'}


def train_classification_models(
    X: pd.DataFrame,
    y_class: pd.Series,
    *,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train XGBoost and Random Forest classifiers on SHI_CLASS (4 classes).

    Target: SHI_CLASS  0=Critical | 1=High | 2=Moderate | 3=Low

    Returns results dict with model, predictions, probabilities, and metrics.
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        classification_report, confusion_matrix,
    )
    import xgboost as xgb

    log.info('=== Stage 6b: Classification Models ===')

    # Drop NaN in target
    valid = y_class.dropna()
    X_cls = X.loc[valid.index].copy()
    y_cls = valid.astype(int).copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cls, y_cls, test_size=0.20,
        stratify=y_cls, random_state=random_state,
    )
    log.info('Train: %d  |  Test: %d  |  Classes: %s',
             len(X_tr), len(X_te), sorted(y_cls.unique()))

    n_cls = y_cls.nunique()

    models_def = {
        'XGBoost_Clf': xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            num_class=n_cls, random_state=random_state, n_jobs=-1,
        ),
        'RandomForest_Clf': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features=0.6, random_state=random_state, n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models_def.items():
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_f1  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='f1_macro', n_jobs=-1)

        metrics = {
            'Accuracy':       round(accuracy_score(y_te, y_pred), 4),
            'F1_macro':       round(f1_score(y_te, y_pred, average='macro'), 4),
            'F1_weighted':    round(f1_score(y_te, y_pred, average='weighted'), 4),
            'AUC_OvR':        round(roc_auc_score(y_te, y_proba, multi_class='ovr',
                                                   average='macro'), 4),
            'CV_Acc_mean':    round(cv_acc.mean(), 4),
            'CV_Acc_std':     round(cv_acc.std(), 4),
            'CV_F1_mean':     round(cv_f1.mean(), 4),
            'CV_F1_std':      round(cv_f1.std(), 4),
        }
        results[name] = {
            'model':         model,
            'metrics':       metrics,
            'y_pred':        y_pred,
            'y_proba':       y_proba,
            'y_test':        y_te,
            'X_test':        X_te,
            'X_train':       X_tr,
            'y_train':       y_tr,
            'clf_report':    classification_report(
                                y_te, y_pred,
                                target_names=[CLASS_LABELS[i] for i in sorted(y_cls.unique())],
                                output_dict=True,
                             ),
            'conf_matrix':   confusion_matrix(y_te, y_pred),
        }
        log.info('%s | Acc=%.4f  F1=%.4f  AUC=%.4f  CV-Acc=%.4f±%.4f',
                 name, metrics['Accuracy'], metrics['F1_macro'],
                 metrics['AUC_OvR'], metrics['CV_Acc_mean'], metrics['CV_Acc_std'])

    return results


def plot_confusion_matrices(
    class_results: dict,
    save_dir: str = '.',
    normalize: bool = True,
) -> None:
    """
    Plot normalised and raw confusion matrices for all classification models.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    n = len(class_results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    class_names = list(CLASS_LABELS.values())  # Critical, High, Moderate, Low

    for j, (name, res) in enumerate(class_results.items()):
        cm     = res['conf_matrix']
        y_test = res['y_test']
        y_pred = res['y_pred']

        # Normalised
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im1 = axes[0, j].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        axes[0, j].set_title(f'{name}\nNormalised Confusion Matrix', fontsize=11)
        plt.colorbar(im1, ax=axes[0, j], fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(class_names))
        axes[0, j].set_xticks(tick_marks); axes[0, j].set_yticks(tick_marks)
        axes[0, j].set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
        axes[0, j].set_yticklabels(class_names, fontsize=9)
        axes[0, j].set_ylabel('True label'); axes[0, j].set_xlabel('Predicted label')
        for i in range(len(class_names)):
            for k in range(len(class_names)):
                axes[0, j].text(k, i, f'{cm_norm[i, k]:.2f}',
                                ha='center', va='center',
                                color='white' if cm_norm[i, k] > 0.5 else 'black',
                                fontsize=10, fontweight='bold')

        # Raw counts
        im2 = axes[1, j].imshow(cm, interpolation='nearest', cmap='Oranges')
        axes[1, j].set_title(f'{name}\nRaw Count Confusion Matrix', fontsize=11)
        plt.colorbar(im2, ax=axes[1, j], fraction=0.046, pad=0.04)
        axes[1, j].set_xticks(tick_marks); axes[1, j].set_yticks(tick_marks)
        axes[1, j].set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
        axes[1, j].set_yticklabels(class_names, fontsize=9)
        axes[1, j].set_ylabel('True label'); axes[1, j].set_xlabel('Predicted label')
        for i in range(len(class_names)):
            for k in range(len(class_names)):
                axes[1, j].text(k, i, str(cm[i, k]),
                                ha='center', va='center',
                                color='white' if cm[i, k] > cm.max() * 0.6 else 'black',
                                fontsize=11, fontweight='bold')

    plt.suptitle('Confusion Matrices — SHI Risk Class Prediction', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_classification_report(
    class_results: dict,
    save_dir: str = '.',
) -> pd.DataFrame:
    """
    Display per-class precision, recall, F1 and support as a heatmap + table.
    Returns a merged DataFrame for all models.
    """
    all_dfs = []
    for name, res in class_results.items():
        report = res['clf_report']
        df_rep = pd.DataFrame(report).T
        df_rep['model'] = name
        all_dfs.append(df_rep)

    combined = pd.concat(all_dfs)

    for name, res in class_results.items():
        report = res['clf_report']
        # Keep only per-class rows
        class_rows = {k: v for k, v in report.items()
                      if k in list(CLASS_LABELS.values())}
        df_cls = pd.DataFrame(class_rows).T[['precision', 'recall', 'f1-score', 'support']]

        fig, ax = plt.subplots(figsize=(9, 4))
        hm_data = df_cls[['precision', 'recall', 'f1-score']].astype(float)
        sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax, linewidths=0.5,
                    annot_kws={'size': 12, 'weight': 'bold'})
        # Add support as text
        for i, (idx, row) in enumerate(df_cls.iterrows()):
            ax.text(3.5, i + 0.5, f'n={int(row["support"])}',
                    ha='center', va='center', fontsize=10)
        ax.set_title(f'{name} — Classification Report\n'
                     f'(Accuracy={res["metrics"]["Accuracy"]:.4f}  '
                     f'F1_macro={res["metrics"]["F1_macro"]:.4f}  '
                     f'AUC={res["metrics"]["AUC_OvR"]:.4f})',
                     fontsize=11, pad=10)
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/clf_report_{name}.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f'\n{name} — Full Metrics:')
        print(pd.DataFrame(res['metrics'], index=[0]).to_string(index=False))

    return combined


def plot_roc_curves(
    class_results: dict,
    save_dir: str = '.',
) -> None:
    """
    One-vs-Rest ROC curves for each class and each model.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    all_classes = sorted(CLASS_LABELS.keys())
    n = len(class_results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ax, (name, res) in zip(axes, class_results.items()):
        y_test  = np.asarray(res['y_test'])
        y_proba = res['y_proba']
        y_bin   = label_binarize(y_test, classes=all_classes)

        for i, cls in enumerate(all_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{CLASS_LABELS[cls]} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set(xlim=(0, 1), ylim=(0, 1.02),
               xlabel='False Positive Rate', ylabel='True Positive Rate',
               title=f'{name}\nOne-vs-Rest ROC Curves')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('ROC Curves — SHI Risk Class (One-vs-Rest)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_regression_vs_classification(
    reg_results: dict,
    clf_results: dict,
) -> pd.DataFrame:
    """
    Print a unified summary table comparing regression and classification metrics.
    """
    rows = []
    for name, res in reg_results.items():
        m = res['metrics']
        rows.append({
            'Model': name, 'Task': 'Regression',
            'R² / Accuracy': m['R2_test'],
            'RMSE / F1_macro': m['RMSE_test'],
            'MAE / AUC': m['MAE_test'],
            'CV Score': f"{m['R2_cv_mean']:.4f} ± {m['R2_cv_std']:.4f}",
        })
    for name, res in clf_results.items():
        m = res['metrics']
        rows.append({
            'Model': name, 'Task': 'Classification',
            'R² / Accuracy': m['Accuracy'],
            'RMSE / F1_macro': m['F1_macro'],
            'MAE / AUC': m['AUC_OvR'],
            'CV Score': f"{m['CV_Acc_mean']:.4f} ± {m['CV_Acc_std']:.4f}",
        })
    df = pd.DataFrame(rows).set_index('Model')
    print('\n╔══════════════════════════════════════════════════════════════╗')
    print('║         REGRESSION vs CLASSIFICATION — FULL SUMMARY         ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    print(df.to_string())
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE  — run everything end-to-end
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    master_df: pd.DataFrame,
    save_dir: str = 'output',
    run_shap: bool = True,
    run_scenarios: bool = True,
    run_hyperparam_search: bool = False,
) -> dict:
    """
    Execute all 8 stages on the Master DataFrame.

    Parameters
    ----------
    master_df  : output of data_preparation.build_master_dataframe()
    save_dir   : directory for saving all plots and artefacts
    run_shap   : set False to skip SHAP (speeds things up)
    run_scenarios : set False to skip scenario analysis
    run_hyperparam_search : set True to run RandomizedSearchCV

    Returns
    -------
    dict with keys: df, shap_results, scenario_results, model_results, feature_names
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Stage 2: EDA ──────────────────────────────────────────────────────
    log.info('── Stage 2: EDA ──')
    run_eda(master_df, save_dir)

    # ── Stage 3: Impute + log-transform ───────────────────────────────────
    log.info('── Stage 3: Imputation & Transforms ──')
    df = impute_features(master_df)
    df = apply_log_transforms(df)

    # ── Stage 4: SHI ──────────────────────────────────────────────────────
    log.info('── Stage 4: SHI Computation ──')
    df = compute_shi(df)
    plot_shi_distribution(df, save_dir)
    plot_shi_spatial(df, save_dir)
    plot_shi_by_land_cover(df, save_dir)

    # ── Stage 5: Feature engineering ──────────────────────────────────────
    log.info('── Stage 5: Feature Engineering ──')
    df = encode_categorical(df)
    X, y, feature_names = build_feature_set(df)
    X_scaled, scaler = scale_features(X)

    # ── Stage 6: Modelling ────────────────────────────────────────────────
    log.info('── Stage 6: Modelling ──')
    model_results = train_and_evaluate(
        X_scaled, y,
        run_hyperparam_search=run_hyperparam_search,
    )
    plot_model_performance(model_results, save_dir)
    plot_feature_importance(model_results, feature_names, save_dir)

    # Hyperparameters + extended metrics
    display_hyperparameters(model_results)
    print_extended_metrics(model_results)

    # Classification branch (confusion matrix, ROC, per-class F1)
    if 'SHI_CLASS' in df.columns:
        y_class = df.loc[X_scaled.index, 'SHI_CLASS']
        clf_results = train_classification_models(X_scaled, y_class)
        plot_confusion_matrices(clf_results, save_dir)
        plot_classification_report(clf_results, save_dir)
        plot_roc_curves(clf_results, save_dir)
        compare_regression_vs_classification(model_results, clf_results)
    else:
        clf_results = {}

    # Pick best model by test R²
    best_name  = max(model_results, key=lambda k: model_results[k]['metrics']['R2_test'])
    best_model = model_results[best_name]['model']
    log.info('Best model: %s  (R²=%.4f)', best_name,
             model_results[best_name]['metrics']['R2_test'])

    # ── Stage 7: SHAP ─────────────────────────────────────────────────────
    shap_results = {}
    if run_shap:
        log.info('── Stage 7: SHAP ──')
        shap_results = run_shap_analysis(model_results, feature_names, save_dir)

        # Diagnostic report for 3 example points (low / mid / high SHI)
        X_test = model_results[best_name]['X_test']
        y_pred = model_results[best_name]['y_pred']
        pred_series = pd.Series(y_pred, index=X_test.index)
        for label, idx in [
            ('CriticalPoint', pred_series.idxmin()),
            ('MidPoint',      pred_series.index[len(pred_series)//2]),
            ('HealthyPoint',  pred_series.idxmax()),
        ]:
            generate_diagnostic_report(
                best_model, X_scaled.loc[idx],
                model_results[best_name]['X_train'],
                feature_names, point_id=label, save_dir=save_dir,
            )

    # ── Stage 8: Scenarios ────────────────────────────────────────────────
    scenario_results = {}
    if run_scenarios:
        log.info('── Stage 8: Scenarios ──')
        scenario_results = run_all_scenarios(
            best_model, X_scaled, feature_names,
            n_sample=500, save_dir=save_dir,
        )
        # Single-point scenario for the most degraded point
        worst_idx = X_scaled.index[y.loc[X_scaled.index].argmin()]
        plot_single_scenario(
            best_model, X_scaled.loc[worst_idx], feature_names,
            save_dir=save_dir, point_id='MostDegraded',
        )

    log.info('═══ Pipeline complete. All outputs in: %s ═══', save_dir)

    return {
        'df':               df,
        'X':                X_scaled,
        'y':                y,
        'feature_names':    feature_names,
        'scaler':           scaler,
        'model_results':    model_results,
        'best_model':       best_model,
        'best_model_name':  best_name,
        'shap_results':     shap_results,
        'scenario_results': scenario_results,
    }
