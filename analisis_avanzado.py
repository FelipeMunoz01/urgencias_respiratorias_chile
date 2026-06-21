"""
================================================================================
Análisis Avanzado — Urgencias Respiratorias Chile 2014–2026
Continuación del estudio principal (analisis_urgencias_respiratorias.py)
================================================================================

Patrones no evidentes descubiertos en este módulo:

    08. Correlaciones cruzadas entre causas con lag temporal
    09. Anomalías regionales durante COVID-19 (z-scores semana × región)
    10. Efecto vacunación: slope de crecimiento pre-pico invernal por año
    11. Recuperación etaria post-pandemia (normalizada vs. 2017-2019)
    12. SAPU vs Hospital: cambio estructural en la distribución post-COVID

Requiere el mismo dataset que el análisis principal:
    data/at_urg_respiratorio_semanal.parquet

Autor: Felipe Muñoz
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import correlate

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_PATH  = Path("data/at_urg_respiratorio_semanal.parquet")
OUTPUT_DIR = Path("outputs/figuras")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAS = {
    "IRA Alta":           "IRA Alta (J00-J06)",
    "Influenza":          "Influenza (J09-J11)",
    "Neumonía":           "Neumonía (J12-J18)",
    "Bronquitis":         "Bronquitis/bronquiolitis aguda (J20-J21)",
    "Crisis Obstructiva": "Crisis obstructiva bronquial (J40-J46)",
    "COVID-19":           "TOTAL ATENCIONES POR COVID-19, Virus Identificado U07.1",
}
CAUSA_TOTAL = "TOTAL CAUSA SISTEMA  RESPIRATORIO (J00-J98)"

COLORES = {
    "IRA Alta":           "#58a6ff",
    "Influenza":          "#e63946",
    "Neumonía":           "#3fb950",
    "Bronquitis":         "#2a9d8f",
    "Crisis Obstructiva": "#e9c46a",
    "COVID-19":           "#f4a261",
}

FONDO  = "#0d1117"
PANEL  = "#161b22"
TEXTO  = "#c9d1d9"
GRID   = "#21262d"
ROJO   = "#f85149"
VERDE  = "#3fb950"
AZUL   = "#58a6ff"
MORADO = "#a371f7"
ANIOS_COVID = {2020, 2021}

plt.rcParams.update({
    "figure.facecolor":  FONDO,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "text.color":        TEXTO,
    "axes.labelcolor":   TEXTO,
    "xtick.color":       TEXTO,
    "ytick.color":       TEXTO,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
})


def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = OUTPUT_DIR / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


# =============================================================================
# CARGA
# =============================================================================

def cargar() -> pd.DataFrame:
    print("Cargando datos...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df["Fecha"] = pd.to_datetime(
        df["Anio"].astype(str) + "-W" +
        df["SemanaEstadistica"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce",
    )
    df = df[~((df["Anio"] == 2026) & (df["SemanaEstadistica"] == 53))]
    print(f"{len(df):,} registros")
    return df


# =============================================================================
# ANÁLISIS 08 — CORRELACIONES CRUZADAS CON LAG
# ¿Cuando sube Influenza, qué pasa con las otras causas 1-4 semanas después?
# =============================================================================

def fig_correlaciones_cruzadas(df: pd.DataFrame) -> plt.Figure:
    causas = list(CAUSAS.keys())
    MAX_LAG = 12  # semanas

    # Series semanales nacionales (excluir COVID para correlaciones limpias)
    pivot = pd.DataFrame()
    for k, v in CAUSAS.items():
        s = df[df["Causa"] == v].groupby("Fecha")["NumTotal"].sum().sort_index()
        pivot[k] = s

    pivot = pivot.dropna()

    # Normalizar cada serie (z-score) para comparar escalas distintas
    norm = (pivot - pivot.mean()) / pivot.std()

    # Calcular correlación cruzada máxima (y el lag donde ocurre)
    n = len(causas)
    corr_max  = np.zeros((n, n))
    lag_max   = np.zeros((n, n), dtype=int)

    for i, c1 in enumerate(causas):
        for j, c2 in enumerate(causas):
            x = norm[c1].values
            y = norm[c2].values
            cc = correlate(x, y, mode="full")
            lags = np.arange(-(len(x) - 1), len(x))
            # Solo lags entre -MAX_LAG y +MAX_LAG
            mask = (lags >= -MAX_LAG) & (lags <= MAX_LAG)
            cc_sub  = cc[mask]
            lag_sub = lags[mask]
            # Normalizar
            cc_norm = cc_sub / (len(x) * norm[c1].std() * norm[c2].std())
            idx_max = np.argmax(np.abs(cc_norm))
            corr_max[i, j] = cc_norm[idx_max]
            lag_max[i, j]  = lag_sub[idx_max]

    # ── Figura: dos paneles ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel izquierdo: correlación máxima
    sns.heatmap(
        corr_max, ax=ax1,
        xticklabels=causas, yticklabels=causas,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.4, linecolor=FONDO,
        cbar_kws={"label": "Correlación cruzada", "shrink": 0.7},
    )
    ax1.set_title("Correlación cruzada máxima\n(ventana ±12 semanas)",
                  fontsize=12, fontweight="bold", pad=12, color=TEXTO)
    ax1.tick_params(axis="x", rotation=35, labelsize=8)
    ax1.tick_params(axis="y", rotation=0,  labelsize=8)

    # Panel derecho: lag donde ocurre la correlación máxima
    sns.heatmap(
        lag_max, ax=ax2,
        xticklabels=causas, yticklabels=causas,
        cmap="coolwarm", center=0,
        annot=True, fmt="d", annot_kws={"size": 9},
        linewidths=0.4, linecolor=FONDO,
        cbar_kws={"label": "Lag (semanas)", "shrink": 0.7},
    )
    ax2.set_title("Lag temporal de la correlación máxima\n(semanas, negativo = fila precede a columna)",
                  fontsize=12, fontweight="bold", pad=12, color=TEXTO)
    ax2.tick_params(axis="x", rotation=35, labelsize=8)
    ax2.tick_params(axis="y", rotation=0,  labelsize=8)

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXTO)
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(colors=TEXTO)
        cb.ax.yaxis.label.set_color(TEXTO)

    fig.suptitle("Correlaciones Cruzadas entre Causas Respiratorias — Chile 2014–2026",
                 fontsize=14, fontweight="bold", y=1.01, color=TEXTO)
    fig.tight_layout()

    # Hallazgo principal
    # Encontrar el par con mayor correlación (fuera de la diagonal)
    corr_off = corr_max.copy()
    np.fill_diagonal(corr_off, 0)
    idx = np.unravel_index(np.argmax(np.abs(corr_off)), corr_off.shape)
    print(f"\n  HALLAZGO 08: '{causas[idx[0]]}' y '{causas[idx[1]]}' "
          f"tienen la mayor correlación cruzada: {corr_off[idx]:.2f} "
          f"con lag {int(lag_max[idx])} semanas.")

    return fig


# =============================================================================
# ANÁLISIS 09 — ANOMALÍAS REGIONALES COVID-19
# Z-score de cada región vs. su propio promedio histórico (2015-2019)
# =============================================================================

def fig_anomalias_regionales(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["Causa"] == CAUSA_TOTAL].copy()

    # Semanas por región y año
    reg_sem = (
        sub.groupby(["RegionGlosa", "Anio", "SemanaEstadistica"])["NumTotal"]
        .sum()
        .reset_index()
    )

    # Baseline: promedio y std por región × semana (años 2015-2019)
    base = (
        reg_sem[reg_sem["Anio"].between(2015, 2019)]
        .groupby(["RegionGlosa", "SemanaEstadistica"])["NumTotal"]
        .agg(["mean", "std"])
        .reset_index()
    )
    base.columns = ["RegionGlosa", "SemanaEstadistica", "media", "std"]
    base["std"] = base["std"].replace(0, np.nan).fillna(1)

    # Años COVID: 2020-2021, semanas 1-52
    covid = reg_sem[reg_sem["Anio"].isin([2020, 2021]) &
                    (reg_sem["SemanaEstadistica"] <= 52)].copy()
    covid = covid.merge(base, on=["RegionGlosa", "SemanaEstadistica"], how="left")
    covid["zscore"] = (covid["NumTotal"] - covid["media"]) / covid["std"]
    covid["periodo"] = covid["Anio"].astype(str) + "-S" + \
                       covid["SemanaEstadistica"].astype(str).str.zfill(2)

    # Pivot: región × semana (promedio de zscore entre 2020 y 2021)
    pivot = (
        covid.groupby(["RegionGlosa", "SemanaEstadistica"])["zscore"]
        .mean()
        .unstack("SemanaEstadistica")
        .fillna(0)
    )
    # Ordenar regiones por anomalía total
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]

    # Recortar nombre de regiones
    pivot.index = pivot.index.str.replace(r"Región (del?|de la|de los|de) ", "", regex=True)
    pivot.index = pivot.index.str.slice(0, 22)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdBu_r", center=0, vmin=-4, vmax=4,
        linewidths=0, linecolor=FONDO,
        cbar_kws={"label": "Z-score vs. histórico 2015–2019", "shrink": 0.6},
        xticklabels=4,
    )
    ax.set_title(
        "Anomalías Regionales durante COVID-19 (2020–2021)\n"
        "Z-score semana × región respecto al promedio histórico 2015–2019",
        fontsize=13, fontweight="bold", pad=14, color=TEXTO,
    )
    ax.set_xlabel("Semana Epidemiológica", fontsize=11, labelpad=6, color=TEXTO)
    ax.set_ylabel("", fontsize=0)
    ax.tick_params(axis="x", colors=TEXTO, labelsize=8)
    ax.tick_params(axis="y", colors=TEXTO, labelsize=8, rotation=0)
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(colors=TEXTO)
    cb.ax.yaxis.label.set_color(TEXTO)
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(PANEL)
    fig.tight_layout()

    # Hallazgo
    region_max = pivot.abs().sum(axis=1).idxmax()
    zscore_max = pivot.abs().values.max()
    print(f"  HALLAZGO 09: '{region_max}' fue la región con mayor desviación "
          f"durante COVID-19 (z-score máx: {zscore_max:.1f}σ).")

    return fig


# =============================================================================
# ANÁLISIS 10 — EFECTO VACUNACIÓN (slope semanas 14→26)
# La campaña de vacunación influenza ocurre en semanas 14-17 (abril).
# ¿Cambió la velocidad de crecimiento hacia el pico invernal?
# =============================================================================

def fig_efecto_vacunacion(df: pd.DataFrame) -> plt.Figure:
    inf_val = CAUSAS["Influenza"]
    sub = df[df["Causa"] == inf_val].copy()

    anios = [a for a in range(2015, 2027) if a not in ANIOS_COVID]
    slopes = {}

    for anio in anios:
        s = (
            sub[sub["Anio"] == anio]
            .groupby("SemanaEstadistica")["NumTotal"].sum()
        )
        s = s[(s.index >= 14) & (s.index <= 26)]
        if len(s) < 5:
            continue
        x = s.index.values.astype(float)
        y = s.values.astype(float)
        slope, intercept, r, p, _ = stats.linregress(x, y)
        slopes[anio] = {"slope": slope, "r2": r**2, "p": p}

    df_slopes = pd.DataFrame(slopes).T.reset_index()
    df_slopes.columns = ["Anio", "slope", "r2", "p"]
    df_slopes["Anio"] = df_slopes["Anio"].astype(int)

    # Tendencia del slope a lo largo de los años
    z = np.polyfit(df_slopes["Anio"], df_slopes["slope"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df_slopes["Anio"].min(), df_slopes["Anio"].max(), 100)

    # Colores: antes de vacunación masiva (2015-2019) vs post (2022+)
    colors = [AZUL if a <= 2019 else VERDE for a in df_slopes["Anio"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel izquierdo: slope por año
    bars = ax1.bar(df_slopes["Anio"], df_slopes["slope"],
                   color=colors, alpha=0.82, edgecolor=FONDO,
                   linewidth=0.5, width=0.7)
    ax1.plot(x_line, p_line(x_line), color=ROJO, linewidth=1.8,
             linestyle="--", alpha=0.7, label="Tendencia")
    ax1.axhline(0, color=TEXTO, linewidth=0.5, alpha=0.3)

    # Leyenda de colores
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color=AZUL,  label="Pre-COVID (2015–2019)"),
        Patch(color=VERDE, label="Post-COVID (2022–2026)"),
        plt.Line2D([0], [0], color=ROJO, linestyle="--", label="Tendencia"),
    ], fontsize=9)

    ax1.set_xticks(df_slopes["Anio"])
    ax1.set_xticklabels(df_slopes["Anio"].astype(str), rotation=45, fontsize=9)
    ax1.set_ylabel("Atenciones/semana (slope)", fontsize=11, labelpad=8)
    ax1.set_title("Velocidad de crecimiento de Influenza\n(semanas 14→26, campaña vacunación = sem. 14–17)",
                  fontsize=11, fontweight="bold", pad=12, color=TEXTO)
    ax1.grid(axis="y")

    # Panel derecho: series semana 14-26 por año superpuestas
    for i, row in df_slopes.iterrows():
        anio = int(row["Anio"])
        s = (
            sub[sub["Anio"] == anio]
            .groupby("SemanaEstadistica")["NumTotal"].sum()
        )
        s = s[(s.index >= 14) & (s.index <= 26)]
        color = AZUL if anio <= 2019 else VERDE
        alpha = 0.6 if anio not in [2024, 2025] else 1.0
        lw    = 1.2 if anio not in [2024, 2025] else 2.2
        ax2.plot(s.index, s.values, color=color, alpha=alpha,
                 linewidth=lw, label=str(anio))

    ax2.axvspan(14, 17, alpha=0.10, color="#e9c46a")
    ax2.text(15.5, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1,
             "Vacunación", ha="center", fontsize=8, color="#e9c46a", alpha=0.8)
    ax2.set_xlabel("Semana Epidemiológica", fontsize=11, labelpad=6)
    ax2.set_ylabel("Atenciones semanales", fontsize=11, labelpad=8)
    ax2.set_title("Series semanales Influenza (sem. 14–26)\npor año",
                  fontsize=11, fontweight="bold", pad=12, color=TEXTO)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"
    ))
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(axis="y")

    fig.suptitle("Efecto Vacunación: Slope de Crecimiento Pre-Pico Invernal",
                 fontsize=14, fontweight="bold", y=1.01, color=TEXTO)
    fig.tight_layout()

    # Hallazgo
    pre  = df_slopes[df_slopes["Anio"] <= 2019]["slope"].mean()
    post = df_slopes[df_slopes["Anio"] >= 2022]["slope"].mean()
    cambio = (post - pre) / abs(pre) * 100
    print(f"  HALLAZGO 10: El slope de crecimiento pre-pico invernal cambió "
          f"{cambio:+.1f}% post-pandemia ({pre:.0f} → {post:.0f} atenciones/semana). "
          f"{'Aceleración' if cambio > 0 else 'Desaceleración'} del brote.")

    return fig


# =============================================================================
# ANÁLISIS 11 — RECUPERACIÓN ETARIA POST-PANDEMIA
# ¿Qué grupo etario volvió a niveles pre-pandemia más rápido?
# =============================================================================

def fig_recuperacion_etaria(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["Causa"] == CAUSA_TOTAL].copy()

    cols_edad = {
        "< 1 año":    "NumMenor1Anio",
        "1–4 años":   "Num1a4Anios",
        "5–14 años":  "Num5a14Anios",
        "15–64 años": "Num15a64Anios",
        "≥ 65 años":  "Num65oMas",
    }

    # Baseline: promedio anual 2017-2019 por grupo etario
    base = (
        sub[sub["Anio"].between(2017, 2019)]
        .groupby("Anio")[list(cols_edad.values())]
        .sum()
        .mean()  # promedio de los 3 años
    )

    # Totales anuales 2020-2026 por grupo etario
    anual = (
        sub[sub["Anio"].between(2020, 2026)]
        .groupby("Anio")[list(cols_edad.values())]
        .sum()
    )

    # Índice de recuperación: 100% = nivel pre-pandemia
    recovery = pd.DataFrame(index=anual.index)
    for label, col in cols_edad.items():
        recovery[label] = 100 * anual[col] / base[col]

    colores_edad = [AZUL, VERDE, "#e9c46a", "#f4a261", MORADO]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel izquierdo: líneas de recuperación por grupo
    for (label, _), color in zip(cols_edad.items(), colores_edad):
        ax1.plot(recovery.index, recovery[label],
                 marker="o", markersize=6, color=color,
                 linewidth=2, label=label)

    ax1.axhline(100, color=TEXTO, linewidth=1, linestyle="--",
                alpha=0.4, label="Nivel pre-pandemia (2017–2019)")
    ax1.fill_between(recovery.index, 100, alpha=0.04, color=VERDE)
    ax1.set_xticks(recovery.index)
    ax1.set_xticklabels(recovery.index.astype(str), rotation=45)
    ax1.set_ylabel("% respecto al promedio 2017–2019", fontsize=11, labelpad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.set_title("Recuperación de Atenciones por Grupo Etario\n(100% = nivel pre-pandemia)",
                  fontsize=11, fontweight="bold", pad=12, color=TEXTO)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y")

    # Panel derecho: heatmap año × grupo (% recuperación)
    sns.heatmap(
        recovery.T, ax=ax2,
        cmap="RdYlGn", center=100, vmin=20, vmax=150,
        annot=True, fmt=".0f", annot_kws={"size": 9},
        linewidths=0.4, linecolor=FONDO,
        cbar_kws={"label": "% vs. 2017–2019", "shrink": 0.7},
    )
    ax2.set_title("Mapa de Recuperación (%)\npor año y grupo etario",
                  fontsize=11, fontweight="bold", pad=12, color=TEXTO)
    ax2.set_xlabel("Año", fontsize=11, labelpad=6, color=TEXTO)
    ax2.tick_params(axis="x", colors=TEXTO, labelsize=9, rotation=45)
    ax2.tick_params(axis="y", colors=TEXTO, labelsize=9, rotation=0)
    cb = ax2.collections[0].colorbar
    cb.ax.tick_params(colors=TEXTO)
    cb.ax.yaxis.label.set_color(TEXTO)
    ax2.set_facecolor(PANEL)

    fig.suptitle("Recuperación Etaria Post-Pandemia — Urgencias Respiratorias Chile",
                 fontsize=14, fontweight="bold", y=1.01, color=TEXTO)
    fig.tight_layout()

    # Hallazgo: grupo que recuperó más rápido (primer año en superar 90%)
    primero = {}
    for label in cols_edad.keys():
        r = recovery[label]
        sup90 = r[r >= 90]
        primero[label] = sup90.index[0] if len(sup90) else 9999

    grupo_rapido = min(primero, key=primero.get)
    año_rapido   = primero[grupo_rapido]
    print(f"  HALLAZGO 11: El grupo '{grupo_rapido}' fue el primero en recuperar "
          f"el 90% del nivel pre-pandemia (año {año_rapido}).")

    return fig


# =============================================================================
# ANÁLISIS 12 — SAPU vs HOSPITAL: cambio estructural post-COVID
# =============================================================================

def fig_sapu_vs_hospital(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["Causa"] == CAUSA_TOTAL].dropna(subset=["TipoUrgencia"]).copy()

    mapa = {
        "Urgencia Hospitalaria (UEH)": "Hospital (UEH)",
        "Urgencia Ambulatoria (SAPU)": "SAPU",
        "Urgencia ambulatoria (SAR)":  "SAR/SUR",
        "Urgencia Ambulatoria (SAR)":  "SAR/SUR",
        "Urgencia Ambulatoria (SUR)":  "SAR/SUR",
    }
    sub["Tipo"] = sub["TipoUrgencia"].map(mapa)
    sub = sub.dropna(subset=["Tipo"])

    anual = (
        sub.groupby(["Anio", "Tipo"])["NumTotal"]
        .sum()
        .unstack("Tipo")
        .fillna(0)
    )

    # Proporción porcentual
    prop = anual.div(anual.sum(axis=1), axis=0) * 100

    colores_tipo = {
        "Hospital (UEH)": AZUL,
        "SAPU":           VERDE,
        "SAR/SUR":        "#e9c46a",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel superior: volumen absoluto apilado
    bottom = np.zeros(len(anual))
    for tipo, color in colores_tipo.items():
        if tipo not in anual.columns:
            continue
        vals = anual[tipo].values / 1e6
        ax1.bar(anual.index, vals, bottom=bottom,
                color=color, alpha=0.82, label=tipo,
                edgecolor=FONDO, linewidth=0.4, width=0.75)
        bottom += vals

    for anio in ANIOS_COVID:
        ax1.axvspan(anio - 0.4, anio + 0.4, alpha=0.12, color=ROJO)

    ax1.set_xticks(anual.index)
    ax1.set_xticklabels(anual.index.astype(str), rotation=45)
    ax1.set_ylabel("Atenciones (millones)", fontsize=11, labelpad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.1f}M"
    ))
    ax1.set_title("Atenciones por Tipo de Establecimiento — Volumen Absoluto",
                  fontsize=12, fontweight="bold", pad=10, color=TEXTO)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y")

    # Panel inferior: proporción porcentual
    bottom_p = np.zeros(len(prop))
    for tipo, color in colores_tipo.items():
        if tipo not in prop.columns:
            continue
        vals = prop[tipo].values
        ax2.bar(prop.index, vals, bottom=bottom_p,
                color=color, alpha=0.82, label=tipo,
                edgecolor=FONDO, linewidth=0.4, width=0.75)
        # Etiqueta dentro de la barra si hay espacio
        for xi, (vi, bi) in enumerate(zip(vals, bottom_p)):
            if vi > 4:
                ax2.text(prop.index[xi], bi + vi / 2,
                         f"{vi:.1f}%",
                         ha="center", va="center",
                         fontsize=7.5, color=FONDO, fontweight="bold")
        bottom_p += vals

    for anio in ANIOS_COVID:
        ax2.axvspan(anio - 0.4, anio + 0.4, alpha=0.12, color=ROJO)

    ax2.set_xticks(prop.index)
    ax2.set_xticklabels(prop.index.astype(str), rotation=45)
    ax2.set_ylabel("Proporción (%)", fontsize=11, labelpad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, 105)
    ax2.set_title("Distribución Porcentual por Tipo de Establecimiento",
                  fontsize=12, fontweight="bold", pad=10, color=TEXTO)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y")

    fig.suptitle("SAPU vs Hospital — ¿Cambió la Distribución Post-COVID?",
                 fontsize=14, fontweight="bold", y=1.01, color=TEXTO)
    fig.tight_layout()

    # Hallazgo: ¿cambió la proporción SAPU entre pre y post COVID?
    pre_sapu  = prop.loc[prop.index.isin(range(2017, 2020)), "SAPU"].mean() if "SAPU" in prop else 0
    post_sapu = prop.loc[prop.index.isin(range(2022, 2027)), "SAPU"].mean() if "SAPU" in prop else 0
    diff = post_sapu - pre_sapu
    print(f"  HALLAZGO 12: La proporción de atenciones en SAPU cambió "
          f"{diff:+.1f} puntos porcentuales post-COVID "
          f"({pre_sapu:.1f}% → {post_sapu:.1f}%). "
          f"{'Migración hacia SAPU' if diff > 0 else 'Migración hacia hospital'}.")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  ANÁLISIS AVANZADO — URGENCIAS RESPIRATORIAS CHILE 2014–2026")
    print("=" * 65)

    df = cargar()
    print()
    print("Generando análisis avanzados...\n")

    print("08. Correlaciones cruzadas entre causas...")
    _guardar(fig_correlaciones_cruzadas(df), "08_correlaciones_cruzadas")

    print("09. Anomalías regionales COVID-19...")
    _guardar(fig_anomalias_regionales(df), "09_anomalias_regionales")

    print("10. Efecto vacunación (slope invernal)...")
    _guardar(fig_efecto_vacunacion(df), "10_efecto_vacunacion")

    print("11. Recuperación etaria post-pandemia...")
    _guardar(fig_recuperacion_etaria(df), "11_recuperacion_etaria")

    print("12. SAPU vs Hospital post-COVID...")
    _guardar(fig_sapu_vs_hospital(df), "12_sapu_vs_hospital")

    print(f"\n  Gráficos en: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
