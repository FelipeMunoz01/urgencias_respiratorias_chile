"""
================================================================================
Atenciones de Urgencia Respiratoria en Chile — 2014 a 2026
Análisis exploratorio, estacional y predicción con SARIMA
================================================================================

Fuente:
    MINSAL / Portal de Datos Abiertos del Gobierno de Chile
    https://datos.gob.cl/dataset/606ef5bb-11d1-475b-b69f-b980da5757f4

Dataset:
    - 3.5 millones de registros semanales por establecimiento
    - Cubre hospitales, SAPU, SAR y SUR de todo Chile
    - Variables: causa (CIE-10), año, semana, grupos etarios, región

Análisis:
    1. Tendencia anual 2014–2026
    2. Series semanales por causa respiratoria
    3. Heatmap de estacionalidad (semana × año)
    4. Distribución etaria a lo largo del tiempo
    5. Atenciones por tipo de establecimiento
    6. Descomposición STL de la serie de Influenza
    7. Modelo SARIMA: ajuste + predicción 2026

Autor: Felipe Muñoz
"""

# =============================================================================
# 1. IMPORTACIONES
# =============================================================================
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

DATA_PATH  = Path("data/at_urg_respiratorio_semanal.parquet")
OUTPUT_DIR = Path("outputs/figuras")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapa de causas (nombre corto → valor exacto en el dataset)
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
    "Neumonía":           "#457b9d",
    "Bronquitis":         "#2a9d8f",
    "Crisis Obstructiva": "#e9c46a",
    "COVID-19":           "#f4a261",
}

# Paleta visual (dark / GitHub style)
FONDO  = "#0d1117"
PANEL  = "#161b22"
TEXTO  = "#c9d1d9"
GRID   = "#21262d"
ROJO   = "#f85149"
VERDE  = "#3fb950"
AZUL   = "#58a6ff"
MORADO = "#a371f7"

ANIOS_COVID = [2020, 2021]

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


def _fmt_miles(v, _=None):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{int(v):,}"
    return str(int(v))


def _guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = OUTPUT_DIR / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


# =============================================================================
# 3. CARGA Y PREPARACIÓN
# =============================================================================

def cargar() -> pd.DataFrame:
    print("Cargando datos...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH)
    # Construir fecha ISO (lunes de cada semana epidemiológica)
    df["Fecha"] = pd.to_datetime(
        df["Anio"].astype(str) + "-W" +
        df["SemanaEstadistica"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce",
    )
    # Excluir la semana 53 de 2026 (artefacto del calendario)
    df = df[~((df["Anio"] == 2026) & (df["SemanaEstadistica"] == 53))]
    print(f"{len(df):,} registros | {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")
    return df


def serie_semanal(df: pd.DataFrame, causa_key: str) -> pd.Series:
    val = CAUSAS[causa_key]
    return (
        df[df["Causa"] == val]
        .groupby("Fecha")["NumTotal"].sum()
        .sort_index()
    )


def serie_semanal_total(df: pd.DataFrame) -> pd.Series:
    return (
        df[df["Causa"] == CAUSA_TOTAL]
        .groupby("Fecha")["NumTotal"].sum()
        .sort_index()
    )


def serie_mensual(s: pd.Series) -> pd.Series:
    return s.resample("MS").sum()


# =============================================================================
# 4. GRÁFICOS ESTÁTICOS
# =============================================================================

# ── Gráfico 1: Tendencia anual ──────────────────────────────────────────────

def fig_tendencia_anual(df: pd.DataFrame) -> plt.Figure:
    sub   = df[df["Causa"] == CAUSA_TOTAL]
    anual = sub.groupby("Anio")["NumTotal"].sum()
    anios = anual.index.astype(int)
    vals  = anual.values

    fig, ax = plt.subplots(figsize=(14, 6))

    # Barras con gradiente de color por año
    cmap   = plt.cm.get_cmap("cool", len(anios))
    bcolors = [ROJO if a in ANIOS_COVID else cmap(i) for i, a in enumerate(anios)]

    bars = ax.bar(anios, vals / 1e6, color=bcolors, alpha=0.80,
                  edgecolor=FONDO, linewidth=0.5, width=0.75, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{v/1e6:.1f}M",
                ha="center", va="bottom", fontsize=8.5,
                color=TEXTO, fontweight="bold")

    # Línea de tendencia (excluye COVID)
    mask  = ~np.isin(anios, ANIOS_COVID)
    x_fit = anios[mask]
    y_fit = vals[mask] / 1e6
    z     = np.polyfit(x_fit, y_fit, 1)
    p     = np.poly1d(z)
    x_line = np.linspace(anios[0], anios[-1], 200)
    ax.plot(x_line, p(x_line), color=VERDE, linewidth=1.5,
            linestyle="--", alpha=0.7, label="Tendencia (sin COVID)", zorder=4)

    # Anotación COVID
    ax.annotate("COVID-19\n(confinamiento)", xy=(2020, vals[anios == 2020][0] / 1e6),
                xytext=(2018.5, 2.8),
                arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.3),
                color=ROJO, fontsize=9, fontweight="bold")

    ax.set_xticks(anios)
    ax.set_xticklabels(anios.astype(str), rotation=45, fontsize=9)
    ax.set_ylabel("Atenciones (millones)", fontsize=11, labelpad=8)
    ax.set_title(
        "Atenciones de Urgencia Respiratoria — Chile 2014–2026",
        fontsize=15, fontweight="bold", pad=18, color=TEXTO,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    return fig


# ── Gráfico 2: Series semanales por causa ───────────────────────────────────

def fig_series_semanales(series: dict) -> plt.Figure:
    causas = ["IRA Alta", "Influenza", "Neumonía", "Bronquitis", "Crisis Obstructiva", "COVID-19"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    axes = axes.flatten()

    for ax, causa in zip(axes, causas):
        s     = series[causa]
        color = COLORES[causa]

        ax.fill_between(s.index, s.values, alpha=0.20, color=color)
        ax.plot(s.index, s.values, color=color, linewidth=0.9, alpha=0.95)

        # Banda COVID
        for anio in ANIOS_COVID:
            ax.axvspan(pd.Timestamp(f"{anio}-01-01"),
                       pd.Timestamp(f"{anio}-12-31"),
                       alpha=0.10, color=ROJO)

        # Peak máximo
        idx_max = s.idxmax()
        ax.annotate(
            f"Máx: {int(s.max()):,}",
            xy=(idx_max, s.max()),
            xytext=(idx_max + pd.DateOffset(weeks=30), s.max() * 0.88),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
            color=color, fontsize=7.5,
        )

        ax.set_title(causa, fontsize=12, fontweight="bold",
                     color=color, pad=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))
        ax.grid(axis="y")

    fig.suptitle(
        "Urgencias Respiratorias Semanales por Causa — Chile 2014–2026",
        fontsize=15, fontweight="bold", y=1.01, color=TEXTO,
    )
    fig.supxlabel("Semana epidemiológica", fontsize=11, color=TEXTO, y=-0.01)
    fig.tight_layout()
    return fig


# ── Gráfico 3: Heatmap semana × año ─────────────────────────────────────────

def fig_heatmap(series: dict) -> plt.Figure:
    s   = series["Influenza"]
    df_h = s.reset_index()
    df_h.columns = ["Fecha", "Casos"]
    df_h["Anio"]   = df_h["Fecha"].dt.isocalendar().year.astype(int)
    df_h["Semana"] = df_h["Fecha"].dt.isocalendar().week.astype(int)

    pivot = (
        df_h[df_h["Semana"] <= 52]
        .pivot_table(index="Semana", columns="Anio",
                     values="Casos", aggfunc="sum")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        pivot, cmap="YlOrRd", ax=ax,
        linewidths=0.15, linecolor="#0a0d12",
        cbar_kws={"label": "Atenciones semanales", "shrink": 0.55},
    )
    ax.set_title(
        "Estacionalidad de Influenza — Semana Epidemiológica × Año",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO,
    )
    ax.set_xlabel("Año", fontsize=11, color=TEXTO, labelpad=6)
    ax.set_ylabel("Semana Epidemiológica", fontsize=11, color=TEXTO, labelpad=6)
    ax.tick_params(colors=TEXTO, labelsize=8)
    ax.collections[0].colorbar.ax.yaxis.label.set_color(TEXTO)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXTO)
    # Marcar columnas COVID
    for i, col in enumerate(pivot.columns):
        if col in ANIOS_COVID:
            ax.add_patch(plt.Rectangle(
                (i, 0), 1, len(pivot),
                fill=False, edgecolor=ROJO, lw=2.5, clip_on=False,
            ))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(PANEL)
    fig.tight_layout()
    return fig


# ── Gráfico 4: Distribución etaria (área apilada) ───────────────────────────

def fig_etaria(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["Causa"] == CAUSA_TOTAL]
    cols_edad = {
        "< 1 año":    "NumMenor1Anio",
        "1–4 años":   "Num1a4Anios",
        "5–14 años":  "Num5a14Anios",
        "15–64 años": "Num15a64Anios",
        "≥ 65 años":  "Num65oMas",
    }

    # Construir serie semanal por grupo de edad
    agg = (
        sub.groupby("Fecha")[list(cols_edad.values()) + ["NumTotal"]]
        .sum()
        .sort_index()
    )
    for label, col in cols_edad.items():
        agg[label] = 100 * agg[col] / agg["NumTotal"].replace(0, np.nan)

    colores_edad = [AZUL, VERDE, "#e9c46a", "#f4a261", MORADO]
    labels       = list(cols_edad.keys())

    # Suavizar con media móvil de 4 semanas
    smooth = agg[labels].rolling(4, center=True).mean()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.stackplot(
        smooth.index,
        [smooth[l].fillna(0) for l in labels],
        labels=labels,
        colors=colores_edad,
        alpha=0.82,
    )

    # Línea COVID
    for anio in ANIOS_COVID:
        ax.axvspan(pd.Timestamp(f"{anio}-01-01"),
                   pd.Timestamp(f"{anio}-12-31"),
                   alpha=0.08, color=ROJO)

    ax.set_ylabel("% del total de atenciones", fontsize=11, labelpad=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Distribución Etaria de Urgencias Respiratorias — 2014–2026",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO,
    )
    ax.legend(loc="lower right", fontsize=9, ncol=5)
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


# ── Gráfico 5: Por tipo de establecimiento ───────────────────────────────────

def fig_tipo_establecimiento(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["Causa"] == CAUSA_TOTAL].dropna(subset=["TipoUrgencia"])

    mapa_tipo = {
        "Urgencia Hospitalaria (UEH)":          "UEH (Hospital)",
        "Urgencia Ambulatoria (SAPU)":          "SAPU",
        "Urgencia ambulatoria (SAR)":           "SAR",
        "Urgencia Ambulatoria (SAR)":           "SAR",
        "Urgencia Ambulatoria (SUR)":           "SUR",
    }
    sub = sub.copy()
    sub["Tipo"] = sub["TipoUrgencia"].map(mapa_tipo)
    sub = sub.dropna(subset=["Tipo"])

    anual = (
        sub.groupby(["Anio", "Tipo"])["NumTotal"]
        .sum()
        .reset_index()
    )

    colores_tipo = {
        "UEH (Hospital)": AZUL,
        "SAPU":           VERDE,
        "SAR":            "#e9c46a",
        "SUR":            MORADO,
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for tipo, color in colores_tipo.items():
        d = anual[anual["Tipo"] == tipo]
        if d.empty:
            continue
        ax.plot(d["Anio"], d["NumTotal"] / 1e6,
                marker="o", markersize=5, color=color,
                linewidth=2, label=tipo)
        ax.fill_between(d["Anio"], d["NumTotal"] / 1e6, alpha=0.10, color=color)

    for anio in ANIOS_COVID:
        ax.axvspan(anio - 0.4, anio + 0.4, alpha=0.12, color=ROJO)

    ax.set_xticks(anual["Anio"].unique())
    ax.set_xticklabels(anual["Anio"].unique().astype(str), rotation=45)
    ax.set_ylabel("Atenciones (millones)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M"))
    ax.set_title(
        "Urgencias Respiratorias por Tipo de Establecimiento — 2014–2026",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


# ── Gráfico 6: Descomposición STL ────────────────────────────────────────────

def fig_stl(series: dict) -> plt.Figure:
    s     = series["Influenza"]
    s_stl = s[s.index.year <= 2025]
    res   = STL(s_stl, period=52, robust=True).fit()

    componentes = {
        "Serie original":  s_stl,
        "Tendencia":       res.trend,
        "Estacionalidad":  res.seasonal,
        "Residuo":         res.resid,
    }
    colores_stl = [AZUL, VERDE, "#e9c46a", ROJO]

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    for ax, (label, data), color in zip(axes, componentes.items(), colores_stl):
        ax.plot(data.index, data.values, color=color, linewidth=0.9)
        if label == "Estacionalidad":
            ax.fill_between(data.index, data.values, alpha=0.25, color=color)
        for anio in ANIOS_COVID:
            ax.axvspan(pd.Timestamp(f"{anio}-01-01"),
                       pd.Timestamp(f"{anio}-12-31"),
                       alpha=0.08, color=ROJO)
        ax.set_ylabel(label, fontsize=9.5, color=color,
                      fontweight="bold", labelpad=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))
        ax.grid(axis="y")

    axes[0].set_title(
        "Descomposición STL — Influenza Semanal Nacional (2014–2025)",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO,
    )
    axes[-1].set_xlabel("Semana Epidemiológica", fontsize=11, color=TEXTO)
    fig.tight_layout()
    return fig


# ── Gráfico 7: SARIMA — ajuste y predicción ──────────────────────────────────

def fig_sarima(series: dict) -> tuple:
    s_m   = serie_mensual(series["Influenza"])
    train = s_m[(s_m.index.year >= 2014) &
                (s_m.index.year <= 2024) &
                (~s_m.index.year.isin(ANIOS_COVID))]
    test  = s_m[s_m.index.year == 2025]

    print(f"\n  SARIMA — train: {len(train)} meses | test: {len(test)} meses")

    modelo = SARIMAX(train, order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)

    print(f"  AIC: {modelo.aic:.1f}  BIC: {modelo.bic:.1f}")

    h_total  = len(test) + 7          # 2025 (12 m) + resto de 2026 (~7 m)
    forecast = modelo.get_forecast(steps=h_total)
    f_idx    = pd.date_range(test.index[0], periods=h_total, freq="MS")
    f_mean   = pd.Series(forecast.predicted_mean.clip(lower=0).values, index=f_idx)
    f_ci     = forecast.conf_int(alpha=0.05)
    f_ci.index = f_idx

    # ── Panel completo: histórico + predicción ──
    fig, ax = plt.subplots(figsize=(16, 6))

    # Histórico completo (fondo tenue)
    ax.plot(s_m.index, s_m.values, color=TEXTO, linewidth=0.7,
            alpha=0.25, label="Histórico")

    # Banda COVID en el histórico
    for anio in ANIOS_COVID:
        ax.axvspan(pd.Timestamp(f"{anio}-01-01"),
                   pd.Timestamp(f"{anio}-12-31"),
                   alpha=0.10, color=ROJO)

    # Datos reales 2025
    ax.plot(test.index, test.values, color=AZUL, linewidth=2.2,
            label="Real 2025")

    # Predicción
    ax.plot(f_idx, f_mean.values, color=VERDE, linewidth=2.2,
            linestyle="--", label="SARIMA predicción")
    ax.fill_between(f_idx,
                    f_ci.iloc[:, 0].clip(lower=0),
                    f_ci.iloc[:, 1],
                    alpha=0.15, color=VERDE, label="IC 95%")

    # Divisor real / proyección
    ax.axvline(test.index[-1], color=TEXTO, linewidth=0.8,
               linestyle=":", alpha=0.4)
    ax.text(test.index[-1] + pd.DateOffset(weeks=3),
            f_mean.max() * 0.92, "→ Proyección 2026",
            color=VERDE, fontsize=9)

    ax.set_title(
        "Influenza — Modelo SARIMA: Ajuste, Validación 2025 y Proyección 2026",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO,
    )
    ax.set_ylabel("Atenciones mensuales", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))
    ax.legend(fontsize=10)
    ax.grid(axis="y")
    fig.tight_layout()

    # Exportar predicción
    df_fc = pd.DataFrame({
        "Fecha":      f_idx,
        "Prediccion": f_mean.values.clip(min=0).astype(int),
        "IC_inf":     f_ci.iloc[:, 0].clip(lower=0).astype(int).values,
        "IC_sup":     f_ci.iloc[:, 1].astype(int).values,
    })

    # Métricas en 2025
    comun = test.index.intersection(f_idx)
    if len(comun):
        real = test.loc[comun].values
        pred = f_mean.loc[comun].values
        mae  = np.mean(np.abs(real - pred))
        mape = np.mean(np.abs((real - pred) / np.clip(real, 1, None))) * 100
        print(f"  MAE 2025: {mae:,.0f} | MAPE: {mape:.1f}%")

    return fig, df_fc


# =============================================================================
# 5. EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> tuple:
    print("=" * 65)
    print("  URGENCIAS RESPIRATORIAS CHILE 2014–2026")
    print("=" * 65)

    df = cargar()
    print()

    # Preparar series semanales
    series = {k: serie_semanal(df, k) for k in CAUSAS}

    print("Generando gráficos estáticos...")
    _guardar(fig_tendencia_anual(df),         "01_tendencia_anual")
    _guardar(fig_series_semanales(series),    "02_series_semanales")
    _guardar(fig_heatmap(series),             "03_heatmap_estacional")
    _guardar(fig_etaria(df),                  "04_distribucion_etaria")
    _guardar(fig_tipo_establecimiento(df),    "05_tipo_establecimiento")
    _guardar(fig_stl(series),                 "06_descomposicion_stl")

    print("\nAjustando SARIMA...")
    fig_pred, df_forecast = fig_sarima(series)
    _guardar(fig_pred, "07_sarima_prediccion")

    # Exportar datos para animaciones
    datos_dir = Path("outputs/datos")
    datos_dir.mkdir(exist_ok=True)

    # Series mensuales de todas las causas
    mensuales = pd.concat(
        {k: serie_mensual(v).rename("Casos") for k, v in series.items()},
        names=["Causa", "Fecha"]
    ).reset_index()
    mensuales.to_csv(datos_dir / "series_mensuales.csv", index=False)

    # Series semanales (Influenza para animación)
    series["Influenza"].rename("Casos").reset_index().to_csv(
        datos_dir / "influenza_semanal.csv", index=False
    )

    df_forecast.to_csv(datos_dir / "prediccion_2026.csv", index=False)

    print(f"\n  Todo guardado en: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    return series, df_forecast


if __name__ == "__main__":
    main()
