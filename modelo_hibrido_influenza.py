"""
================================================================================
Modelo Híbrido de Predicción — Influenza Chile 2014–2026
================================================================================

Problema del modelo original:
    SARIMA entrenado con 2014-2019 + 2022-2024 sobreestima 2026 porque
    trata los años post-pandemia (elevados por deuda inmunológica) como
    nivel estructural normal. MAPE original: 42.6%

Solución:
    1. modelo_base   → SARIMA 2014-2019 (nivel estructural puro)
    2. exceso        → desviación real 2022-2025 vs. proyección base
    3. decaimiento   → curva exponencial: ¿rebote transitorio o permanente?
    4. modelo_híb.   → base + exceso_residual (transitorio)
                       o SARIMA con dummy post-COVID (permanente)
    5. validación    → walk-forward CV 2023 / 2024 / 2025
    6. gráfico       → 3 modelos + bandas 80/95% → 07b_modelo_hibrido.png

Autor: Felipe Muñoz
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_PATH  = Path("data/at_urg_respiratorio_semanal.parquet")
OUTPUT_DIR = Path("outputs/figuras")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSA_INF   = "Influenza (J09-J11)"
ANIOS_COVID = {2020, 2021}

FONDO   = "#0d1117"
PANEL   = "#161b22"
TEXTO   = "#c9d1d9"
GRID    = "#21262d"
ROJO    = "#f85149"
VERDE   = "#3fb950"
AZUL    = "#58a6ff"
GRIS    = "#6e7681"
MORADO  = "#a371f7"

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

_SARIMA_KW = dict(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                  enforce_stationarity=False, enforce_invertibility=False)


def _fmt(v, _=None):
    if v >= 1_000_000: return f"{v/1e6:.1f}M"
    if v >= 1_000:     return f"{int(v/1e3)}K"
    return str(int(v))

def _mape(real, pred):
    m = real > 0
    return np.mean(np.abs((real[m] - pred[m]) / real[m])) * 100

def _mae(real, pred):
    return np.mean(np.abs(real - pred))


# =============================================================================
# 1. CARGA Y SERIE MENSUAL
# =============================================================================

def cargar_serie_mensual() -> pd.Series:
    print("Cargando datos...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df["Fecha"] = pd.to_datetime(
        df["Anio"].astype(str) + "-W" +
        df["SemanaEstadistica"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce",
    )
    df = df[~((df["Anio"] == 2026) & (df["SemanaEstadistica"] == 53))]
    s = (df[df["Causa"] == CAUSA_INF]
         .groupby("Fecha")["NumTotal"].sum()
         .resample("MS").sum()
         .sort_index())
    print(f"{len(s)} meses | {s.index[0].date()} → {s.index[-1].date()}")
    return s


# =============================================================================
# 2. MODELO BASE (solo 2014-2019)
# =============================================================================

def entrenar_base(s_m: pd.Series):
    train = s_m[s_m.index.year.isin(range(2014, 2020))]
    m = SARIMAX(train, **_SARIMA_KW).fit(disp=False)
    return m, train

def proyectar_desde(modelo, desde: str, hasta: str,
                    exog=None) -> pd.DataFrame:
    idx = pd.date_range(desde, hasta, freq="MS")
    kw  = {"exog": exog} if exog is not None else {}
    fc  = modelo.get_forecast(steps=len(idx), **kw)
    mean = pd.Series(fc.predicted_mean.values, index=idx).clip(lower=0)
    ci95 = fc.conf_int(alpha=0.05);  ci95.index = idx
    ci80 = fc.conf_int(alpha=0.20);  ci80.index = idx
    return pd.DataFrame({
        "mean":  mean,
        "lo95":  ci95.iloc[:, 0].clip(lower=0),
        "hi95":  ci95.iloc[:, 1],
        "lo80":  ci80.iloc[:, 0].clip(lower=0),
        "hi80":  ci80.iloc[:, 1],
    })


# =============================================================================
# 3. ANÁLISIS DEL REBOTE POST-PANDEMIA
# =============================================================================

def analizar_rebote(s_m: pd.Series, proy_base: pd.DataFrame) -> dict:
    comun = s_m.index.intersection(proy_base.index)
    exceso_pct = (
        (s_m.loc[comun] - proy_base.loc[comun, "mean"])
        / proy_base.loc[comun, "mean"].replace(0, np.nan)
        * 100
    )
    anual = (exceso_pct
             .groupby(exceso_pct.index.year).mean()
             .loc[lambda s: s.index.isin(range(2022, 2026))])

    t = np.arange(len(anual), dtype=float)
    y = anual.values.astype(float)

    tipo = "indeterminado"
    r2   = np.nan
    nivel_2026 = float(np.mean(y)) if len(y) else 0.0

    if len(y) >= 2:
        ss_tot = np.sum((y - y.mean()) ** 2)

        # Decaimiento puro hacia cero
        try:
            popt_d, _ = curve_fit(
                lambda t, A, k: A * np.exp(-k * t),
                t, y, p0=[y[0], 0.5], maxfev=8000)
            res_d  = y - popt_d[0] * np.exp(-popt_d[1] * t)
            r2_d   = 1 - np.sum(res_d**2) / max(ss_tot, 1e-9)
            niv_d  = float(popt_d[0] * np.exp(-popt_d[1] * len(anual)))
        except Exception:
            r2_d, niv_d = -999, y.mean()

        # Decaimiento hacia plateau
        try:
            popt_p, _ = curve_fit(
                lambda t, A, k, C: A * np.exp(-k * t) + C,
                t, y, p0=[y[0] - y[-1], 0.5, y[-1]], maxfev=8000)
            res_p  = y - (popt_p[0] * np.exp(-popt_p[1] * t) + popt_p[2])
            r2_p   = 1 - np.sum(res_p**2) / max(ss_tot, 1e-9)
            niv_p  = float(popt_p[0] * np.exp(-popt_p[1] * len(anual)) + popt_p[2])
        except Exception:
            r2_p, niv_p = -999, y.mean()

        if r2_p > r2_d + 0.05 and r2_p > 0:
            tipo, r2, nivel_2026 = "permanente", r2_p, niv_p
        elif r2_d > 0:
            tipo, r2, nivel_2026 = "transitorio", r2_d, niv_d
        else:
            tipo, r2, nivel_2026 = "transitorio", 0.0, float(y[-1])

    return {
        "exceso_pct_anual":    anual,
        "tipo_rebote":         tipo,
        "r2_curva":            r2,
        "nivel_residual_2026": nivel_2026,
    }


# =============================================================================
# 4. MODELO HÍBRIDO
# =============================================================================

def entrenar_hibrido(s_m: pd.Series, tipo_rebote: str):
    if tipo_rebote == "permanente":
        anios = [a for a in range(2014, 2026) if a not in ANIOS_COVID]
        train = s_m[s_m.index.year.isin(anios)]
        dummy = pd.Series(
            (train.index.year >= 2022).astype(float),
            index=train.index)
        m = SARIMAX(train, exog=dummy, **_SARIMA_KW).fit(disp=False)
        return m, "permanente"
    else:
        train = s_m[s_m.index.year.isin(range(2014, 2020))]
        m = SARIMAX(train, **_SARIMA_KW).fit(disp=False)
        return m, "transitorio"

def proyectar_hibrido(modelo, tipo: str, nivel_residual: float,
                      desde: str, hasta: str) -> pd.DataFrame:
    idx = pd.date_range(desde, hasta, freq="MS")
    if tipo == "permanente":
        exog_fc = pd.Series(np.ones(len(idx)), index=idx)
        df = proyectar_desde(modelo, desde, hasta, exog=exog_fc)
    else:
        df = proyectar_desde(modelo, desde, hasta)
        if nivel_residual > 0:
            factor = 1 + nivel_residual / 100
            for col in df.columns:
                df[col] = (df[col] * factor).clip(lower=0)
    return df


# =============================================================================
# 5. VALIDACIÓN CRUZADA WALK-FORWARD
# =============================================================================

def validacion_cruzada(s_m: pd.Series) -> pd.DataFrame:
    ventanas = [
        (list(range(2014, 2023)), 2023),
        (list(range(2014, 2024)), 2024),
        (list(range(2014, 2025)), 2025),
    ]
    rows = []
    for anios_all, anio_test in ventanas:
        anios_limpios = [a for a in anios_all if a not in ANIOS_COVID]
        train = s_m[s_m.index.year.isin(anios_limpios)]
        test  = s_m[s_m.index.year == anio_test]
        if len(test) == 0:
            continue
        m   = SARIMAX(train, **_SARIMA_KW).fit(disp=False)
        fc  = m.get_forecast(steps=len(test))
        pred = fc.predicted_mean.clip(lower=0).values
        rows.append({
            "Ventana":  f"hasta {max(anios_limpios)} → pred {anio_test}",
            "Año test": anio_test,
            "MAE":      _mae(test.values, pred),
            "MAPE":     _mape(test.values, pred),
        })
        print(f"    CV {anio_test}: MAE {_mae(test.values,pred):,.0f} | "
              f"MAPE {_mape(test.values,pred):.1f}%")
    return pd.DataFrame(rows)


# =============================================================================
# 6. GRÁFICO COMPARATIVO
# =============================================================================

def generar_grafico(s_m, proy_base, proy_orig, proy_hib,
                    mape_orig, mape_hib, info_rebote) -> plt.Figure:

    desde_hist = pd.Timestamp("2017-01-01")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(16, 11),
        gridspec_kw={"height_ratios": [2.5, 1]})

    # ── Panel principal ──────────────────────────────────────────────────────
    # Histórico
    hist_plot = s_m[s_m.index >= desde_hist]
    ax.plot(hist_plot.index, hist_plot.values,
            color=TEXTO, linewidth=0.9, alpha=0.25, label="Histórico real")

    # Banda COVID
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.09, color=ROJO)
    ax.text(pd.Timestamp("2020-08-01"), hist_plot.max() * 1.04,
            "COVID-19", color=ROJO, fontsize=8, ha="center", alpha=0.7)

    # Real post-pandemia
    real_post = s_m[(s_m.index.year >= 2022) & (s_m.index.year <= 2025)]
    ax.plot(real_post.index, real_post.values,
            color=AZUL, linewidth=2.2, label="Real 2022–2025")

    # Línea divisoria
    div = s_m.index[-1]
    ax.axvline(div, color=TEXTO, linewidth=0.8, linestyle=":", alpha=0.35)
    ax.text(div + pd.DateOffset(weeks=2), hist_plot.max() * 0.88,
            "→ Proyección 2026", color=TEXTO, fontsize=8, alpha=0.5)

    # Proyección original (gris punteada)
    orig_2026 = proy_orig[proy_orig.index.year == 2026]
    ax.plot(orig_2026.index, orig_2026["mean"].values,
            color=GRIS, linewidth=1.6, linestyle=":",
            label=f"SARIMA original  MAPE={mape_orig:.1f}%")

    # Proyección base (azul punteada)
    base_2026 = proy_base[proy_base.index.year == 2026]
    ax.plot(base_2026.index, base_2026["mean"].values,
            color=AZUL, linewidth=1.6, linestyle="--",
            label="Modelo base 2014–2019 (nivel estructural)")

    # Proyección híbrida (verde sólida)
    hib_2026 = proy_hib[proy_hib.index.year == 2026]
    ax.fill_between(hib_2026.index,
                    hib_2026["lo95"], hib_2026["hi95"],
                    alpha=0.10, color=VERDE)
    ax.fill_between(hib_2026.index,
                    hib_2026["lo80"], hib_2026["hi80"],
                    alpha=0.20, color=VERDE)
    ax.plot(hib_2026.index, hib_2026["mean"].values,
            color=VERDE, linewidth=2.5,
            label=f"Modelo híbrido (oficial)  MAPE={mape_hib:.1f}%")

    # Anotación peak
    peak_mes = hib_2026["mean"].idxmax()
    peak_val = hib_2026["mean"].max()
    lo80_pk  = hib_2026.loc[peak_mes, "lo80"]
    hi80_pk  = hib_2026.loc[peak_mes, "hi80"]
    ax.annotate(
        f"Peak proyectado\n{int(peak_val):,} atenciones\nIC80: [{int(lo80_pk):,}–{int(hi80_pk):,}]",
        xy=(peak_mes, peak_val),
        xytext=(peak_mes - pd.DateOffset(months=4), peak_val * 1.14),
        arrowprops=dict(arrowstyle="->", color=VERDE, lw=1.2),
        color=VERDE, fontsize=8.5, fontweight="bold",
    )

    # Caja métricas
    mejora = mape_orig - mape_hib
    caja = (f"MAPE original :  {mape_orig:.1f}%\n"
            f"MAPE híbrido  :  {mape_hib:.1f}%\n"
            f"Mejora        :  {mejora:+.1f} pp\n"
            f"Rebote        :  {info_rebote['tipo_rebote']}\n"
            f"R² decaim.    :  {info_rebote['r2_curva']:.3f}")
    ax.text(0.015, 0.97, caja, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            color=TEXTO, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor=PANEL, edgecolor=GRID, alpha=0.92))

    ax.set_xlim(desde_hist, pd.Timestamp("2026-12-01"))
    ax.set_ylabel("Atenciones mensuales (Influenza)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y")
    ax.set_title(
        "Influenza Chile 2014–2026 — Modelo Híbrido vs. SARIMA Original",
        fontsize=14, fontweight="bold", pad=14, color=TEXTO)

    # ── Panel inferior: exceso anual ─────────────────────────────────────────
    exc = info_rebote["exceso_pct_anual"]
    colores_b = [ROJO if v > 0 else AZUL for v in exc.values]
    ax2.bar(exc.index.astype(str), exc.values,
            color=colores_b, alpha=0.82,
            edgecolor=FONDO, linewidth=0.4, width=0.5)
    ax2.axhline(0, color=TEXTO, linewidth=0.6, alpha=0.35)
    niv = info_rebote["nivel_residual_2026"]
    ax2.axhline(niv, color=VERDE, linewidth=1.4, linestyle="--",
                label=f"Exceso proyectado 2026: {niv:.1f}%")

    for xi, (año, val) in enumerate(zip(exc.index, exc.values)):
        ax2.text(xi, val + np.sign(val) * 1.2, f"{val:+.1f}%",
                 ha="center",
                 va="bottom" if val > 0 else "top",
                 color=TEXTO, fontsize=9, fontweight="bold")

    ax2.set_ylabel("Exceso vs. base (%)", fontsize=10, labelpad=8)
    ax2.set_title(
        f"Exceso Post-Pandemia 2022–2025 vs. Proyección Base — "
        f"Rebote {info_rebote['tipo_rebote'].upper()}  (R²={info_rebote['r2_curva']:.3f})",
        fontsize=11, fontweight="bold", pad=8, color=TEXTO)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y")

    fig.tight_layout(h_pad=2.5)
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  MODELO HÍBRIDO — INFLUENZA CHILE 2014–2026")
    print("=" * 65)

    s_m = cargar_serie_mensual()

    # 1. Modelo base
    print("\n[1] Modelo base (2014–2019)...")
    m_base, _ = entrenar_base(s_m)
    print(f"    AIC: {m_base.aic:.1f} | BIC: {m_base.bic:.1f}")
    proy_base = proyectar_desde(m_base, "2020-01-01", "2026-12-01")

    # 2. Rebote
    print("\n[2] Analizando rebote post-pandemia...")
    info = analizar_rebote(s_m, proy_base)
    for anio, v in info["exceso_pct_anual"].items():
        print(f"    {anio}: {v:+.1f}%")
    print(f"    Tipo: {info['tipo_rebote'].upper()}  "
          f"R²={info['r2_curva']:.3f}  "
          f"Exceso 2026≈{info['nivel_residual_2026']:.1f}%")

    # 3. Modelo híbrido
    print(f"\n[3] Modelo híbrido (estrategia: {info['tipo_rebote']})...")
    m_hib, tipo = entrenar_hibrido(s_m, info["tipo_rebote"])
    print(f"    AIC: {m_hib.aic:.1f} | BIC: {m_hib.bic:.1f}")
    proy_hib = proyectar_hibrido(
        m_hib, tipo, info["nivel_residual_2026"],
        "2026-01-01", "2026-12-01")

    # 4. Validación cruzada
    print("\n[4] Validación cruzada walk-forward...")
    df_cv = validacion_cruzada(s_m)

    # MAPE modelos en 2025
    test_2025 = s_m[s_m.index.year == 2025]

    m_orig = SARIMAX(
        s_m[s_m.index.year.isin(
            [a for a in range(2014, 2025) if a not in ANIOS_COVID])],
        **_SARIMA_KW).fit(disp=False)
    mape_orig = _mape(test_2025.values,
                      m_orig.get_forecast(steps=len(test_2025))
                            .predicted_mean.clip(lower=0).values)

    if tipo == "permanente":
        exog_2025 = pd.Series(np.ones(len(test_2025)), index=test_2025.index)
        fc_hib_2025 = m_hib.get_forecast(steps=len(test_2025), exog=exog_2025)
    else:
        fc_hib_2025 = m_hib.get_forecast(steps=len(test_2025))
    mape_hib = _mape(test_2025.values,
                     fc_hib_2025.predicted_mean.clip(lower=0).values)

    proy_orig = proyectar_desde(m_orig, "2025-01-01", "2026-12-01")

    # 5. Gráfico
    print("\n[5] Generando gráfico...")
    fig = generar_grafico(s_m, proy_base, proy_orig, proy_hib,
                          mape_orig, mape_hib, info)
    ruta = OUTPUT_DIR / "07b_modelo_hibrido.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  ✓ {ruta.name}")

    # 6. Reporte
    peak_mes  = proy_hib["mean"].idxmax()
    peak_val  = int(proy_hib["mean"].max())
    lo80      = int(proy_hib.loc[peak_mes, "lo80"])
    hi80      = int(proy_hib.loc[peak_mes, "hi80"])
    mejora    = mape_orig - mape_hib

    print("\n" + "=" * 65)
    print("  REPORTE FINAL")
    print("=" * 65)
    print(f"  MAPE SARIMA original (2025): {mape_orig:.1f}%")
    print(f"  MAPE modelo híbrido  (2025): {mape_hib:.1f}%")
    print(f"  Mejora:                      {mejora:+.1f} pp")
    print()
    print(f"  Rebote post-pandemia: {info['tipo_rebote'].upper()}")
    print(f"  R² curva decaimiento: {info['r2_curva']:.3f}")
    print()
    print(f"  Peak invernal 2026:")
    print(f"    Mes:     {peak_mes.strftime('%B %Y')}")
    print(f"    Central: {peak_val:,} atenciones/mes")
    print(f"    IC 80%:  [{lo80:,} – {hi80:,}]")
    print()

    if mejora >= 10:
        print(f"  ✓ El modelo híbrido mejora {mejora:.1f} pp. Recomendado.")
    else:
        print(f"  ✗ Mejora {mejora:.1f} pp (< 10 pp umbral).")
        print("    Limitación: el pico invernal de Influenza varía ±3-4")
        print("    semanas entre años — variabilidad intrínseca que ningún")
        print("    modelo paramétrico captura con solo 3-4 años post-pandemia.")

    if info["tipo_rebote"] == "transitorio":
        msg = ("El rebote es transitorio: la deuda inmunológica se está"
               " disipando y los niveles convergerán al estructural pre-"
               "pandemia en 2027-2028. Las campañas de vacunación pueden"
               " planificarse con los umbrales históricos 2014-2019.")
    else:
        msg = ("El rebote se ha estabilizado en un nivel permanente."
               " Los umbrales de alerta del sistema de salud deben"
               " actualizarse para reflejar este nuevo baseline post-COVID.")

    print(f"\n  Implicancia salud pública:\n  {msg}")
    print("=" * 65)

    if not df_cv.empty:
        print("\n  Validación cruzada:")
        for _, r in df_cv.iterrows():
            print(f"    {r['Ventana']}: MAPE {r['MAPE']:.1f}%")
    print()


if __name__ == "__main__":
    main()
