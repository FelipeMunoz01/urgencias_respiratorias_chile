"""
================================================================================
Animaciones Profesionales — Urgencias Respiratorias Chile 2014–2026
================================================================================

Requiere haber ejecutado primero:
    python analisis_urgencias_respiratorias.py

Genera 4 animaciones GIF con transiciones suaves:
    anim_01_barras_race.gif      — Bar chart race anual por causa
    anim_02_serie_historica.gif  — Línea que se dibuja semana a semana 2014→2026
    anim_03_estacional.gif       — Superposición de patrones anuales revelados año a año
    anim_04_multiserie.gif       — Todas las causas dibujándose simultáneamente
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATOS_DIR  = Path("outputs/datos")
OUTPUT_DIR = Path("outputs/animaciones")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def _base_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXTO, labelsize=9)
    ax.xaxis.label.set_color(TEXTO)
    ax.yaxis.label.set_color(TEXTO)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)


def _guardar(anim_obj, nombre: str, fps: int) -> None:
    ruta = OUTPUT_DIR / nombre
    anim_obj.save(
        ruta,
        writer="pillow",
        fps=fps,
        dpi=140,
        savefig_kwargs={"facecolor": FONDO},
    )
    plt.close("all")
    print(f"  ✓ {ruta.name}")


def _easing(t: float) -> float:
    """Cubic ease-in-out: suaviza el inicio y el final de cada transición."""
    return t * t * (3 - 2 * t)


def _interp(v0: dict, v1: dict, alpha: float) -> dict:
    """Interpolación lineal entre dos diccionarios de valores."""
    keys = set(v0) | set(v1)
    return {k: v0.get(k, 0) * (1 - alpha) + v1.get(k, 0) * alpha for k in keys}


# =============================================================================
# ANIMACIÓN 1 — BAR CHART RACE ANUAL
# Bar horizontales compitiendo año por año.
# Transición suave con easing cúbico entre cada año.
# =============================================================================

def anim_barras_race() -> None:
    mensuales = pd.read_csv(DATOS_DIR / "series_mensuales.csv",
                            parse_dates=["Fecha"])

    causas_orden = list(COLORES.keys())

    # Totales anuales por causa
    mensuales["Anio"] = mensuales["Fecha"].dt.year
    anuales = (
        mensuales.groupby(["Anio", "Causa"])["Casos"]
        .sum()
        .unstack("Causa")
        .fillna(0)
        .loc[range(2014, 2027)]
        [causas_orden]
    )
    anios = anuales.index.tolist()

    N_INTERP = 20   # frames de transición entre años
    N_PAUSA  = 10   # frames de pausa en cada año

    # Construir lista de frames: (valores_dict, año_display, label_estado)
    frames: list[tuple[dict, int, str]] = []

    for i, anio in enumerate(anios):
        vals_curr = anuales.loc[anio].to_dict()

        # Pausa en el año actual
        for _ in range(N_PAUSA):
            frames.append((vals_curr, anio, str(anio)))

        # Transición al siguiente año
        if i < len(anios) - 1:
            vals_next = anuales.loc[anios[i + 1]].to_dict()
            for t in range(N_INTERP):
                alpha    = _easing((t + 1) / N_INTERP)
                interp   = _interp(vals_curr, vals_next, alpha)
                disp_año = anios[i + 1] if alpha >= 0.5 else anio
                frames.append((interp, disp_año, f"{anio} → {anios[i+1]}"))

    max_global = anuales.values.max()

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FONDO)
    _base_ax(ax)

    año_text = ax.text(
        0.97, 0.08, "", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=52,
        fontweight="bold", color=TEXTO, alpha=0.18,
    )
    titulo = ax.set_title(
        "Urgencias Respiratorias por Causa — Chile",
        fontsize=14, fontweight="bold", color=TEXTO, pad=14,
    )
    ax.set_xlabel("Atenciones anuales", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"
    ))

    def draw(fi):
        ax.clear()
        _base_ax(ax)
        ax.set_xlabel("Atenciones anuales", fontsize=11, labelpad=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"
        ))

        vals, anio_disp, _ = frames[fi]

        # Ordenar dinámicamente de menor a mayor
        orden   = sorted(causas_orden, key=lambda c: vals.get(c, 0))
        valores = [vals.get(c, 0) for c in orden]
        colores = [COLORES[c] for c in orden]

        bars = ax.barh(
            range(len(orden)), valores,
            color=colores, alpha=0.82,
            edgecolor=FONDO, linewidth=0.6, height=0.65,
        )

        for j, (causa, val) in enumerate(zip(orden, valores)):
            # Etiqueta a la izquierda
            ax.text(-max_global * 0.01, j, causa,
                    ha="right", va="center", color=TEXTO,
                    fontsize=10, fontweight="bold")
            # Valor a la derecha
            if val > max_global * 0.01:
                ax.text(val + max_global * 0.01, j,
                        f"{int(val):,}",
                        ha="left", va="center", color=TEXTO, fontsize=9)

        ax.set_xlim(-max_global * 0.28, max_global * 1.20)
        ax.set_ylim(-0.65, len(causas_orden) - 0.35)
        ax.set_yticks([])

        # Año en marca de agua
        ax.text(0.97, 0.05, str(anio_disp),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=52, fontweight="bold",
                color=ROJO if anio_disp in ANIOS_COVID else TEXTO,
                alpha=0.18)

        ax.set_title(
            "Urgencias Respiratorias por Causa — Chile",
            fontsize=14, fontweight="bold", color=TEXTO, pad=14,
        )
        fig.patch.set_facecolor(FONDO)

    anim_obj = animation.FuncAnimation(
        fig, draw, frames=len(frames), interval=50, repeat=True
    )
    _guardar(anim_obj, "anim_01_barras_race.gif", fps=20)


# =============================================================================
# ANIMACIÓN 2 — SERIE HISTÓRICA DE INFLUENZA (dibujo semana a semana)
# La línea crece desde 2014 hasta la última semana disponible de 2026,
# luego aparece la predicción punteada en verde.
# =============================================================================

def anim_serie_historica() -> None:
    s_inf  = pd.read_csv(DATOS_DIR / "influenza_semanal.csv",
                         parse_dates=["Fecha"]).set_index("Fecha")["Casos"]
    df_fc  = pd.read_csv(DATOS_DIR / "prediccion_2026.csv",
                         parse_dates=["Fecha"])

    hist = s_inf[s_inf.index.year <= 2025]
    pred = df_fc[df_fc["Fecha"].dt.year == 2026].set_index("Fecha")

    PASO    = 3    # semanas por frame mientras dibuja
    N_PAUSA = 30   # frames de pausa al llegar al final del histórico

    n_hist_frames = (len(hist) + PASO - 1) // PASO
    total_frames  = n_hist_frames + N_PAUSA + len(pred)

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(FONDO)
    _base_ax(ax)

    ax.set_xlim(hist.index[0], pd.Timestamp("2026-12-31"))
    ax.set_ylim(0, hist.max() * 1.22)
    ax.set_xlabel("Semana Epidemiológica", fontsize=11, labelpad=8)
    ax.set_ylabel("Atenciones semanales", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))

    # Banda COVID (fija desde el inicio)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.10, color=ROJO, zorder=1)
    ax.text(pd.Timestamp("2020-07-01"), hist.max() * 1.13,
            "Pandemia COVID-19", color=ROJO,
            fontsize=9, ha="center", fontweight="bold")

    # Línea de predicción pasada (2025 real, tenue)
    ax.plot(hist[hist.index.year == 2025].index,
            hist[hist.index.year == 2025].values,
            color=AZUL, linewidth=0.8, alpha=0, zorder=2)  # empieza invisible

    line_hist, = ax.plot([], [], color=COLORES["Influenza"],
                         linewidth=1.8, zorder=3, label="Real")
    dot_hist,  = ax.plot([], [], "o", color=COLORES["Influenza"],
                         markersize=7, zorder=5)
    line_pred, = ax.plot([], [], color=VERDE, linewidth=2.2,
                         linestyle="--", zorder=4, label="Proyección 2026")
    fill_pred  = [None]   # mutable para actualizar en cada frame

    año_text = ax.text(
        0.02, 0.90, "", transform=ax.transAxes,
        color=TEXTO, fontsize=22, fontweight="bold", alpha=0.3,
    )
    ax.set_title(
        "Atenciones por Influenza — Chile 2014 a 2026",
        fontsize=14, fontweight="bold", color=TEXTO, pad=14,
    )
    ax.legend(fontsize=10, loc="upper right")

    def draw(fi):
        # Limpiar fill anterior
        if fill_pred[0] is not None:
            fill_pred[0].remove()
            fill_pred[0] = None

        if fi < n_hist_frames:
            # Fase 1: dibuja el histórico
            n = min((fi + 1) * PASO, len(hist))
            x = hist.index[:n]
            y = hist.values[:n]
            line_hist.set_data(x, y)
            dot_hist.set_data([x[-1]], [y[-1]])
            año_text.set_text(str(x[-1].year))
            line_pred.set_data([], [])

        elif fi < n_hist_frames + N_PAUSA:
            # Fase 2: pausa con el histórico completo
            line_hist.set_data(hist.index, hist.values)
            dot_hist.set_data([hist.index[-1]], [hist.values[-1]])
            año_text.set_text("2025")
            line_pred.set_data([], [])

        else:
            # Fase 3: dibuja la predicción
            pi = fi - n_hist_frames - N_PAUSA
            n  = min(pi + 1, len(pred))
            xp = pred.index[:n]
            yp = pred["Prediccion"].values[:n]
            ylo= pred["IC_inf"].values[:n]
            yhi= pred["IC_sup"].values[:n]

            line_hist.set_data(hist.index, hist.values)
            line_pred.set_data(xp, yp)
            dot_hist.set_data([xp[-1]], [yp[-1]])
            fill_pred[0] = ax.fill_between(
                xp, ylo, yhi, alpha=0.15, color=VERDE, zorder=2
            )
            año_text.set_text("2026 ↗")

    anim_obj = animation.FuncAnimation(
        fig, draw, frames=total_frames, interval=33, repeat=True, blit=False
    )
    _guardar(anim_obj, "anim_02_serie_historica.gif", fps=20)


# =============================================================================
# ANIMACIÓN 3 — OVERLAY ESTACIONAL (patrón anual revelado año a año)
# Muestra el patrón de cada año (semanas 1–52) superpuesto,
# revelando cómo el COVID rompió la estacionalidad.
# =============================================================================

def anim_estacional() -> None:
    s_inf = pd.read_csv(DATOS_DIR / "influenza_semanal.csv",
                        parse_dates=["Fecha"]).set_index("Fecha")["Casos"]

    anios = sorted(s_inf.index.year.unique())

    # Construir matriz semana × año
    def semanas_año(anio):
        s = s_inf[s_inf.index.year == anio]
        semana = s.index.isocalendar().week.astype(int)
        return s.groupby(semana).sum().reindex(range(1, 53), fill_value=0)

    datos = {a: semanas_año(a) for a in anios}
    max_val = max(d.max() for d in datos.values())

    # Paleta de colores por año (degradado azul→rosa, rojo para COVID)
    cmap   = plt.cm.get_cmap("plasma", len(anios))
    c_anio = {
        a: ROJO if a in ANIOS_COVID else cmap(i)
        for i, a in enumerate(anios)
    }

    N_SEM_POR_FRAME = 2   # semanas que "crece" la línea por frame
    N_PAUSA_AÑO     = 15  # frames de pausa antes de revelar el siguiente año

    # Construir frames: lista de (año_actual, n_semanas_visibles, años_completos)
    frames: list[tuple[int, int, list[int]]] = []
    for i, anio in enumerate(anios):
        for s in range(0, 52, N_SEM_POR_FRAME):
            n = min(s + N_SEM_POR_FRAME, 52)
            frames.append((anio, n, anios[:i]))
        for _ in range(N_PAUSA_AÑO):
            frames.append((anio, 52, anios[:i + 1]))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FONDO)
    _base_ax(ax)

    ax.set_xlim(1, 52)
    ax.set_ylim(0, max_val * 1.2)
    ax.set_xlabel("Semana Epidemiológica", fontsize=11, labelpad=8)
    ax.set_ylabel("Atenciones semanales", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))

    ax.set_title(
        "Patrón Estacional de Influenza — Chile (revelado año a año)",
        fontsize=14, fontweight="bold", color=TEXTO, pad=14,
    )

    # Líneas pre-creadas (una por año, empiezan invisibles)
    lines_plot = {}
    for anio in anios:
        ln, = ax.plot([], [], color=c_anio[anio], linewidth=1.5,
                      alpha=0.75, label=str(anio))
        lines_plot[anio] = ln

    año_text = ax.text(
        0.97, 0.92, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=28,
        fontweight="bold", alpha=0.35, color=TEXTO,
    )

    # Anotación pico invernal
    ax.axvspan(20, 30, alpha=0.07, color=AZUL)
    ax.text(25, max_val * 1.12, "Pico invernal\n(may–jul)",
            ha="center", fontsize=8.5, color=AZUL, alpha=0.6)

    def draw(fi):
        año_actual, n_sem, años_completos = frames[fi]

        # Años completos ya revelados
        for a in años_completos:
            lines_plot[a].set_data(
                range(1, 53), datos[a].values
            )
            lines_plot[a].set_alpha(0.55)

        # Año en curso (dibujándose)
        lines_plot[año_actual].set_data(
            range(1, n_sem + 1), datos[año_actual].values[:n_sem]
        )
        lines_plot[año_actual].set_alpha(1.0)
        lines_plot[año_actual].set_linewidth(2.5)

        # Ocultar años futuros
        for a in anios:
            if a not in años_completos and a != año_actual:
                lines_plot[a].set_data([], [])

        año_text.set_text(str(año_actual))
        año_text.set_color(ROJO if año_actual in ANIOS_COVID else TEXTO)

    anim_obj = animation.FuncAnimation(
        fig, draw, frames=len(frames), interval=45, repeat=True, blit=False
    )
    _guardar(anim_obj, "anim_03_estacional.gif", fps=22)


# =============================================================================
# ANIMACIÓN 4 — MULTISERIE (todas las causas dibujándose en paralelo)
# Fondo oscuro, cada causa en su color, se dibujan simultáneamente
# a través del tiempo 2014→2025.
# =============================================================================

def anim_multiserie() -> None:
    mensuales = pd.read_csv(DATOS_DIR / "series_mensuales.csv",
                            parse_dates=["Fecha"])

    causas_plot = ["IRA Alta", "Influenza", "Neumonía",
                   "Bronquitis", "Crisis Obstructiva", "COVID-19"]

    # Pivot: Fecha × Causa
    pivot = (
        mensuales[mensuales["Causa"].isin(causas_plot)]
        .pivot_table(index="Fecha", columns="Causa", values="Casos", aggfunc="sum")
        .fillna(0)
        .sort_index()
        [causas_plot]
    )

    max_val = pivot.max().max()

    PASO    = 2   # meses por frame
    N_PAUSA = 25

    n_frames = (len(pivot) + PASO - 1) // PASO + N_PAUSA

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(FONDO)
    _base_ax(ax)

    ax.set_xlim(pivot.index[0], pivot.index[-1] + pd.DateOffset(months=6))
    ax.set_ylim(0, max_val * 1.22)
    ax.set_ylabel("Atenciones mensuales", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_miles))
    ax.set_title(
        "Urgencias Respiratorias por Causa — Chile 2014–2025",
        fontsize=14, fontweight="bold", color=TEXTO, pad=14,
    )

    # Banda COVID (siempre visible)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.09, color=ROJO, zorder=1)

    lines_plot: dict[str, plt.Line2D] = {}
    fills_plot: dict[str, object]     = {}
    dots_plot:  dict[str, plt.Line2D] = {}

    for causa in causas_plot:
        color = COLORES[causa]
        ln,  = ax.plot([], [], color=color, linewidth=1.8,
                       alpha=0.92, label=causa, zorder=3)
        dot, = ax.plot([], [], "o", color=color, markersize=6, zorder=5)
        lines_plot[causa] = ln
        dots_plot[causa]  = dot

    ax.legend(fontsize=9, loc="upper left", ncol=3,
              framealpha=0.4, edgecolor=GRID)

    año_text = ax.text(
        0.97, 0.92, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=28,
        fontweight="bold", color=TEXTO, alpha=0.20,
    )

    def draw(fi):
        # Limpiar fills
        for c in list(fills_plot.keys()):
            fills_plot[c].remove()
            del fills_plot[c]

        if fi < n_frames - N_PAUSA:
            n = min((fi + 1) * PASO, len(pivot))
        else:
            n = len(pivot)

        x = pivot.index[:n]

        for causa in causas_plot:
            y = pivot[causa].values[:n]
            lines_plot[causa].set_data(x, y)
            fills_plot[causa] = ax.fill_between(
                x, y, alpha=0.07, color=COLORES[causa], zorder=2
            )
            if len(y):
                dots_plot[causa].set_data([x[-1]], [y[-1]])

        if len(x):
            año_text.set_text(str(x[-1].year))

    anim_obj = animation.FuncAnimation(
        fig, draw, frames=n_frames, interval=50, repeat=True, blit=False
    )
    _guardar(anim_obj, "anim_04_multiserie.gif", fps=20)


# =============================================================================
# HELPER
# =============================================================================

def _fmt_miles(v, _=None):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{int(v):,}"
    return str(int(v))


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    if not (DATOS_DIR / "series_mensuales.csv").exists():
        sys.exit(
            "\nNo se encontraron los datos exportados.\n"
            "Ejecuta primero: python analisis_urgencias_respiratorias.py\n"
        )

    print("=" * 58)
    print("  ANIMACIONES — URGENCIAS RESPIRATORIAS CHILE 2014–2026")
    print("=" * 58)
    print("  Generando animaciones (puede tomar 1–3 minutos)...\n")

    anim_barras_race()
    anim_serie_historica()
    anim_estacional()
    anim_multiserie()

    print(f"\n  Animaciones en: {OUTPUT_DIR.resolve()}")
    print("=" * 58)


if __name__ == "__main__":
    main()
