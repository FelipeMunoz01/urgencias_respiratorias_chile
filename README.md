# Urgencias Respiratorias en Chile — Análisis 2014–2026

**Serie temporal de 12 años · Estacionalidad · SARIMA · Modelo Híbrido · Animaciones**

---

## Descripción

Este proyecto analiza las **Atenciones de Urgencia Respiratoria** registradas semanalmente en Chile entre 2014 y 2026, a partir de los datos abiertos del **MINSAL** (Ministerio de Salud).

El dataset cubre más de **3.5 millones de registros** de establecimientos SAPU, UEH, SAR y SUR de todo el país, desagregados por causa (CIE-10), grupo etario y región.

El análisis abarca:
- Tendencia de largo plazo y el quiebre histórico de la pandemia COVID-19
- Patrones de estacionalidad respiratoria (pico invernal, semanas 20–30)
- Descomposición STL de la serie de Influenza
- Modelo híbrido SARIMA con separación pre/post pandemia (MAPE 36.1%)
- Correlaciones cruzadas entre causas respiratorias
- Anomalías regionales durante COVID-19
- Efecto vacunación sobre la velocidad de crecimiento del pico invernal
- Recuperación etaria post-pandemia por grupo de edad
- Cambio estructural en distribución SAPU vs Hospital
- Visualizaciones animadas con transiciones profesionales

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | MINSAL / datos.gob.cl |
| URL | https://datos.gob.cl/dataset/606ef5bb-11d1-475b-b69f-b980da5757f4 |
| Formato | Parquet (~66 MB comprimido) |
| Registros | 3,545,524 filas |
| Cobertura temporal | Semanas epidemiológicas 2014 – 2026 |
| Tipos de establecimiento | Hospital (UEH), SAPU, SAR, SUR |

---

## Causas Analizadas (CIE-10)

| Causa | Códigos CIE-10 |
|---|---|
| IRA Alta | J00–J06 |
| Influenza | J09–J11 |
| Neumonía | J12–J18 |
| Bronquitis/Bronquiolitis aguda | J20–J21 |
| Crisis Obstructiva Bronquial | J40–J46 |
| COVID-19 | U07.1 |

---

## Scripts

| Archivo | Contenido |
|---|---|
| `analisis_urgencias_respiratorias.py` | Análisis principal: 7 gráficos estáticos + SARIMA base |
| `analisis_avanzado.py` | 5 análisis profundos: correlaciones, anomalías regionales, vacunación, recuperación etaria, SAPU vs Hospital |
| `modelo_hibrido_influenza.py` | Modelo híbrido SARIMA con nivel estructural pre-pandemia y rebote post-COVID |
| `animaciones_urgencias.py` | 4 animaciones GIF con proyección 2026 corregida |

---

## Visualizaciones Generadas

### Gráficos Estáticos (`outputs/figuras/`)

| Archivo | Contenido |
|---|---|
| `01_tendencia_anual.png` | Totales anuales 2014–2026 con línea de tendencia y quiebre COVID |
| `02_series_semanales.png` | Serie semanal por causa (6 subgráficos) con bandas COVID y peaks |
| `03_heatmap_estacional.png` | Heatmap semana epidemiológica × año (Influenza) |
| `04_distribucion_etaria.png` | Área apilada: evolución de la distribución etaria 2014–2026 |
| `05_tipo_establecimiento.png` | Atenciones anuales por tipo (UEH, SAPU, SAR, SUR) |
| `06_descomposicion_stl.png` | STL: tendencia, estacionalidad y residuo de Influenza |
| `07_sarima_prediccion.png` | SARIMA original: ajuste histórico y proyección 2026 (MAPE 42.6%) |
| `07b_modelo_hibrido.png` | Modelo híbrido: comparación 3 modelos + proyección 2026 corregida (MAPE 36.1%) |
| `08_correlaciones_cruzadas.png` | Correlaciones cruzadas entre causas respiratorias con lag temporal |
| `09_anomalias_regionales.png` | Z-scores por región durante COVID-19 vs histórico 2015–2019 |
| `10_efecto_vacunacion.png` | Slope de crecimiento pre-pico invernal pre y post pandemia |
| `11_recuperacion_etaria.png` | Recuperación por grupo etario respecto al nivel pre-pandemia |
| `12_sapu_vs_hospital.png` | Cambio estructural en distribución por tipo de establecimiento |

### Animaciones (`outputs/animaciones/`)

| Archivo | Contenido |
|---|---|
| `anim_01_barras_race.gif` | Bar chart race: causas compitiendo año a año (2014–2026) |
| `anim_02_serie_historica.gif` | Serie Influenza dibujada semana a semana + proyección 2026 corregida |
| `anim_03_estacional.gif` | Patrón estacional de cada año revelado progresivamente |
| `anim_04_multiserie.gif` | Todas las causas dibujándose en paralelo 2014–2025 |

---

## Modelo Híbrido de Predicción

El modelo SARIMA original (MAPE 42.6%) sobreestimaba los peaks de 2026 porque aprendió la tendencia inflada post-pandemia de 2022–2024 como si fuera normal.

El modelo híbrido resuelve esto separando dos componentes:

- **Nivel estructural**: entrenado solo con 2014–2019 (pre-pandemia pura)
- **Rebote post-COVID**: modelado con variable dummy para años post-pandemia

```
Modelo híbrido SARIMA
Nivel base: 2014–2019
MAPE: 36.1% (mejora de 6.5 puntos sobre SARIMA original)
Rebote post-pandemia: PERMANENTE (R²=0.941)
Exceso 2024: +27.6% | Exceso 2025: +33.8%
Peak proyectado 2026: ~23,675 atenciones mensuales (IC80%: 18,968–28,666)
```

---

## Hallazgos Principales

- Las atenciones respiratorias cayeron **de 5M a 1.2M anuales en 2020** por el confinamiento.
- La recuperación post-pandemia superó los niveles históricos: **5.5M en 2023 y 2024**.
- El **rebote post-COVID es permanente** (R²=0.941): Chile opera con un nivel estructural más alto de Influenza.
- **Bronquitis y Crisis Obstructiva** tienen correlación r=0.96 con lag 0 — se mueven en sincronía perfecta.
- **COVID-19 es la única causa inversamente correlacionada** con todas las demás: desplazó el patrón respiratorio normal.
- **Anomalía Arica**: única región con z-score positivo en semana 5 de 2020, cuando el resto del país caía. Posible circulación viral temprana.
- El **slope de crecimiento pre-pico invernal bajó -45.7%** post-pandemia, posible efecto de mayor cobertura vacunal.
- Los **niños 1–4 años** nunca superaron el 100% del nivel pre-pandemia. Los **adultos 65+** llegaron al 140% en 2024–2025.
- **SAR/SUR pasó de 18.8% a 32.5%** de las atenciones entre 2014 y 2026: cambio estructural silencioso en el sistema de salud.

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/FelipeMunoz01/urgencias_respiratorias_chile.git
cd urgencias_respiratorias_chile
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar los datos

Descarga el archivo Parquet desde el portal oficial:

> **https://datos.gob.cl/dataset/606ef5bb-11d1-475b-b69f-b980da5757f4**

Guárdalo como:

```
data/at_urg_respiratorio_semanal.parquet
```

### 4. Ejecutar en orden

```bash
python analisis_urgencias_respiratorias.py   # Gráficos 01–07
python analisis_avanzado.py                  # Gráficos 08–12
python modelo_hibrido_influenza.py           # Gráfico 07b + proyección corregida
python animaciones_urgencias.py              # 4 GIFs animados
```

---

## Estructura del Proyecto

```
urgencias_respiratorias_chile/
├── analisis_urgencias_respiratorias.py   # Análisis principal
├── analisis_avanzado.py                  # Análisis extendido
├── modelo_hibrido_influenza.py           # Modelo híbrido SARIMA
├── animaciones_urgencias.py              # Animaciones GIF
├── requirements.txt
├── .gitignore
├── README.md
└── outputs/
    ├── figuras/                          # Gráficos estáticos PNG
    ├── animaciones/                      # GIFs animados
    └── datos/                            # CSVs intermedios (ignorados por git)
```

---

## Tecnologías

- **Python 3.10+**
- `pandas` — Manipulación de series temporales
- `numpy` — Cálculos numéricos e interpolación
- `matplotlib` — Gráficos estáticos y animaciones (FuncAnimation + Pillow)
- `seaborn` — Heatmaps y visualizaciones estadísticas
- `statsmodels` — Descomposición STL y modelos SARIMA

---

## Fuente de Datos

- **MINSAL Chile** — Sistema de Información de Urgencias (SIU)
- **Acceso**: Portal de Datos Abiertos del Gobierno de Chile (Ley N° 20.285)
- **Codificación diagnóstica**: CIE-10 (10ª revisión)

---

## Autor

**Felipe Muñoz**  
Análisis de datos en salud pública — Chile
