# Urgencias Respiratorias en Chile — Análisis 2014–2026

**Serie temporal de 12 años · Estacionalidad · SARIMA · Animaciones**

---

## Descripción

Este proyecto analiza las **Atenciones de Urgencia Respiratoria** registradas semanalmente en Chile entre 2014 y 2026, a partir de los datos abiertos del **MINSAL** (Ministerio de Salud).

El dataset cubre más de **3.5 millones de registros** de establecimientos SAPU, UEH, SAR y SUR de todo el país, desagregados por causa (CIE-10), grupo etario y región.

El análisis abarca:
- Tendencia de largo plazo y el quiebre histórico de la pandemia COVID-19
- Patrones de estacionalidad respiratoria (pico invernal, semanas 20–30)
- Descomposición STL de la serie de Influenza
- Modelo SARIMA calibrado sobre 2014–2024 y proyección hasta 2026
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
| `07_sarima_prediccion.png` | SARIMA: ajuste histórico, validación 2025 y proyección 2026 |

### Animaciones (`outputs/animaciones/`)

| Archivo | Contenido |
|---|---|
| `anim_01_barras_race.gif` | Bar chart race: causas compitiendo año a año (2014–2026) |
| `anim_02_serie_historica.gif` | Línea que se dibuja semana a semana + predicción 2026 |
| `anim_03_estacional.gif` | Patrón estacional de cada año revelado progresivamente |
| `anim_04_multiserie.gif` | Todas las causas dibujándose en paralelo 2014–2025 |

---

## Modelo SARIMA

Entrenado sobre Influenza mensual agregada, excluyendo los años pandémicos (2020–2021):

```
SARIMA(1,1,1)(1,1,1)[12]
Training: 2014–2019 + 2022–2024  →  108 meses
AIC: 1616.7 | BIC: 1628.7
MAE 2025: 3,649 | MAPE: 42.6%
```

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/FelipeMunoz01/GRD-Analisis-Respiratorio-Python.git
cd GRD-Analisis-Respiratorio-Python
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

### 4. Ejecutar el análisis principal

```bash
python analisis_urgencias_respiratorias.py
```

Genera los 7 gráficos estáticos y exporta los datos intermedios a `outputs/datos/`.

### 5. Generar las animaciones

```bash
python animaciones_urgencias.py
```

Requiere haber ejecutado el paso anterior. Genera los 4 GIFs animados.

---

## Estructura del Proyecto

```
GRD-Analisis-Respiratorio-Python/
├── analisis_urgencias_respiratorias.py   # Análisis principal (7 gráficos + SARIMA)
├── animaciones_urgencias.py              # 4 animaciones GIF
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
- `statsmodels` — Descomposición STL y modelo SARIMA

---

## Hallazgos Principales

- Las atenciones respiratorias cayeron **de 5M a 1.2M anuales en 2020** por el confinamiento.
- La recuperación post-pandemia superó los niveles históricos: **5.5M en 2023 y 2024**.
- La **Influenza** muestra el pico estacional más marcado (semanas 20–26, mayo–junio).
- En 2020–2021 la estacionalidad de Influenza desapareció completamente.
- Los menores de 5 años y los mayores de 65 concentran la mayor proporción de atenciones.

---

## Fuente de Datos

- **MINSAL Chile** — Sistema de Información de Urgencias (SIU)
- **Acceso**: Portal de Datos Abiertos del Gobierno de Chile (Ley N° 20.285)
- **Codificación diagnóstica**: CIE-10 (10ª revisión)

---

## Autor

**Felipe Muñoz**  
Análisis de datos en salud pública — Chile
