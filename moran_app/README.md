# 🗺️ Sistema de Análisis de Autocorrelación Espacial — Índice de Moran

Aplicativo web profesional para análisis de **autocorrelación espacial** mediante el 
**Índice de Moran Global y Local (LISA)** usando Python + Streamlit.

---

## 📁 Estructura del Proyecto

```
moran_app/
│
├── app.py                   # Aplicación principal Streamlit
│
├── modules/
│   ├── __init__.py          # Paquete de módulos
│   ├── carga.py             # Carga de CSV, Excel, GeoJSON, Shapefile
│   ├── moran.py             # Cálculo Moran Global + LISA
│   ├── mapas.py             # Folium, Plotly, scatter plots
│   ├── reportes.py          # Exportación Excel, CSV, PDF
│   └── utils.py             # Constantes, helpers, estadísticas
│
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

---

## ⚙️ Instalación Paso a Paso

### Requisitos previos
- Python 3.10 o superior
- pip actualizado: `python -m pip install --upgrade pip`

### 1. Crear y activar entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** En Windows, si geopandas falla, instala primero:
> ```bash
> pip install wheel
> pip install pipwin
> pipwin install gdal
> pipwin install fiona
> pip install geopandas
> ```
> O usa conda: `conda install -c conda-forge geopandas`

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

El navegador se abrirá en `http://localhost:8501`

---

## 🚀 Uso del Sistema

### 1. Cargar Datos
Formatos soportados:
- **CSV**: Debe tener columnas de latitud y longitud (se detectan automáticamente)
- **Excel (.xlsx)**: Igual que CSV, soporta múltiples hojas
- **GeoJSON**: Incluye geometría directamente
- **Shapefile (.shp)**: Subir junto con `.dbf` y `.shx`

### 2. Análisis Moran
En el menú lateral configura:
- **Variable**: Columna numérica a analizar
- **Método de pesos**: KNN (puntos), Queen/Rook (polígonos), Distancia
- **K vecinos**: Solo para KNN (recomendado: 5-8)
- **Permutaciones**: 999 para resultados robustos (9999 para publicación)
- **Nivel α**: 0.05 estándar

### 3. Interpretación
| Índice I | Patrón | Significado |
|----------|--------|-------------|
| I > 0, p < 0.05 | Agrupamiento | Hot/Cold spots presentes |
| I < 0, p < 0.05 | Dispersión | Outliers espaciales |
| p ≥ 0.05 | Aleatorio | Sin patrón espacial |

### Clusters LISA
| Cluster | Color | Significado |
|---------|-------|-------------|
| HH | 🔴 Rojo | Hot Spot: valor alto rodeado de valores altos |
| LL | 🔵 Azul | Cold Spot: valor bajo rodeado de valores bajos |
| HL | 🟠 Naranja | Outlier: valor alto rodeado de valores bajos |
| LH | 🩵 Azul claro | Outlier: valor bajo rodeado de valores altos |
| NS | ⚪ Gris | No significativo estadísticamente |

---

## 📊 Formatos de Exportación

| Formato | Contenido |
|---------|-----------|
| **Excel** | Datos + Moran Global + LISA + Estadísticas (5 hojas) |
| **CSV** | Datos originales + columnas LISA_I, LISA_p, LISA_Cluster |
| **PDF** | Reporte profesional con tablas e interpretación |

---

## 🔬 Fundamentos Estadísticos

### Índice de Moran Global (Moran, 1950)

```
I = (N / S₀) × [Σᵢ Σⱼ wᵢⱼ(xᵢ-x̄)(xⱼ-x̄)] / [Σᵢ(xᵢ-x̄)²]
```
- N = número de unidades
- S₀ = suma de todos los pesos wᵢⱼ
- Valor esperado: E[I] = -1/(N-1)

### Índice de Moran Local (Anselin, 1995)

```
Iᵢ = (xᵢ - x̄) / m₂ × Σⱼ wᵢⱼ(xⱼ - x̄)
```
- m₂ = varianza de la variable
- Inferencia por permutación condicional

### Matrices de Pesos Espaciales
- **KNN**: Garantiza k vecinos por unidad (robusto para puntos)
- **Queen**: wᵢⱼ=1 si comparten al menos un punto (polígonos)
- **Rook**: wᵢⱼ=1 si comparten un borde (polígonos, más estricto)
- **Distancia fija**: wᵢⱼ=1 si dist(i,j) ≤ umbral

---

## 📚 Referencias

- Moran, P.A.P. (1950). *Notes on continuous stochastic phenomena*. Biometrika, 37(1-2).
- Anselin, L. (1995). *Local Indicators of Spatial Association — LISA*. Geographical Analysis, 27(2).
- Rey, S., Anselin, L. et al. PySAL: Python Spatial Analysis Library. https://pysal.org/

---

## 🛠️ Tecnologías

| Librería | Uso |
|----------|-----|
| Streamlit | Interfaz web interactiva |
| GeoPandas | Manejo de datos geoespaciales |
| libpysal | Matrices de pesos espaciales |
| esda | Estadísticos de autocorrelación |
| Folium | Mapas interactivos Leaflet |
| Plotly | Gráficos estadísticos interactivos |
| ReportLab | Generación de reportes PDF |

---

*Proyecto universitario/profesional de Estadística Espacial — v1.0.0*
