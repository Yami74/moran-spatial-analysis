# =============================================================================
# modules/utils.py
# Utilidades generales del sistema de análisis espacial Moran
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# CONSTANTES DE DISEÑO Y CONFIGURACIÓN
# -----------------------------------------------------------------------------

COLORS = {
    "primary":    "#1B4F72",   # Azul institucional
    "secondary":  "#2E86AB",   # Azul medio
    "accent":     "#F18F01",   # Naranja acento
    "success":    "#27AE60",   # Verde éxito
    "warning":    "#E67E22",   # Naranja advertencia
    "danger":     "#C0392B",   # Rojo peligro
    "neutral":    "#7F8C8D",   # Gris neutro
    "bg_dark":    "#0F1923",   # Fondo oscuro
    "bg_card":    "#1A2634",   # Fondo tarjeta
    "text_light": "#ECF0F1",   # Texto claro
}

# Paleta de colores para clusters LISA
LISA_COLORS = {
    "HH": "#D73027",   # High-High → Rojo intenso
    "LL": "#4575B4",   # Low-Low  → Azul intenso
    "HL": "#FC8D59",   # High-Low → Naranja
    "LH": "#91BFDB",   # Low-High → Azul claro
    "NS": "#EEEEEE",   # No significativo → Gris
}

# Nombres descriptivos de clusters
LISA_LABELS = {
    "HH": "Alto-Alto (Hot Spot)",
    "LL": "Bajo-Bajo (Cold Spot)",
    "HL": "Alto-Bajo (Outlier)",
    "LH": "Bajo-Alto (Outlier)",
    "NS": "No significativo",
}


# -----------------------------------------------------------------------------
# FUNCIONES DE VALIDACIÓN
# -----------------------------------------------------------------------------

def validar_columna_numerica(df: pd.DataFrame, columna: str) -> bool:
    """
    Verifica que una columna sea numérica y tenga suficientes datos válidos.
    
    Parámetros:
        df: DataFrame con los datos
        columna: Nombre de la columna a validar
    
    Retorna:
        True si la columna es válida para análisis
    """
    if columna not in df.columns:
        return False
    
    serie = pd.to_numeric(df[columna], errors='coerce')
    n_validos = serie.notna().sum()
    
    if n_validos < 10:
        st.warning(f"⚠️ La columna '{columna}' tiene solo {n_validos} valores válidos. Se recomiendan al menos 10.")
        return n_validos >= 4  # Mínimo absoluto
    
    return True


def detectar_columnas_geograficas(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detecta automáticamente columnas de latitud, longitud y geometría.
    
    Estrategia:
        1. Busca por nombres comunes (case-insensitive)
        2. Verifica rangos numéricos típicos de coordenadas
    
    Retorna:
        Diccionario con keys 'lat', 'lon', 'geometry'
    """
    resultado = {"lat": None, "lon": None, "geometry": None}
    
    # Patrones para latitud
    patrones_lat = ["lat", "latitude", "latitud", "y", "coord_y", "ycoord", "northing"]
    # Patrones para longitud
    patrones_lon = ["lon", "long", "longitude", "longitud", "x", "coord_x", "xcoord", "easting"]
    # Patrones para geometría
    patrones_geo = ["geometry", "geom", "shape", "the_geom", "wkt"]
    
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Buscar latitud
    for patron in patrones_lat:
        if patron in cols_lower:
            col_real = cols_lower[patron]
            serie = pd.to_numeric(df[col_real], errors='coerce')
            # Verificar rango válido de latitud (-90 a 90)
            if serie.notna().any() and serie.dropna().between(-90, 90).all():
                resultado["lat"] = col_real
                break
    
    # Buscar longitud
    for patron in patrones_lon:
        if patron in cols_lower:
            col_real = cols_lower[patron]
            serie = pd.to_numeric(df[col_real], errors='coerce')
            # Verificar rango válido de longitud (-180 a 180)
            if serie.notna().any() and serie.dropna().between(-180, 180).all():
                resultado["lon"] = col_real
                break
    
    # Buscar geometría
    for patron in patrones_geo:
        if patron in cols_lower:
            resultado["geometry"] = cols_lower[patron]
            break
    
    return resultado


def detectar_columnas_numericas(df: pd.DataFrame) -> List[str]:
    """
    Retorna lista de columnas numéricas con suficientes datos válidos.
    Excluye columnas que parecen IDs o coordenadas puras.
    """
    excluir_patrones = ["id", "fid", "objectid", "gid", "codigo", "code", 
                         "lat", "lon", "longitude", "latitude", "x", "y"]
    
    numericas = []
    for col in df.columns:
        if col.lower() in excluir_patrones:
            continue
        try:
            serie = pd.to_numeric(df[col], errors='coerce')
            if serie.notna().sum() >= 4 and serie.nunique() > 2:
                numericas.append(col)
        except Exception:
            continue
    
    return numericas


# -----------------------------------------------------------------------------
# FUNCIONES DE ESTADÍSTICAS DESCRIPTIVAS
# -----------------------------------------------------------------------------

def calcular_estadisticas(serie: pd.Series) -> Dict[str, float]:
    """
    Calcula estadísticas descriptivas completas para una serie numérica.
    
    Retorna:
        Diccionario con todas las métricas calculadas
    """
    serie_limpia = pd.to_numeric(serie, errors='coerce').dropna()
    
    return {
        "n":               len(serie_limpia),
        "media":           float(serie_limpia.mean()),
        "mediana":         float(serie_limpia.median()),
        "moda":            float(serie_limpia.mode().iloc[0]) if not serie_limpia.mode().empty else np.nan,
        "desv_std":        float(serie_limpia.std()),
        "varianza":        float(serie_limpia.var()),
        "minimo":          float(serie_limpia.min()),
        "maximo":          float(serie_limpia.max()),
        "rango":           float(serie_limpia.max() - serie_limpia.min()),
        "q1":              float(serie_limpia.quantile(0.25)),
        "q3":              float(serie_limpia.quantile(0.75)),
        "iqr":             float(serie_limpia.quantile(0.75) - serie_limpia.quantile(0.25)),
        "asimetria":       float(serie_limpia.skew()),
        "curtosis":        float(serie_limpia.kurtosis()),
        "cv":              float(serie_limpia.std() / serie_limpia.mean() * 100) if serie_limpia.mean() != 0 else np.nan,
        "nulos":           int(serie.isna().sum()),
    }


def formatear_numero(valor: float, decimales: int = 4) -> str:
    """Formatea un número para presentación, manejando NaN e infinitos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "N/A"
    if isinstance(valor, float) and np.isinf(valor):
        return "∞"
    return f"{valor:,.{decimales}f}"


# -----------------------------------------------------------------------------
# FUNCIONES DE INTERPRETACIÓN
# -----------------------------------------------------------------------------

def interpretar_moran(indice: float, p_valor: float, z_score: float, 
                       alpha: float = 0.05) -> Dict[str, str]:
    """
    Interpreta los resultados del Índice de Moran Global.
    
    Parámetros:
        indice:  Valor del Índice de Moran I
        p_valor: Valor p de la prueba de significancia
        z_score: Puntuación Z estandarizada
        alpha:   Nivel de significancia (por defecto 0.05)
    
    Retorna:
        Diccionario con interpretación, tipo de patrón, color e ícono
    """
    significativo = p_valor < alpha
    
    if not significativo:
        return {
            "patron":          "Aleatoriedad espacial",
            "descripcion":     f"No existe evidencia estadística de autocorrelación espacial (p={p_valor:.4f} > α={alpha}). La distribución es aleatoria.",
            "tipo":            "aleatorio",
            "color":           COLORS["neutral"],
            "icono":           "🎲",
            "recomendacion":   "Considere aumentar el tamaño de muestra o revisar la especificación de la matriz de pesos.",
        }
    
    if indice > 0:
        intensidad = "fuerte" if abs(indice) > 0.3 else "moderada" if abs(indice) > 0.1 else "débil"
        return {
            "patron":          f"Agrupamiento espacial ({intensidad})",
            "descripcion":     f"Existe autocorrelación espacial POSITIVA {intensidad} (I={indice:.4f}, p={p_valor:.4f}). Las unidades similares tienden a estar geográficamente cerca.",
            "tipo":            "agrupamiento",
            "color":           COLORS["danger"],
            "icono":           "🔴",
            "recomendacion":   "Identifique los clusters Hot-Spot (HH) y Cold-Spot (LL) con el análisis LISA.",
        }
    else:
        intensidad = "fuerte" if abs(indice) > 0.3 else "moderada" if abs(indice) > 0.1 else "débil"
        return {
            "patron":          f"Dispersión espacial ({intensidad})",
            "descripcion":     f"Existe autocorrelación espacial NEGATIVA {intensidad} (I={indice:.4f}, p={p_valor:.4f}). Las unidades similares tienden a estar alejadas entre sí.",
            "tipo":            "dispersion",
            "color":           COLORS["secondary"],
            "icono":           "🔵",
            "recomendacion":   "Examine los outliers espaciales (HL y LH) con el análisis LISA.",
        }


def nivel_significancia_texto(p_valor: float) -> str:
    """Retorna texto con nivel de significancia convencional."""
    if p_valor < 0.001:
        return "*** (p < 0.001)"
    elif p_valor < 0.01:
        return "** (p < 0.01)"
    elif p_valor < 0.05:
        return "* (p < 0.05)"
    elif p_valor < 0.1:
        return ". (p < 0.10)"
    else:
        return "ns (no significativo)"


# -----------------------------------------------------------------------------
# FUNCIONES DE SESIÓN Y CACHE
# -----------------------------------------------------------------------------

def inicializar_sesion():
    """Inicializa variables de sesión de Streamlit si no existen."""
    defaults = {
        "datasets":          {},      # {nombre: GeoDataFrame}
        "resultados_moran":  {},      # {nombre: dict con resultados}
        "dataset_activo":    None,    # Nombre del dataset actualmente analizado
        "comparacion":       [],      # Lista de nombres para comparar
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def guardar_dataset(nombre: str, gdf) -> None:
    """Guarda un GeoDataFrame en la sesión de Streamlit."""
    st.session_state["datasets"][nombre] = gdf


def guardar_resultado(nombre: str, resultado: Dict[str, Any]) -> None:
    """Guarda resultados del análisis Moran en la sesión."""
    st.session_state["resultados_moran"][nombre] = resultado


def obtener_datasets() -> Dict:
    """Retorna todos los datasets cargados en la sesión."""
    return st.session_state.get("datasets", {})


def obtener_resultados() -> Dict:
    """Retorna todos los resultados calculados en la sesión."""
    return st.session_state.get("resultados_moran", {})
