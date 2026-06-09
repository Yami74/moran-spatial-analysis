# =============================================================================
# modules/carga.py
# Módulo de carga y procesamiento de datos geográficos
# =============================================================================

import io
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
from typing import Optional, Tuple
from modules.utils import detectar_columnas_geograficas, detectar_columnas_numericas

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DE CARGA
# -----------------------------------------------------------------------------

def cargar_archivo(archivo_subido) -> Optional[gpd.GeoDataFrame]:
    """
    Punto de entrada único para carga de archivos.
    Detecta el tipo de archivo y llama al loader correspondiente.
    
    Parámetros:
        archivo_subido: Objeto UploadedFile de Streamlit
    
    Retorna:
        GeoDataFrame con los datos cargados o None si falla
    """
    nombre = archivo_subido.name.lower()
    
    try:
        if nombre.endswith(".csv"):
            return _cargar_csv(archivo_subido)
        
        elif nombre.endswith((".xlsx", ".xls")):
            return _cargar_excel(archivo_subido)
        
        elif nombre.endswith(".geojson") or nombre.endswith(".json"):
            return _cargar_geojson(archivo_subido)
        
        elif nombre.endswith(".shp"):
            return _cargar_shapefile(archivo_subido)
        
        else:
            st.error(f"❌ Formato no soportado: `{nombre}`\n\nFormatos válidos: CSV, XLSX, GeoJSON, SHP")
            return None
            
    except Exception as e:
        st.error(f"❌ Error al cargar `{archivo_subido.name}`: {str(e)}")
        return None


# -----------------------------------------------------------------------------
# LOADERS ESPECÍFICOS POR FORMATO
# -----------------------------------------------------------------------------

def _cargar_csv(archivo) -> Optional[gpd.GeoDataFrame]:
    """
    Carga un CSV y lo convierte a GeoDataFrame.
    
    Proceso:
        1. Lee el CSV con pandas
        2. Detecta columnas de coordenadas
        3. Crea geometría Point a partir de lat/lon
        4. Proyecta al CRS WGS84 (EPSG:4326)
    """
    # Intentar diferentes encodings comunes
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            archivo.seek(0)
            df = pd.read_csv(archivo, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("❌ No se pudo decodificar el CSV. Guárdelo en UTF-8.")
        return None
    
    if df.empty:
        st.error("❌ El CSV está vacío.")
        return None
    
    # Detectar columnas geográficas automáticamente
    cols_geo = detectar_columnas_geograficas(df)
    
    lat_col = cols_geo.get("lat")
    lon_col = cols_geo.get("lon")
    
    if lat_col and lon_col:
        # Crear geometría a partir de coordenadas
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        # Eliminar filas sin coordenadas válidas
        df_limpio = df.dropna(subset=[lat_col, lon_col]).copy()
        n_eliminados = len(df) - len(df_limpio)
        
        if n_eliminados > 0:
            st.info(f"ℹ️ Se eliminaron {n_eliminados} filas con coordenadas inválidas.")
        
        if df_limpio.empty:
            st.error("❌ No hay filas con coordenadas válidas.")
            return None
        
        # Crear GeoDataFrame con puntos
        geometria = [Point(lon, lat) for lon, lat in 
                     zip(df_limpio[lon_col], df_limpio[lat_col])]
        
        gdf = gpd.GeoDataFrame(df_limpio, geometry=geometria, crs="EPSG:4326")
        
        st.success(f"✅ CSV cargado: **{len(gdf)} filas**, columnas lat=`{lat_col}`, lon=`{lon_col}`")
        return gdf
    
    elif cols_geo.get("geometry"):
        # Si hay columna WKT de geometría
        return _convertir_wkt(df, cols_geo["geometry"])
    
    else:
        # Permitir al usuario seleccionar columnas manualmente
        return _solicitar_columnas_manuales(df, archivo.name)


def _cargar_excel(archivo) -> Optional[gpd.GeoDataFrame]:
    """
    Carga un archivo Excel (.xlsx/.xls) y lo convierte a GeoDataFrame.
    Soporta múltiples hojas — el usuario elige cuál analizar.
    """
    archivo.seek(0)
    contenido = archivo.read()
    
    # Leer hojas disponibles
    try:
        xl = pd.ExcelFile(io.BytesIO(contenido), engine='openpyxl')
        hojas = xl.sheet_names
    except Exception as e:
        st.error(f"❌ Error leyendo Excel: {e}")
        return None
    
    # Si hay múltiples hojas, preguntar al usuario
    if len(hojas) > 1:
        hoja_sel = st.selectbox(
            f"📑 Selecciona la hoja de `{archivo.name}`:",
            hojas,
            key=f"hoja_{archivo.name}"
        )
    else:
        hoja_sel = hojas[0]
    
    df = pd.read_excel(io.BytesIO(contenido), sheet_name=hoja_sel, engine='openpyxl')
    
    if df.empty:
        st.error("❌ La hoja de Excel está vacía.")
        return None
    
    # Mismo proceso que CSV
    cols_geo = detectar_columnas_geograficas(df)
    lat_col = cols_geo.get("lat")
    lon_col = cols_geo.get("lon")
    
    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df_limpio = df.dropna(subset=[lat_col, lon_col]).copy()
        
        geometria = [Point(lon, lat) for lon, lat in 
                     zip(df_limpio[lon_col], df_limpio[lat_col])]
        
        gdf = gpd.GeoDataFrame(df_limpio, geometry=geometria, crs="EPSG:4326")
        st.success(f"✅ Excel cargado (hoja '{hoja_sel}'): **{len(gdf)} filas**")
        return gdf
    else:
        return _solicitar_columnas_manuales(df, archivo.name)


def _cargar_geojson(archivo) -> Optional[gpd.GeoDataFrame]:
    """
    Carga un archivo GeoJSON directamente con GeoPandas.
    GeoJSON ya incluye geometría, no necesita procesamiento extra.
    """
    archivo.seek(0)
    contenido = archivo.read()
    
    try:
        gdf = gpd.read_file(io.BytesIO(contenido))
    except Exception as e:
        st.error(f"❌ Error leyendo GeoJSON: {e}")
        return None
    
    if gdf.empty:
        st.error("❌ El GeoJSON está vacío.")
        return None
    
    # Asegurar proyección WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    st.success(f"✅ GeoJSON cargado: **{len(gdf)} features**, tipo `{gdf.geometry.geom_type.iloc[0]}`")
    return gdf


def _cargar_shapefile(archivo) -> Optional[gpd.GeoDataFrame]:
    """
    Carga un Shapefile individual.
    NOTA: Para SHP completo se necesita también .dbf y .shx.
    Streamlit permite subir múltiples archivos; aquí manejamos el .shp principal.
    """
    archivo.seek(0)
    contenido = archivo.read()
    
    try:
        gdf = gpd.read_file(io.BytesIO(contenido))
    except Exception:
        # Intentar guardar temporalmente en disco para fiona
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name
        try:
            gdf = gpd.read_file(tmp_path)
        except Exception as e:
            st.error(f"❌ Error leyendo Shapefile: {e}\n\n💡 Para Shapefiles, suba también los archivos `.dbf` y `.shx`.")
            return None
        finally:
            os.unlink(tmp_path)
    
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    st.success(f"✅ Shapefile cargado: **{len(gdf)} features**")
    return gdf


# -----------------------------------------------------------------------------
# CONVERSIÓN Y FALLBACK
# -----------------------------------------------------------------------------

def _convertir_wkt(df: pd.DataFrame, col_geom: str) -> Optional[gpd.GeoDataFrame]:
    """Convierte una columna WKT a geometría GeoPandas."""
    from shapely import wkt
    try:
        df["geometry"] = df[col_geom].apply(
            lambda x: wkt.loads(str(x)) if pd.notna(x) else None
        )
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        st.success(f"✅ Geometría WKT convertida: **{len(gdf)} filas**")
        return gdf
    except Exception as e:
        st.error(f"❌ Error convirtiendo WKT: {e}")
        return None


def _solicitar_columnas_manuales(df: pd.DataFrame, nombre_archivo: str) -> Optional[gpd.GeoDataFrame]:
    """
    Fallback: pide al usuario que indique manualmente las columnas de coordenadas
    cuando la detección automática falla.
    """
    st.warning(f"⚠️ No se detectaron columnas geográficas automáticamente en `{nombre_archivo}`.")
    
    cols_numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(cols_numericas) < 2:
        st.error("❌ No hay suficientes columnas numéricas para asignar coordenadas.")
        return None
    
    col1, col2 = st.columns(2)
    with col1:
        lat_manual = st.selectbox(
            "Selecciona columna de LATITUD:", cols_numericas,
            key=f"lat_manual_{nombre_archivo}"
        )
    with col2:
        lon_manual = st.selectbox(
            "Selecciona columna de LONGITUD:", cols_numericas,
            index=min(1, len(cols_numericas)-1),
            key=f"lon_manual_{nombre_archivo}"
        )
    
    if st.button("✅ Confirmar columnas", key=f"confirmar_{nombre_archivo}"):
        df[lat_manual] = pd.to_numeric(df[lat_manual], errors='coerce')
        df[lon_manual] = pd.to_numeric(df[lon_manual], errors='coerce')
        df_limpio = df.dropna(subset=[lat_manual, lon_manual]).copy()
        
        geometria = [Point(lon, lat) for lon, lat in 
                     zip(df_limpio[lon_manual], df_limpio[lat_manual])]
        
        gdf = gpd.GeoDataFrame(df_limpio, geometry=geometria, crs="EPSG:4326")
        st.success(f"✅ Dataset configurado manualmente: **{len(gdf)} filas**")
        return gdf
    
    return None


# -----------------------------------------------------------------------------
# VISTA PREVIA DE DATOS
# -----------------------------------------------------------------------------

def mostrar_preview(gdf: gpd.GeoDataFrame, nombre: str, n_filas: int = 5):
    """
    Muestra una vista previa del GeoDataFrame cargado con estadísticas básicas.
    
    Parámetros:
        gdf:    GeoDataFrame a previsualizar
        nombre: Nombre del dataset para mostrar
        n_filas: Número de filas a mostrar (por defecto 5)
    """
    st.markdown(f"### 📊 Vista previa: `{nombre}`")
    
    # Métricas rápidas en columnas
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📍 Filas", f"{len(gdf):,}")
    with c2:
        st.metric("📋 Columnas", f"{len(gdf.columns):,}")
    with c3:
        # Tipo de geometría predominante
        tipos = gdf.geometry.geom_type.value_counts()
        st.metric("🗺️ Geometría", tipos.index[0] if not tipos.empty else "N/A")
    with c4:
        col_num = detectar_columnas_numericas(gdf)
        st.metric("🔢 Cols. numéricas", f"{len(col_num)}")
    
    # Tabla de datos (sin columna geometry para mejor visualización)
    cols_mostrar = [c for c in gdf.columns if c != "geometry"]
    st.dataframe(
        gdf[cols_mostrar].head(n_filas),
        use_container_width=True,
        hide_index=False
    )
    
    # Información del CRS
    if gdf.crs:
        st.caption(f"🌐 Sistema de coordenadas: `{gdf.crs.to_string()}`")
    
    # Bounding box
    if not gdf.geometry.is_empty.all():
        bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
        st.caption(
            f"📐 Extensión: lon [{bbox[0]:.4f}, {bbox[2]:.4f}] | "
            f"lat [{bbox[1]:.4f}, {bbox[3]:.4f}]"
        )


def mostrar_dtypes(gdf: gpd.GeoDataFrame):
    """Muestra tipos de datos de cada columna en forma amigable."""
    info = []
    for col in gdf.columns:
        if col == "geometry":
            continue
        nulos = gdf[col].isna().sum()
        info.append({
            "Columna": col,
            "Tipo": str(gdf[col].dtype),
            "Valores únicos": gdf[col].nunique(),
            "Nulos": nulos,
            "% Nulos": f"{nulos/len(gdf)*100:.1f}%"
        })
    
    df_info = pd.DataFrame(info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)
