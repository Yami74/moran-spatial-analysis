# =============================================================================
# app.py
# Aplicativo principal — Sistema de Análisis de Autocorrelación Espacial
# Índice de Moran Global y Local (LISA)
# =============================================================================
# 
# Arquitectura:
#   app.py              → Orquestador principal y UI
#   modules/carga.py    → Carga y procesamiento de archivos geográficos
#   modules/moran.py    → Cálculo del Índice de Moran Global y Local
#   modules/mapas.py    → Visualizaciones (Folium, Plotly, scatter)
#   modules/reportes.py → Exportación a Excel, CSV y PDF
#   modules/utils.py    → Utilidades, constantes y helpers
#
# Para ejecutar:
#   streamlit run app.py
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

# ── Importar módulos del sistema ───────────────────────────────────────────
from modules.utils import (
    inicializar_sesion, guardar_dataset, guardar_resultado,
    obtener_datasets, obtener_resultados,
    calcular_estadisticas, detectar_columnas_numericas,
    interpretar_moran, COLORS, LISA_COLORS, LISA_LABELS
)
from modules.carga   import cargar_archivo, mostrar_preview, mostrar_dtypes
from modules.moran   import (construir_pesos, calcular_moran_global,
                              calcular_moran_local, mostrar_resumen_global,
                              mostrar_resumen_lisa)
from modules.mapas   import (grafico_moran_scatter, grafico_histograma,
                              grafico_boxplot, grafico_distribucion_simulacion,
                              grafico_clusters_lisa_barras, mapa_coropletico,
                              mapa_lisa, grafico_comparacion, tabla_comparacion)
from modules.reportes import botones_descarga


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Análisis Moran — Autocorrelación Espacial",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/pysal/esda",
        "Report a bug": None,
        "About":        "Sistema de Análisis de Autocorrelación Espacial — Índice de Moran I"
    }
)

# ── CSS personalizado para diseño profesional ──────────────────────────────
st.markdown("""
<style>
    /* Fuente y colores globales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1923 0%, #1A2634 100%);
        border-right: 1px solid #2E86AB33;
    }
    [data-testid="stSidebar"] * { color: #ECF0F1 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #BDC3C7 !important; }
    
    /* Título del sidebar */
    .sidebar-logo {
        text-align: center; padding: 1rem 0.5rem 0.5rem;
        border-bottom: 1px solid #2E86AB44; margin-bottom: 1rem;
    }
    .sidebar-logo h2 { color: #2E86AB !important; font-size: 1.1rem; margin: 0; }
    .sidebar-logo p  { color: #7F8C8D !important; font-size: 0.75rem; margin: 0; }
    
    /* Métricas */
    [data-testid="metric-container"] {
        background: #F8FAFE;
        border: 1px solid #D5E8F7;
        border-radius: 8px;
        padding: 0.7rem 1rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1B4F72; font-weight: 600; font-size: 1.4rem;
    }
    
    /* Encabezados de secciones */
    h1 { color: #1B4F72; font-weight: 700; }
    h2 { color: #2E86AB; font-weight: 600; }
    h3 { color: #1B4F72; font-weight: 600; border-bottom: 2px solid #2E86AB22; padding-bottom: 4px; }
    
    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab"] { 
        font-weight: 500; color: #7F8C8D;
    }
    [data-testid="stTabs"] [aria-selected="true"] { 
        color: #1B4F72 !important; border-bottom-color: #F18F01 !important;
    }
    
    /* Badges / Chips */
    .badge-hh { background:#D73027; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-ll { background:#4575B4; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-hl { background:#FC8D59; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-lh { background:#91BFDB; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-ns { background:#AAAAAA; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #2E86AB; border-radius: 3px; }
    
    /* Botones de descarga */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1B4F72, #2E86AB) !important;
        color: white !important; border: none !important;
        font-weight: 500 !important; border-radius: 6px !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #2E86AB, #1B4F72) !important;
        transform: translateY(-1px);
    }
    
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F1923 0%, #1B4F72 50%, #2E86AB 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 { color: white !important; margin: 0; font-size: 1.8rem; }
    .hero-banner p  { color: #AED6F1; margin: 0.3rem 0 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INICIALIZACIÓN DE SESIÓN
# =============================================================================

inicializar_sesion()


# =============================================================================
# SIDEBAR — MENÚ LATERAL
# =============================================================================

with st.sidebar:
    # Logo y título
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem;">🗺️</div>
        <h2>Análisis Moran</h2>
        <p>Autocorrelación Espacial</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Navegación principal ───────────────────────────────────────────
    pagina = st.radio(
        "📍 Navegación",
        options=[
            "🏠 Inicio",
            "📁 Cargar Datos",
            "📊 Análisis Moran",
            "🗺️ Mapas Interactivos",
            "📈 Visualizaciones",
            "⚖️ Comparación",
            "📥 Exportar",
        ],
        label_visibility="visible",
    )
    
    st.divider()
    
    # ── Configuración del análisis (solo visible si hay datos) ─────────
    datasets = obtener_datasets()
    
    if datasets:
        st.markdown("**⚙️ Configuración del Análisis**")
        
        dataset_sel = st.selectbox(
            "Dataset activo:",
            options=list(datasets.keys()),
            key="dataset_activo_sel",
        )
        st.session_state["dataset_activo"] = dataset_sel
        
        # Selección de variable
        gdf_activo = datasets.get(dataset_sel)
        if gdf_activo is not None:
            cols_num = detectar_columnas_numericas(gdf_activo)
            if cols_num:
                variable_sel = st.selectbox(
                    "Variable para análisis:",
                    options=cols_num,
                    key="variable_activa",
                )
            else:
                st.warning("No hay columnas numéricas disponibles.")
                variable_sel = None
        
        st.divider()
        
        # Parámetros de la matriz de pesos
        st.markdown("**🔧 Matriz de Pesos Espaciales**")
        
        metodo_pesos = st.selectbox(
            "Método:",
            ["KNN", "Queen", "Rook", "Distancia"],
            help="KNN: K vecinos más cercanos (recomendado para puntos)\n"
                 "Queen/Rook: Contigüidad (recomendado para polígonos)\n"
                 "Distancia: Radio fijo"
        )
        
        k_vecinos = None
        distancia_umbral = None
        
        if metodo_pesos == "KNN":
            k_vecinos = st.slider("K vecinos:", 2, 20, 5, 
                                   help="Número de vecinos más cercanos para cada unidad.")
        elif metodo_pesos == "Distancia":
            distancia_umbral = st.slider(
                "Radio (grados):", 0.1, 5.0, 1.0, 0.1,
                help="1 grado ≈ 111 km en el ecuador."
            )
        
        n_simulaciones = st.select_slider(
            "Permutaciones Monte Carlo:",
            options=[99, 199, 499, 999, 9999],
            value=999,
            help="Más permutaciones = mayor precisión del p-valor, mayor tiempo."
        )
        
        alpha = st.select_slider(
            "Nivel de significancia α:",
            options=[0.01, 0.05, 0.10],
            value=0.05,
        )
        
        st.divider()
    
    # ── Información de datasets cargados ─────────────────────────────
    if datasets:
        st.markdown(f"**📦 Datasets cargados: {len(datasets)}**")
        for nombre in datasets.keys():
            n = len(datasets[nombre])
            tiene_resultado = nombre in obtener_resultados()
            icono = "✅" if tiene_resultado else "⏳"
            st.caption(f"{icono} {nombre} ({n:,} filas)")
    
    st.divider()
    st.caption("📚 Estadística Espacial — PySAL / esda\nv1.0.0")


# =============================================================================
# PÁGINA: INICIO
# =============================================================================

if pagina == "🏠 Inicio":
    
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🗺️ Sistema de Análisis de Autocorrelación Espacial</h1>
        <p>Índice de Moran Global y Local (LISA) • Clusters HH/LL/HL/LH • Mapas Interactivos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas de estado actual
    datasets = obtener_datasets()
    resultados = obtener_resultados()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📁 Datasets cargados", len(datasets))
    with c2:
        st.metric("✅ Análisis completados", len(resultados))
    with c3:
        total_obs = sum(len(gdf) for gdf in datasets.values())
        st.metric("📍 Total observaciones", f"{total_obs:,}")
    with c4:
        total_sig = sum(
            1 for r in resultados.values()
            if r["moran_global"]["p_valor_sim"] < 0.05
        )
        st.metric("🔴 Resultados significativos", total_sig)
    
    st.divider()
    
    # Descripción de capacidades
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📌 ¿Qué es el Índice de Moran?")
        st.markdown("""
        El **Índice de Moran I** (1950) es el estadístico más utilizado para medir 
        la **autocorrelación espacial** — la tendencia de valores similares a 
        concentrarse geográficamente.
        
        **Interpretación del valor I:**
        
        | Valor | Patrón | Descripción |
        |-------|--------|-------------|
        | I > 0 | 🔴 Agrupamiento | Valores similares están próximos |
        | I ≈ 0 | 🎲 Aleatorio | Sin patrón espacial detectable |
        | I < 0 | 🔵 Dispersión | Valores disímiles son vecinos |
        
        El análisis **LISA** (Local Indicators of Spatial Association) 
        descompone el índice global e identifica zonas específicas de:
        - 🔴 **Hot Spots** (HH): Alta concentración de valores altos
        - 🔵 **Cold Spots** (LL): Alta concentración de valores bajos  
        - 🟠 **Outliers** (HL/LH): Anomalías espaciales locales
        """)
    
    with col_b:
        st.markdown("### 🚀 Cómo usar el sistema")
        st.markdown("""
        Siga estos pasos para realizar un análisis completo:
        
        **1️⃣ Cargar Datos**
        - Soporte para CSV, Excel, GeoJSON y Shapefile
        - Detección automática de coordenadas
        - Vista previa y validación de datos
        
        **2️⃣ Análisis Moran**
        - Seleccionar variable de interés
        - Configurar matriz de pesos espaciales
        - Calcular Moran Global + LISA automáticamente
        
        **3️⃣ Visualizaciones**
        - Moran Scatter Plot con clusters coloreados
        - Mapas interactivos coroplético y LISA
        - Histograma, boxplot y distribución de permutaciones
        
        **4️⃣ Comparación**
        - Analizar múltiples datasets simultáneamente
        - Tabla comparativa con interpretación automática
        
        **5️⃣ Exportar**
        - Reporte Excel con múltiples hojas
        - CSV con datos + resultados LISA
        - PDF profesional para presentación
        """)
    
    # Datasets de ejemplo si no hay datos
    if not datasets:
        st.info("👆 Ve a **📁 Cargar Datos** en el menú lateral para comenzar el análisis.")
    
    # Resumen rápido si hay resultados
    if resultados:
        st.markdown("### 📊 Resumen de Análisis Realizados")
        df_resumen = tabla_comparacion(resultados)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)


# =============================================================================
# PÁGINA: CARGAR DATOS
# =============================================================================

elif pagina == "📁 Cargar Datos":
    st.markdown("## 📁 Carga de Datos Geográficos")
    
    st.info("""
    **Formatos soportados:** CSV · Excel (.xlsx) · GeoJSON · Shapefile (.shp)  
    El sistema detecta automáticamente las columnas de latitud/longitud y geometría.
    """)
    
    # ── Zona de carga ────────────────────────────────────────────────
    archivos = st.file_uploader(
        "Arrastra aquí tus archivos geográficos",
        type=["csv", "xlsx", "xls", "geojson", "json", "shp"],
        accept_multiple_files=True,
        help="Puedes cargar múltiples archivos simultáneamente para comparar después."
    )
    
    if archivos:
        for archivo in archivos:
            nombre = archivo.name.rsplit(".", 1)[0]  # Nombre sin extensión
            
            # Evitar recargar si ya está en sesión
            if nombre in obtener_datasets():
                st.warning(f"⚠️ `{nombre}` ya está cargado. Será sobreescrito.")
            
            with st.spinner(f"Procesando `{archivo.name}`..."):
                gdf = cargar_archivo(archivo)
            
            if gdf is not None:
                guardar_dataset(nombre, gdf)
                
                # Mostrar preview
                with st.expander(f"👁️ Vista previa: `{nombre}`", expanded=True):
                    tab1, tab2 = st.tabs(["📊 Datos", "📋 Tipos de columnas"])
                    with tab1:
                        mostrar_preview(gdf, nombre)
                    with tab2:
                        mostrar_dtypes(gdf)
        
        st.success(f"✅ {len(archivos)} archivo(s) procesados correctamente.")
        st.info("👈 Ve a **📊 Análisis Moran** en el menú lateral para continuar.")
    
    # ── Gestión de datasets cargados ──────────────────────────────────
    datasets = obtener_datasets()
    if datasets and not archivos:
        st.markdown("### 📦 Datasets en sesión")
        for nombre, gdf in datasets.items():
            col_n, col_info, col_del = st.columns([2, 4, 1])
            with col_n:
                st.write(f"**`{nombre}`**")
            with col_info:
                st.caption(
                    f"{len(gdf):,} filas · {len(gdf.columns)} cols · "
                    f"{gdf.geometry.geom_type.value_counts().index[0]}"
                )
            with col_del:
                if st.button("🗑️", key=f"del_{nombre}", help=f"Eliminar {nombre}"):
                    del st.session_state["datasets"][nombre]
                    if nombre in st.session_state["resultados_moran"]:
                        del st.session_state["resultados_moran"][nombre]
                    st.rerun()


# =============================================================================
# PÁGINA: ANÁLISIS MORAN
# =============================================================================

elif pagina == "📊 Análisis Moran":
    st.markdown("## 📊 Análisis de Autocorrelación Espacial")
    
    datasets = obtener_datasets()
    
    if not datasets:
        st.warning("⚠️ No hay datos cargados. Ve a **📁 Cargar Datos** primero.")
        st.stop()
    
    # Recuperar configuración del sidebar
    dataset_sel  = st.session_state.get("dataset_activo_sel", list(datasets.keys())[0])
    variable_sel = st.session_state.get("variable_activa")
    
    if not variable_sel:
        st.error("❌ No se encontraron columnas numéricas en el dataset seleccionado.")
        st.stop()
    
    gdf = datasets[dataset_sel]
    
    # ── Panel de configuración visual ─────────────────────────────────
    with st.expander("⚙️ Configuración del Análisis", expanded=False):
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            st.markdown(f"**Dataset:** `{dataset_sel}`")
            st.markdown(f"**Variable:** `{variable_sel}`")
        with cfg2:
            st.markdown(f"**Método pesos:** `{metodo_pesos}`")
            if metodo_pesos == "KNN":
                st.markdown(f"**K vecinos:** `{k_vecinos}`")
        with cfg3:
            st.markdown(f"**Permutaciones:** `{n_simulaciones}`")
            st.markdown(f"**α:** `{alpha}`")
    
    # ── Estadísticas descriptivas ──────────────────────────────────────
    st.markdown("### 📈 Estadísticas Descriptivas")
    stats = calcular_estadisticas(gdf[variable_sel])
    
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Media",         f"{stats['media']:.4f}")
    s2.metric("Mediana",       f"{stats['mediana']:.4f}")
    s3.metric("Desv. Std",     f"{stats['desv_std']:.4f}")
    s4.metric("Mínimo",        f"{stats['minimo']:.4f}")
    s5.metric("Máximo",        f"{stats['maximo']:.4f}")
    
    with st.expander("📋 Estadísticas completas"):
        df_stats = pd.DataFrame({
            "Estadístico": ["N", "Media", "Mediana", "Moda", "Desv. Std", 
                             "Varianza", "Mínimo", "Máximo", "Rango",
                             "Q1", "Q3", "IQR", "Asimetría", "Curtosis", "CV%", "Nulos"],
            "Valor": [
                stats["n"], round(stats["media"],6), round(stats["mediana"],6),
                round(stats["moda"],6), round(stats["desv_std"],6), round(stats["varianza"],6),
                round(stats["minimo"],6), round(stats["maximo"],6), round(stats["rango"],6),
                round(stats["q1"],6), round(stats["q3"],6), round(stats["iqr"],6),
                round(stats["asimetria"],6), round(stats["curtosis"],6),
                round(stats["cv"],4) if stats["cv"] is not None and not np.isnan(stats["cv"]) else "N/A",
                stats["nulos"]
            ]
        })
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ── Botón de análisis ──────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        ejecutar = st.button(
            "🔬 Ejecutar Análisis Moran",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.info(f"Analizará la variable **`{variable_sel}`** en **{len(gdf):,} unidades** con método **{metodo_pesos}**.")
    
    if ejecutar:
        with st.status("🔄 Ejecutando análisis espacial...", expanded=True) as status:
            
            # ── Paso 1: Construir matriz de pesos ──────────────────────
            st.write("📐 Construyendo matriz de pesos espaciales...")
            w = construir_pesos(
                gdf, metodo_pesos,
                k=k_vecinos or 5,
                distancia=distancia_umbral or 1.0
            )
            
            if w is None:
                status.update(label="❌ Error en matriz de pesos", state="error")
                st.stop()
            
            # ── Paso 2: Moran Global ──────────────────────────────────
            st.write("📊 Calculando Índice de Moran Global...")
            resultado_global = calcular_moran_global(
                gdf, variable_sel, w, n_simulaciones
            )
            
            if resultado_global is None:
                status.update(label="❌ Error en Moran Global", state="error")
                st.stop()
            
            # ── Paso 3: Moran Local (LISA) ─────────────────────────────
            st.write("🗺️ Calculando LISA (Moran Local)...")
            resultado_lisa = calcular_moran_local(
                gdf, variable_sel, w, n_simulaciones, alpha
            )
            
            # ── Guardar en sesión ──────────────────────────────────────
            guardar_resultado(dataset_sel, {
                "moran_global": resultado_global,
                "moran_lisa":   resultado_lisa,
                "variable":     variable_sel,
                "metodo_pesos": metodo_pesos,
                "alpha":        alpha,
            })
            
            status.update(label="✅ Análisis completado", state="complete")
    
    # ── Mostrar resultados si existen ──────────────────────────────────
    resultados = obtener_resultados()
    
    if dataset_sel in resultados:
        res = resultados[dataset_sel]
        
        st.divider()
        
        # Mostrar resumen Global
        mostrar_resumen_global(res["moran_global"], dataset_sel)
        
        # Mostrar resumen LISA
        if res.get("moran_lisa"):
            st.divider()
            mostrar_resumen_lisa(res["moran_lisa"])


# =============================================================================
# PÁGINA: MAPAS INTERACTIVOS
# =============================================================================

elif pagina == "🗺️ Mapas Interactivos":
    from streamlit_folium import st_folium
    
    st.markdown("## 🗺️ Mapas Interactivos")
    
    datasets  = obtener_datasets()
    resultados = obtener_resultados()
    
    if not datasets:
        st.warning("⚠️ No hay datos cargados.")
        st.stop()
    
    dataset_sel = st.session_state.get("dataset_activo_sel", list(datasets.keys())[0])
    gdf = datasets[dataset_sel]
    
    tipo_mapa = st.radio(
        "Tipo de mapa:",
        ["🎨 Mapa Coroplético", "🗺️ Mapa de Clusters LISA"],
        horizontal=True,
    )
    
    if tipo_mapa == "🎨 Mapa Coroplético":
        cols_num = detectar_columnas_numericas(gdf)
        if not cols_num:
            st.error("No hay columnas numéricas para el mapa coroplético.")
            st.stop()
        
        variable_mapa = st.selectbox("Variable para colorear:", cols_num)
        
        with st.spinner("Generando mapa..."):
            m = mapa_coropletico(gdf, variable_mapa, f"Mapa Coroplético — {variable_mapa}")
        
        st_folium(m, width=None, height=550, returned_objects=[])
        
    else:  # LISA
        if dataset_sel not in resultados:
            st.warning("⚠️ Primero ejecuta el análisis Moran en la sección **📊 Análisis Moran**.")
            st.stop()
        
        res = resultados[dataset_sel]
        if not res.get("moran_lisa"):
            st.warning("⚠️ No hay resultados LISA disponibles.")
            st.stop()
        
        with st.spinner("Generando mapa LISA..."):
            m = mapa_lisa(gdf, res["variable"], res["moran_lisa"])
        
        st_folium(m, width=None, height=550, returned_objects=[])
        
        # Leyenda textual
        st.markdown("#### Leyenda de Clusters")
        cols_ley = st.columns(5)
        for col, (clave, color) in zip(cols_ley, [
            ("HH", "#D73027"), ("LL", "#4575B4"), 
            ("HL", "#FC8D59"), ("LH", "#91BFDB"), ("NS", "#EEEEEE")
        ]):
            col.markdown(
                f"<span style='background:{color}; color:{'white' if clave != 'NS' else 'gray'}; "
                f"padding:4px 10px; border-radius:12px; font-size:0.85rem; "
                f"border:1px solid #ccc;'><b>{clave}</b> — {LISA_LABELS[clave]}</span>",
                unsafe_allow_html=True
            )


# =============================================================================
# PÁGINA: VISUALIZACIONES
# =============================================================================

elif pagina == "📈 Visualizaciones":
    st.markdown("## 📈 Visualizaciones Estadísticas")
    
    datasets  = obtener_datasets()
    resultados = obtener_resultados()
    
    if not datasets:
        st.warning("⚠️ No hay datos cargados.")
        st.stop()
    
    dataset_sel = st.session_state.get("dataset_activo_sel", list(datasets.keys())[0])
    gdf = datasets[dataset_sel]
    
    if dataset_sel not in resultados:
        st.warning("⚠️ Ejecuta el análisis Moran primero en **📊 Análisis Moran**.")
        st.stop()
    
    res = resultados[dataset_sel]
    res_global = res["moran_global"]
    res_lisa   = res.get("moran_lisa")
    variable   = res["variable"]
    
    # ── Layout de tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔵 Moran Scatter", "📊 Distribución", "📦 Boxplot",
        "🎲 Permutaciones", "🗺️ LISA Barras"
    ])
    
    with tab1:
        st.markdown("#### Diagrama de Dispersión de Moran")
        st.caption(
            "Cada punto es una unidad espacial. La pendiente de la línea roja es el Índice de Moran I. "
            "Los puntos se colorean según el cluster LISA cuando está disponible."
        )
        fig_scatter = grafico_moran_scatter(res_global, res_lisa, f"Moran Scatter — {variable}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.markdown("#### Histograma con Densidad KDE")
        st.caption(f"Distribución de la variable `{variable}` con línea de media y curva de densidad.")
        fig_hist = grafico_histograma(gdf[variable], variable)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown("#### Diagrama de Cajas (Boxplot)")
        st.caption("Visualiza la mediana, cuartiles y valores atípicos de la distribución.")
        fig_box = grafico_boxplot(gdf[variable], variable)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.markdown("#### Distribución de Permutaciones Monte Carlo")
        st.caption(
            "Distribución del Índice de Moran bajo la hipótesis nula de aleatoriedad espacial. "
            "La línea roja es el valor observado."
        )
        fig_sim = grafico_distribucion_simulacion(res_global)
        st.plotly_chart(fig_sim, use_container_width=True)
    
    with tab5:
        if res_lisa:
            st.markdown("#### Distribución de Clusters LISA")
            st.caption("Número de unidades por tipo de cluster identificado en el análisis LISA.")
            fig_lisa = grafico_clusters_lisa_barras(res_lisa)
            st.plotly_chart(fig_lisa, use_container_width=True)
        else:
            st.warning("No hay resultados LISA disponibles para este dataset.")


# =============================================================================
# PÁGINA: COMPARACIÓN
# =============================================================================

elif pagina == "⚖️ Comparación":
    st.markdown("## ⚖️ Comparación entre Datasets")
    
    resultados = obtener_resultados()
    
    if len(resultados) < 2:
        st.warning("""
        ⚠️ Necesitas al menos **2 datasets analizados** para la comparación.
        
        Pasos:
        1. Carga múltiples archivos en **📁 Cargar Datos**
        2. Analiza cada uno en **📊 Análisis Moran**
        3. Vuelve aquí para comparar
        """)
        if resultados:
            st.info(f"Tienes {len(resultados)} dataset(s) analizado(s). Necesitas al menos 1 más.")
        st.stop()
    
    # ── Tabla comparativa ──────────────────────────────────────────────
    st.markdown("### 📋 Tabla Resumen Comparativa")
    df_comp = tabla_comparacion(resultados)
    
    # Colorear columna significativo
    def colorear_fila(row):
        if row["Significativo"] == "Sí":
            return ["background-color: #FDECEA"] * len(row)
        return [""] * len(row)
    
    st.dataframe(
        df_comp.style.apply(colorear_fila, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    
    # ── Gráfico de comparación ─────────────────────────────────────────
    st.markdown("### 📊 Comparación Visual del Índice de Moran I")
    fig_comp = grafico_comparacion(resultados)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # ── Interpretación comparativa ─────────────────────────────────────
    st.markdown("### 💡 Interpretación Comparativa")
    
    for nombre, res in resultados.items():
        mg = res["moran_global"]
        interp = interpretar_moran(mg["indice_i"], mg["p_valor_sim"], mg["z_score"])
        
        st.markdown(
            f"""
            <div style="border-left: 4px solid {interp['color']}; 
                        background: {interp['color']}11;
                        padding: 0.7rem 1rem; border-radius: 4px; margin: 0.5rem 0;">
                <strong>{interp['icono']} {nombre}</strong> — {interp['patron']}<br>
                <small>I = {mg['indice_i']:.4f} | p = {mg['p_valor_sim']:.4f} | Variable: {mg['columna']}</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ── Descargar tabla comparativa ───────────────────────────────────
    from modules.reportes import exportar_csv_resumen
    csv_comp = exportar_csv_resumen(resultados)
    st.download_button(
        "📄 Descargar Tabla Comparativa CSV",
        data=csv_comp,
        file_name="comparacion_moran.csv",
        mime="text/csv",
    )


# =============================================================================
# PÁGINA: EXPORTAR
# =============================================================================

elif pagina == "📥 Exportar":
    st.markdown("## 📥 Exportar Resultados")
    
    datasets  = obtener_datasets()
    resultados = obtener_resultados()
    
    if not resultados:
        st.warning("⚠️ No hay resultados para exportar. Ejecuta el análisis Moran primero.")
        st.stop()
    
    dataset_sel = st.session_state.get("dataset_activo_sel", list(resultados.keys())[0])
    
    if dataset_sel not in resultados:
        dataset_sel = list(resultados.keys())[0]
    
    gdf = datasets.get(dataset_sel)
    res = resultados[dataset_sel]
    
    st.info(f"Exportando resultados del dataset: **`{dataset_sel}`** — Variable: **`{res['variable']}`**")
    
    if gdf is not None:
        botones_descarga(
            gdf=gdf,
            resultado_global=res["moran_global"],
            resultado_lisa=res.get("moran_lisa"),
            nombre_dataset=dataset_sel,
        )
    
    # ── Comparación multi-dataset ──────────────────────────────────────
    if len(resultados) > 1:
        st.divider()
        st.markdown("### 📊 Exportar Comparación entre Datasets")
        
        from modules.reportes import exportar_csv_resumen
        csv_comp = exportar_csv_resumen(resultados)
        st.download_button(
            "📄 Descargar Tabla Comparativa (CSV)",
            data=csv_comp,
            file_name="comparacion_moran_todos.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    # ── Información de qué contiene cada formato ───────────────────────
    st.divider()
    st.markdown("### 📋 Contenido de cada formato")
    
    col_e, col_c, col_p = st.columns(3)
    with col_e:
        st.markdown("""
        **📊 Excel (.xlsx)**
        - Hoja 1: Datos originales
        - Hoja 2: Moran Global (todos los estadísticos)
        - Hoja 3: LISA (valor por unidad)
        - Hoja 4: Resumen LISA (por cluster)
        - Hoja 5: Estadísticas descriptivas
        """)
    with col_c:
        st.markdown("""
        **📄 CSV (.csv)**
        - Datos originales + resultados LISA
        - Columnas: LISA_I, LISA_p, LISA_Cluster
        - Codificación UTF-8 con BOM
        - Compatible con Excel y R/Python
        """)
    with col_p:
        st.markdown("""
        **📑 PDF (.pdf)**
        - Portada con metadatos del análisis
        - Tabla de resultados Moran Global
        - Distribución de clusters LISA
        - Estadísticas descriptivas
        - Pie con fecha y firma del sistema
        """)
