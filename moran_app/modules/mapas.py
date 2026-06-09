# =============================================================================
# modules/mapas.py
# Módulo de visualizaciones: Folium, Plotly, Moran Scatter, LISA maps
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, Dict, Any, List
from modules.utils import LISA_COLORS, LISA_LABELS, COLORS

warnings.filterwarnings('ignore')


# =============================================================================
# 1. MORAN SCATTER PLOT
# =============================================================================

def grafico_moran_scatter(resultado_global: Dict[str, Any], 
                           resultado_lisa: Optional[Dict[str, Any]] = None,
                           titulo: str = "Diagrama de Dispersión de Moran") -> go.Figure:
    """
    Crea el Moran Scatter Plot (Diagrama de Dispersión de Moran).
    
    El eje X muestra la variable estandarizada (z),
    el eje Y muestra el lag espacial (W·z).
    La pendiente de la línea de regresión es el Índice de Moran I.
    
    Los cuadrantes representan:
        I (++): High-High  |  II (-+): Low-High
        IV (+-): High-Low  |  III (--): Low-Low
    """
    y_std = np.array(resultado_global["y_estandar"])
    lag   = np.array(resultado_global["lag_espacial"])
    indice_i = resultado_global["indice_i"]
    
    # Color por cluster LISA si está disponible
    if resultado_lisa and resultado_lisa.get("cuadrante"):
        colores   = [LISA_COLORS.get(c, "#AAAAAA") for c in resultado_lisa["cuadrante"]]
        etiquetas = [LISA_LABELS.get(c, "N/A")    for c in resultado_lisa["cuadrante"]]
    else:
        colores   = [COLORS["secondary"]] * len(y_std)
        etiquetas = ["Observación"] * len(y_std)
    
    fig = go.Figure()
    
    # ── Scatter de puntos ──────────────────────────────────────────────────
    if resultado_lisa and resultado_lisa.get("cuadrante"):
        # Trazar por grupo para la leyenda
        grupos = pd.DataFrame({
            "x": y_std, "y": lag, 
            "color": colores, "label": etiquetas
        }).groupby("label")
        
        for nombre, grupo in grupos:
            fig.add_trace(go.Scatter(
                x=grupo["x"], y=grupo["y"],
                mode="markers",
                name=nombre,
                marker=dict(color=grupo["color"].iloc[0], size=8, opacity=0.8,
                            line=dict(width=0.5, color="white")),
                hovertemplate=f"<b>{nombre}</b><br>z: %{{x:.3f}}<br>Lag: %{{y:.3f}}<extra></extra>"
            ))
    else:
        fig.add_trace(go.Scatter(
            x=y_std, y=lag,
            mode="markers",
            name="Observaciones",
            marker=dict(color=COLORS["secondary"], size=7, opacity=0.7),
        ))
    
    # ── Línea de regresión (pendiente = I de Moran) ────────────────────────
    x_range = np.linspace(y_std.min() * 1.1, y_std.max() * 1.1, 100)
    y_fit   = indice_i * x_range  # y = I·x (pasando por el origen)
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        mode="lines",
        name=f"I = {indice_i:.4f}",
        line=dict(color=COLORS["accent"], width=2.5, dash="solid"),
    ))
    
    # ── Líneas de referencia (ejes de cuadrantes) ──────────────────────────
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), opacity=0.6)
    fig.add_vline(x=0, line=dict(color="gray", width=1, dash="dash"), opacity=0.6)
    
    # ── Anotaciones de cuadrantes ─────────────────────────────────────────
    x_max, y_max = y_std.max() * 0.9, lag.max() * 0.9
    x_min, y_min = y_std.min() * 0.9, lag.min() * 0.9
    
    for texto, px_pos, py_pos, color in [
        ("HH (+,+)", x_max,  y_max,  LISA_COLORS["HH"]),
        ("HL (+,−)", x_max,  y_min,  LISA_COLORS["HL"]),
        ("LH (−,+)", x_min,  y_max,  LISA_COLORS["LH"]),
        ("LL (−,−)", x_min,  y_min,  LISA_COLORS["LL"]),
    ]:
        fig.add_annotation(
            x=px_pos, y=py_pos, text=f"<b>{texto}</b>",
            showarrow=False, font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.7)", bordercolor=color,
        )
    
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=COLORS["primary"])),
        xaxis_title="Variable estandarizada (z)",
        yaxis_title="Lag espacial (Wz)",
        plot_bgcolor="white",
        legend=dict(orientation="v", x=1.02, y=0.5),
        height=480,
        margin=dict(l=60, r=160, t=60, b=60),
    )
    
    return fig


# =============================================================================
# 2. VISUALIZACIONES ESTADÍSTICAS
# =============================================================================

def grafico_histograma(serie: pd.Series, columna: str, 
                        color: str = "#2E86AB") -> go.Figure:
    """Histograma con curva KDE para la distribución de la variable."""
    valores = pd.to_numeric(serie, errors='coerce').dropna()
    
    fig = go.Figure()
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=valores, nbinsx=30,
        name="Frecuencia",
        marker_color=color,
        opacity=0.75,
        histnorm="probability density",
    ))
    
    # Curva KDE (estimación de densidad por kernel)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(valores)
    x_range = np.linspace(valores.min(), valores.max(), 200)
    fig.add_trace(go.Scatter(
        x=x_range, y=kde(x_range),
        mode="lines", name="KDE",
        line=dict(color=COLORS["accent"], width=2.5),
    ))
    
    # Línea de la media
    fig.add_vline(
        x=float(valores.mean()),
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        annotation_text=f"Media: {valores.mean():.2f}",
        annotation_position="top right",
    )
    
    fig.update_layout(
        title=f"Distribución de '{columna}'",
        xaxis_title=columna,
        yaxis_title="Densidad",
        plot_bgcolor="white",
        height=380,
    )
    
    return fig


def grafico_boxplot(serie: pd.Series, columna: str) -> go.Figure:
    """Boxplot con puntos jitter para inspección de outliers."""
    valores = pd.to_numeric(serie, errors='coerce').dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=valores,
        name=columna,
        boxpoints="outliers",  # Mostrar solo outliers como puntos
        marker=dict(color=COLORS["accent"], size=5),
        line=dict(color=COLORS["primary"], width=2),
        fillcolor=f"{COLORS['secondary']}44",
        whiskerwidth=0.5,
        notched=True,  # Muesca para intervalo de confianza de la mediana
    ))
    
    fig.update_layout(
        title=f"Boxplot de '{columna}'",
        yaxis_title=columna,
        plot_bgcolor="white",
        height=380,
        showlegend=False,
    )
    
    return fig


def grafico_distribucion_simulacion(resultado_global: Dict[str, Any]) -> go.Figure:
    """
    Histograma de la distribución de permutaciones Monte Carlo
    con la línea del Índice de Moran observado.
    """
    sim_valores = np.array(resultado_global["sim_valores"])
    i_obs = resultado_global["indice_i"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=sim_valores,
        nbinsx=50,
        name="Permutaciones",
        marker_color=COLORS["secondary"],
        opacity=0.7,
        histnorm="probability density",
    ))
    
    # Línea del I observado
    fig.add_vline(
        x=i_obs,
        line=dict(color=COLORS["danger"], width=3),
        annotation_text=f"I = {i_obs:.4f}",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["danger"], size=13),
    )
    
    # Línea del I esperado
    fig.add_vline(
        x=resultado_global["valor_esperado"],
        line=dict(color="gray", width=2, dash="dot"),
        annotation_text=f"E[I] = {resultado_global['valor_esperado']:.4f}",
        annotation_position="top left",
    )
    
    fig.update_layout(
        title="Distribución de Permutaciones Monte Carlo",
        xaxis_title="Índice de Moran I (simulado)",
        yaxis_title="Densidad",
        plot_bgcolor="white",
        height=380,
    )
    
    return fig


def grafico_clusters_lisa_barras(resultado_lisa: Dict[str, Any]) -> go.Figure:
    """Gráfico de barras con el conteo de unidades por tipo de cluster LISA."""
    
    categorias = ["HH", "LL", "HL", "LH", "NS"]
    conteos = [
        resultado_lisa["n_HH"],
        resultado_lisa["n_LL"],
        resultado_lisa["n_HL"],
        resultado_lisa["n_LH"],
        resultado_lisa["n_NS"],
    ]
    colores = [LISA_COLORS[c] for c in categorias]
    labels  = [LISA_LABELS[c] for c in categorias]
    n_total = resultado_lisa["n_total"]
    pcts    = [f"{c/n_total*100:.1f}%" for c in conteos]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=conteos,
        marker_color=colores,
        text=pcts,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>N: %{y}<br>%: %{text}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Distribución de Clusters LISA",
        xaxis_title="Tipo de Cluster",
        yaxis_title="Número de unidades",
        plot_bgcolor="white",
        height=380,
        showlegend=False,
    )
    
    return fig


# =============================================================================
# 3. MAPA INTERACTIVO FOLIUM — COROPLÉTICO
# =============================================================================

def mapa_coropletico(gdf: gpd.GeoDataFrame, columna: str, 
                      titulo: str = "Mapa Coroplético") -> folium.Map:
    """
    Crea un mapa coroplético interactivo con Folium.
    Colorea cada unidad según el valor de la variable seleccionada.
    
    Parámetros:
        gdf:     GeoDataFrame con geometría y datos
        columna: Variable para colorear el mapa
        titulo:  Título del mapa
    
    Retorna:
        Objeto folium.Map listo para renderizar
    """
    # Centro del mapa (centroide del conjunto)
    centro = [
        gdf.geometry.centroid.y.mean(),
        gdf.geometry.centroid.x.mean()
    ]
    
    # Crear mapa base
    m = folium.Map(
        location=centro,
        zoom_start=6,
        tiles="CartoDB positron",  # Tiles oscuros modernos
    )
    
    # Valores para la escala de colores
    valores = pd.to_numeric(gdf[columna], errors='coerce')
    vmin, vmax = valores.min(), valores.max()
    
    # GeoJson con colormap automático
    choropleth = folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf[[columna]].reset_index(),
        columns=["index", columna],
        key_on="feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name=f"{columna}",
        nan_fill_color="lightgray",
        highlight=True,
        name=f"Coroplético: {columna}",
    )
    choropleth.add_to(m)
    
    # Tooltips con información al hacer hover
    cols_tooltip = [c for c in gdf.columns[:6] if c != "geometry"]
    folium.GeoJson(
        gdf,
        tooltip=folium.GeoJsonTooltip(
            fields=cols_tooltip,
            aliases=[f"{c}:" for c in cols_tooltip],
            localize=True,
            sticky=True,
        ),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0},
    ).add_to(m)
    
    # Control de capas
    folium.LayerControl().add_to(m)
    
    # Minimap
    from folium.plugins import MiniMap
    MiniMap(toggle_display=True).add_to(m)
    
    return m


# =============================================================================
# 4. MAPA DE CLUSTERS LISA
# =============================================================================

def mapa_lisa(gdf: gpd.GeoDataFrame, columna: str,
              resultado_lisa: Dict[str, Any]) -> folium.Map:
    """
    Crea el mapa de clusters LISA (Local Indicators of Spatial Association).
    Cada unidad se colorea según su tipo de cluster:
        Rojo intenso: HH (Hot Spot)
        Azul intenso: LL (Cold Spot)
        Naranja: HL (outlier positivo)
        Azul claro: LH (outlier negativo)
        Gris: NS (no significativo)
    """
    from modules.moran import agregar_resultados_lisa
    
    # Enriquecer GDF con resultados LISA
    gdf_lisa = agregar_resultados_lisa(gdf, resultado_lisa)
    
    centro = [
        gdf_lisa.geometry.centroid.y.mean(),
        gdf_lisa.geometry.centroid.x.mean()
    ]
    
    m = folium.Map(location=centro, zoom_start=6, tiles="CartoDB positron")
    
    # Renderizar cada unidad con su color LISA
    for _, row in gdf_lisa.iterrows():
        cluster = row.get("lisa_cluster", "NS")
        color   = LISA_COLORS.get(cluster, "#EEEEEE")
        label   = LISA_LABELS.get(cluster, "N/A")
        
        # Construir popup con información de la unidad
        val_col = pd.to_numeric(row.get(columna, np.nan), errors='coerce')
        popup_html = f"""
        <div style='font-family: Arial; min-width: 150px;'>
            <b>Cluster:</b> {label}<br>
            <b>{columna}:</b> {val_col:.4f if not np.isnan(val_col) else 'N/A'}<br>
            <b>LISA I:</b> {row.get('lisa_I', 'N/A'):.4f if pd.notna(row.get('lisa_I')) else 'N/A'}<br>
            <b>p-valor:</b> {row.get('lisa_p', 'N/A'):.4f if pd.notna(row.get('lisa_p')) else 'N/A'}
        </div>
        """
        
        # Determinar función de estilo (diferente para puntos y polígonos)
        geom_type = row.geometry.geom_type if row.geometry else "Unknown"
        
        if "Point" in geom_type:
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{cluster}: {label}",
            ).add_to(m)
        else:
            # Polígonos / Líneas
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor":   c,
                    "color":       "white",
                    "weight":      0.8,
                    "fillOpacity": 0.80,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{cluster}: {label}",
            ).add_to(m)
    
    # Leyenda HTML personalizada
    leyenda_html = """
    <div style='
        position: fixed; bottom: 30px; left: 30px; z-index: 999;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: Arial; font-size: 13px;
    '>
        <b>🗺️ Clusters LISA</b><br><br>
    """
    for clave, color in LISA_COLORS.items():
        leyenda_html += (
            f"<span style='background:{color}; width:14px; height:14px; "
            f"display:inline-block; border-radius:2px; margin-right:6px; "
            f"border:1px solid #ccc;'></span>{LISA_LABELS[clave]}<br>"
        )
    leyenda_html += "</div>"
    
    m.get_root().html.add_child(folium.Element(leyenda_html))
    folium.LayerControl().add_to(m)
    
    return m


# =============================================================================
# 5. COMPARACIÓN ENTRE DATASETS
# =============================================================================

def grafico_comparacion(resultados: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Gráfico de barras comparativo del Índice de Moran entre múltiples datasets.
    Incluye barras de error con el intervalo de confianza.
    """
    nombres  = list(resultados.keys())
    indices  = [r["moran_global"]["indice_i"] for r in resultados.values()]
    pvalores = [r["moran_global"]["p_valor_sim"] for r in resultados.values()]
    
    # Color por significancia
    colores = [
        COLORS["danger"] if p < 0.05 else COLORS["neutral"] 
        for p in pvalores
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=nombres,
        y=indices,
        marker_color=colores,
        text=[f"I={i:.4f}<br>p={p:.4f}" for i, p in zip(indices, pvalores)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Moran I: %{y:.4f}<extra></extra>",
        name="Índice de Moran I",
    ))
    
    # Línea de referencia en I=0 (aleatoriedad)
    fig.add_hline(
        y=0, line=dict(color="gray", dash="dash", width=1.5),
        annotation_text="I = 0 (aleatorio)",
    )
    
    fig.update_layout(
        title="Comparación del Índice de Moran entre Datasets",
        xaxis_title="Dataset",
        yaxis_title="Índice de Moran I",
        plot_bgcolor="white",
        height=420,
        showlegend=False,
    )
    
    return fig


def tabla_comparacion(resultados: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Genera la tabla resumen de comparación entre datasets.
    Incluye nombre, I de Moran, p-valor e interpretación.
    """
    from modules.utils import interpretar_moran
    
    filas = []
    for nombre, res in resultados.items():
        mg = res["moran_global"]
        interp = interpretar_moran(mg["indice_i"], mg["p_valor_sim"], mg["z_score"])
        
        filas.append({
            "Dataset":        nombre,
            "Variable":       mg["columna"],
            "N unidades":     mg["n"],
            "Moran I":        round(mg["indice_i"], 6),
            "Z-score":        round(mg["z_score"], 4),
            "p-valor":        round(mg["p_valor_sim"], 6),
            "Significativo":  "Sí" if mg["p_valor_sim"] < 0.05 else "No",
            "Patrón":         interp["patron"],
        })
    
    return pd.DataFrame(filas)
