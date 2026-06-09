# =============================================================================
# modules/moran.py
# Módulo de cálculo del Índice de Moran Global y Local (LISA)
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from typing import Optional, Dict, Any, Tuple
from modules.utils import LISA_COLORS, LISA_LABELS

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# CONSTRUCCIÓN DE MATRICES DE PESOS ESPACIALES
# -----------------------------------------------------------------------------

def construir_pesos(gdf: gpd.GeoDataFrame, metodo: str, 
                    k: int = 5, distancia: float = 1.0) -> Optional[Any]:
    """
    Construye la matriz de pesos espaciales W según el método seleccionado.
    
    La matriz W define la "vecindad" entre unidades espaciales:
    W[i,j] = 1 si i y j son vecinos, 0 si no.
    
    Métodos disponibles:
        - KNN: K vecinos más cercanos — siempre produce vecinos (útil para puntos)
        - Queen: Contigüidad reina — comparte al menos un punto (polígonos)
        - Rook: Contigüidad torre — comparte un borde completo (polígonos)
        - Distancia fija: Vecinos dentro de un radio definido
    
    Parámetros:
        gdf:       GeoDataFrame con las geometrías
        metodo:    Tipo de peso ('KNN', 'Queen', 'Rook', 'Distancia')
        k:         Número de vecinos para KNN (por defecto 5)
        distancia: Radio en grados para distancia fija (por defecto 1.0)
    
    Retorna:
        Objeto W de libpysal o None si falla
    """
    try:
        from libpysal import weights
        
        # Reproducibilidad: usar proyección métrica para KNN y distancia
        # EPSG:3857 (Web Mercator) para cálculos de distancia
        gdf_proj = gdf.copy()
        
        if metodo == "KNN":
            # K-Nearest Neighbors: garantiza que cada unidad tenga exactamente k vecinos
            # Ideal para datos de puntos y cuando se necesita matriz no-singular
            k_real = min(k, len(gdf) - 1)  # No puede haber más vecinos que registros
            w = weights.KNN.from_dataframe(gdf_proj, k=k_real)
            st.info(f"ℹ️ Matriz KNN construida con k={k_real} vecinos por unidad.")
            
        elif metodo == "Queen":
            # Contigüidad Queen: dos polígonos son vecinos si comparten al menos un punto
            # Más permisiva que Rook, captura vecinos diagonales
            w = weights.Queen.from_dataframe(gdf_proj)
            
        elif metodo == "Rook":
            # Contigüidad Rook: dos polígonos son vecinos si comparten un segmento de borde
            # Más estricta que Queen, excluye vecinos solo por un punto
            w = weights.Rook.from_dataframe(gdf_proj)
            
        elif metodo == "Distancia":
            # Distancia fija: vecinos dentro de un radio específico
            # Convertir distancia de grados a unidades si es necesario
            gdf_merc = gdf_proj.to_crs("EPSG:3857")
            distancia_m = distancia * 111_000  # Aproximación: 1 grado ≈ 111 km
            w = weights.DistanceBand.from_dataframe(
                gdf_merc, threshold=distancia_m, binary=True
            )
            st.info(f"ℹ️ Umbral de distancia: {distancia:.2f}° ≈ {distancia_m/1000:.1f} km.")
            
        else:
            st.error(f"❌ Método de pesos desconocido: {metodo}")
            return None
        
        # Verificar islas (unidades sin vecinos)
        if w.n_components > 1:
            islas = [i for i, vecinos in w.neighbors.items() if len(vecinos) == 0]
            if islas:
                st.warning(
                    f"⚠️ Hay {len(islas)} unidades sin vecinos ('islas'). "
                    f"Esto puede afectar los resultados. Considere otro método de pesos."
                )
        
        # Estandarización por fila: w[i,j] / Σw[i,*]
        # Esto hace que la suma de pesos de cada fila sea 1 (promedio ponderado)
        w.transform = 'r'
        
        st.success(
            f"✅ Matriz **{metodo}** construida: {w.n} unidades, "
            f"{w.pct_nonzero:.1f}% de conexiones no-cero."
        )
        return w
        
    except ImportError:
        st.error("❌ libpysal no está instalado. Ejecute: pip install libpysal")
        return None
    except Exception as e:
        st.error(f"❌ Error construyendo matriz de pesos: {str(e)}")
        return None


# -----------------------------------------------------------------------------
# ÍNDICE DE MORAN GLOBAL
# -----------------------------------------------------------------------------

def calcular_moran_global(gdf: gpd.GeoDataFrame, columna: str, 
                           w, n_simulaciones: int = 999) -> Optional[Dict[str, Any]]:
    """
    Calcula el Índice de Moran Global I.
    
    Fórmula:
        I = (N / S₀) × [Σᵢ Σⱼ wᵢⱼ(xᵢ-x̄)(xⱼ-x̄)] / [Σᵢ(xᵢ-x̄)²]
    
    Donde:
        N  = número de unidades
        S₀ = suma de todos los pesos
        wᵢⱼ = peso espacial entre i y j
        xᵢ = valor de la variable en unidad i
        x̄  = media de la variable
    
    Valores de I:
        I ≈ E[I] → Aleatoriedad espacial
        I > E[I] → Autocorrelación positiva (agrupamiento)
        I < E[I] → Autocorrelación negativa (dispersión)
    
    Parámetros:
        gdf:            GeoDataFrame con los datos
        columna:        Variable a analizar
        w:              Matriz de pesos espaciales
        n_simulaciones: Permutaciones Monte Carlo para el p-valor
    
    Retorna:
        Diccionario con todos los estadísticos calculados
    """
    try:
        from esda.moran import Moran
        
        # Extraer y limpiar la variable de análisis
        y = pd.to_numeric(gdf[columna], errors='coerce')
        
        # Alinear con la matriz de pesos (puede tener menos unidades por islas)
        idx_validos = y.dropna().index
        if len(idx_validos) < len(y):
            st.warning(f"⚠️ Se usaron {len(idx_validos)} de {len(y)} filas (se descartaron NaN).")
        
        y_clean = y[idx_validos].values
        
        # Calcular Moran I con inferencia por permutación Monte Carlo
        # La inferencia por permutación es más robusta que la asintótica
        mi = Moran(y_clean, w, permutations=n_simulaciones, two_tailed=True)
        
        # Valor esperado teórico: E[I] = -1/(N-1)
        ei_teorico = -1 / (len(y_clean) - 1)
        
        resultado = {
            # ── Estadísticos principales ──────────────────────────────────
            "indice_i":       float(mi.I),          # Índice de Moran I
            "valor_esperado": float(mi.EI),         # E[I] calculado
            "ei_teorico":     float(ei_teorico),    # E[I] teórico = -1/(n-1)
            "varianza":       float(mi.VI_norm),    # Var[I] bajo normalidad
            "z_score":        float(mi.z_norm),     # Z = (I - E[I]) / sqrt(Var[I])
            "p_valor":        float(mi.p_norm),     # p-valor bajo normalidad
            "p_valor_sim":    float(mi.p_sim),      # p-valor por simulación
            "z_sim":          float(mi.z_sim),      # Z por simulación
            # ── Distribución de permutaciones ─────────────────────────────
            "sim_valores":    mi.sim.tolist(),       # I de cada permutación
            "n":              int(len(y_clean)),
            "n_sims":         int(n_simulaciones),
            # ── Datos para Moran Scatter Plot ─────────────────────────────
            "y_estandar":     mi.z.tolist(),         # Variable estandarizada
            "lag_espacial":   mi.Ez.tolist() if hasattr(mi, 'Ez') else [],
            "columna":        columna,
        }
        
        # Calcular lag espacial manualmente si no está en mi
        from libpysal.weights import lag_spatial
        y_std = (y_clean - y_clean.mean()) / y_clean.std()
        lag = lag_spatial(w, y_std)
        resultado["y_estandar"] = y_std.tolist()
        resultado["lag_espacial"] = lag.tolist()
        
        return resultado
        
    except ImportError:
        st.error("❌ esda no está instalado. Ejecute: pip install esda")
        return None
    except Exception as e:
        st.error(f"❌ Error calculando Moran Global: {str(e)}")
        return None


# -----------------------------------------------------------------------------
# ÍNDICE DE MORAN LOCAL (LISA)
# -----------------------------------------------------------------------------

def calcular_moran_local(gdf: gpd.GeoDataFrame, columna: str,
                          w, n_simulaciones: int = 999,
                          alpha: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    Calcula el Índice de Moran Local (LISA - Local Indicators of Spatial Association).
    
    LISA descompone el Índice Global en contribuciones locales.
    Para cada unidad i se calcula:
        Iᵢ = (xᵢ - x̄) / m₂ × Σⱼ wᵢⱼ(xⱼ - x̄)
    
    Donde m₂ = Σ(xᵢ - x̄)² / N (varianza)
    
    Clasificación de clusters:
        HH (High-High): Iᵢ > 0, xᵢ > x̄ → Hot Spot
        LL (Low-Low):   Iᵢ > 0, xᵢ < x̄ → Cold Spot
        HL (High-Low):  Iᵢ < 0, xᵢ > x̄ → Outlier positivo
        LH (Low-High):  Iᵢ < 0, xᵢ < x̄ → Outlier negativo
        NS:             No significativo estadísticamente
    
    Parámetros:
        gdf:            GeoDataFrame con los datos
        columna:        Variable a analizar
        w:              Matriz de pesos espaciales
        n_simulaciones: Permutaciones para p-valor local
        alpha:          Nivel de significancia (por defecto 0.05)
    
    Retorna:
        Diccionario con resultados LISA y clasificación de clusters
    """
    try:
        from esda.moran import Moran_Local
        from libpysal.weights import lag_spatial
        
        # Preparar datos
        y = pd.to_numeric(gdf[columna], errors='coerce')
        y_clean = y.fillna(y.median()).values  # Imputar NaN con mediana para LISA
        
        # Calcular LISA
        lisa = Moran_Local(y_clean, w, permutations=n_simulaciones)
        
        # ── Clasificación de cuadrantes del Moran Scatter Plot ────────────
        # Estandarizar la variable
        y_std = (y_clean - y_clean.mean()) / y_clean.std()
        
        # Lag espacial estandarizado
        lag_std = lag_spatial(w, y_std)
        
        # Asignar cuadrante según posición en scatter plot
        # Eje X: variable propia (y_std), Eje Y: lag espacial (lag_std)
        cuadrante = []
        for i in range(len(y_std)):
            if lisa.p_sim[i] < alpha:  # Solo si es estadísticamente significativo
                if y_std[i] >= 0 and lag_std[i] >= 0:
                    cuadrante.append("HH")   # Alto-Alto: Cuadrante I
                elif y_std[i] < 0 and lag_std[i] < 0:
                    cuadrante.append("LL")   # Bajo-Bajo: Cuadrante III
                elif y_std[i] >= 0 and lag_std[i] < 0:
                    cuadrante.append("HL")   # Alto-Bajo: Cuadrante IV
                else:
                    cuadrante.append("LH")   # Bajo-Alto: Cuadrante II
            else:
                cuadrante.append("NS")       # No significativo
        
        # Conteo de clusters
        conteo = pd.Series(cuadrante).value_counts().to_dict()
        
        resultado = {
            # ── Estadísticos LISA ──────────────────────────────────────────
            "Is":          lisa.Is.tolist(),       # Valor LISA por unidad
            "p_sim":       lisa.p_sim.tolist(),    # p-valor por simulación
            "z_sim":       lisa.z_sim.tolist(),    # Z por simulación
            "cuadrante":   cuadrante,               # Clasificación HH/LL/HL/LH/NS
            # ── Variables para scatter plot ────────────────────────────────
            "y_std":       y_std.tolist(),
            "lag_std":     lag_std.tolist(),
            # ── Resumen de clusters ────────────────────────────────────────
            "n_HH":        conteo.get("HH", 0),
            "n_LL":        conteo.get("LL", 0),
            "n_HL":        conteo.get("HL", 0),
            "n_LH":        conteo.get("LH", 0),
            "n_NS":        conteo.get("NS", 0),
            "n_total":     len(cuadrante),
            "n_significativo": sum(1 for c in cuadrante if c != "NS"),
            "alpha":       alpha,
            "columna":     columna,
        }
        
        return resultado
        
    except ImportError:
        st.error("❌ esda no está instalado.")
        return None
    except Exception as e:
        st.error(f"❌ Error calculando Moran Local: {str(e)}")
        return None


# -----------------------------------------------------------------------------
# ENRIQUECIMIENTO DEL GEODATAFRAME
# -----------------------------------------------------------------------------

def agregar_resultados_lisa(gdf: gpd.GeoDataFrame, 
                             resultado_lisa: Dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Agrega las columnas de resultados LISA al GeoDataFrame original.
    Permite su uso en visualizaciones cartográficas.
    
    Columnas añadidas:
        lisa_I:         Valor del índice local
        lisa_p:         p-valor de significancia
        lisa_cluster:   Clasificación HH/LL/HL/LH/NS
        lisa_color:     Color hexadecimal del cluster
        lisa_label:     Etiqueta descriptiva del cluster
    """
    gdf = gdf.copy()
    
    n = min(len(gdf), len(resultado_lisa["Is"]))
    
    gdf["lisa_I"]       = resultado_lisa["Is"][:n]
    gdf["lisa_p"]       = resultado_lisa["p_sim"][:n]
    gdf["lisa_cluster"] = resultado_lisa["cuadrante"][:n]
    gdf["lisa_color"]   = [LISA_COLORS.get(c, "#EEEEEE") for c in resultado_lisa["cuadrante"][:n]]
    gdf["lisa_label"]   = [LISA_LABELS.get(c, "N/A") for c in resultado_lisa["cuadrante"][:n]]
    
    return gdf


def mostrar_resumen_global(resultado: Dict[str, Any], nombre_dataset: str):
    """
    Muestra el resumen del Índice de Moran Global en la interfaz.
    Presenta métricas en tarjetas y la interpretación estadística.
    """
    from modules.utils import interpretar_moran, nivel_significancia_texto, formatear_numero
    
    st.markdown(f"#### 📊 Resultados: Moran Global — `{nombre_dataset}`")
    
    # Métricas principales en 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Índice de Moran I", formatear_numero(resultado["indice_i"]))
    with col2:
        st.metric("Valor p (normal.)", formatear_numero(resultado["p_valor"]))
    with col3:
        st.metric("Z-score", formatear_numero(resultado["z_score"]))
    with col4:
        st.metric("p (simulación)", formatear_numero(resultado["p_valor_sim"]))
    
    # Interpretación
    interpretacion = interpretar_moran(
        resultado["indice_i"], 
        resultado["p_valor_sim"],
        resultado["z_score"]
    )
    
    st.markdown(
        f"""
        <div style="
            background: {interpretacion['color']}22; 
            border-left: 4px solid {interpretacion['color']};
            padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;
        ">
            <strong>{interpretacion['icono']} {interpretacion['patron']}</strong><br>
            {interpretacion['descripcion']}<br><br>
            <em>💡 {interpretacion['recomendacion']}</em>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Tabla de significancia
    sig_texto = nivel_significancia_texto(resultado["p_valor_sim"])
    
    with st.expander("📋 Tabla estadística completa"):
        tabla = pd.DataFrame({
            "Estadístico": [
                "Índice de Moran (I)", "Valor esperado E[I]", 
                "Varianza Var[I]", "Z-score (normal.)",
                "p-valor (normal.)", "p-valor (simulación)",
                "N unidades", "N permutaciones", "Significancia"
            ],
            "Valor": [
                formatear_numero(resultado["indice_i"]),
                formatear_numero(resultado["valor_esperado"]),
                formatear_numero(resultado["varianza"]),
                formatear_numero(resultado["z_score"]),
                formatear_numero(resultado["p_valor"]),
                formatear_numero(resultado["p_valor_sim"]),
                str(resultado["n"]),
                str(resultado["n_sims"]),
                sig_texto,
            ]
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)


def mostrar_resumen_lisa(resultado_lisa: Dict[str, Any]):
    """Muestra el resumen de clusters LISA con conteos y porcentajes."""
    st.markdown("#### 🗺️ Distribución de Clusters LISA")
    
    n_total = resultado_lisa["n_total"]
    clusters = [
        ("🔴", "HH", resultado_lisa["n_HH"], "Alto-Alto"),
        ("🔵", "LL", resultado_lisa["n_LL"], "Bajo-Bajo"),
        ("🟠", "HL", resultado_lisa["n_HL"], "Alto-Bajo"),
        ("🩵", "LH", resultado_lisa["n_LH"], "Bajo-Alto"),
        ("⚪", "NS", resultado_lisa["n_NS"], "No significativo"),
    ]
    
    cols = st.columns(5)
    for col, (icono, clave, n, label) in zip(cols, clusters):
        pct = n / n_total * 100 if n_total > 0 else 0
        col.metric(f"{icono} {clave}", f"{n}", f"{pct:.1f}%")
        col.caption(label)
