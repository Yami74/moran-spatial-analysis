# =============================================================================
# modules/reportes.py
# Módulo de exportación de resultados: Excel, CSV, PDF
# =============================================================================

import io
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# EXPORTAR A EXCEL
# -----------------------------------------------------------------------------

def exportar_excel(gdf: gpd.GeoDataFrame, resultado_global: Dict[str, Any],
                   resultado_lisa: Optional[Dict[str, Any]],
                   nombre_dataset: str) -> bytes:
    """
    Genera un archivo Excel con múltiples hojas de resultados:
        1. Datos originales
        2. Resultados Moran Global
        3. Resultados LISA
        4. Estadísticas descriptivas
    
    Parámetros:
        gdf:             GeoDataFrame con los datos
        resultado_global: Dict con resultados del Moran Global
        resultado_lisa:   Dict con resultados LISA (puede ser None)
        nombre_dataset:  Nombre del dataset para encabezados
    
    Retorna:
        Bytes del archivo Excel listo para descargar
    """
    from modules.utils import calcular_estadisticas, interpretar_moran
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        
        # ── Hoja 1: Datos originales ──────────────────────────────────────
        cols_datos = [c for c in gdf.columns if c != "geometry"]
        df_datos = gdf[cols_datos].copy()
        df_datos.to_excel(writer, sheet_name="Datos", index=False)
        
        # ── Hoja 2: Moran Global ──────────────────────────────────────────
        interp = interpretar_moran(
            resultado_global["indice_i"],
            resultado_global["p_valor_sim"],
            resultado_global["z_score"]
        )
        
        df_global = pd.DataFrame({
            "Parámetro": [
                "Dataset", "Variable analizada", "N unidades",
                "Índice de Moran I", "Valor esperado E[I]",
                "Varianza Var[I]", "Z-score (normalidad)",
                "p-valor (normalidad)", "p-valor (simulación)",
                "N permutaciones", "Patrón detectado",
                "Interpretación", "Fecha de análisis"
            ],
            "Valor": [
                nombre_dataset,
                resultado_global["columna"],
                resultado_global["n"],
                resultado_global["indice_i"],
                resultado_global["valor_esperado"],
                resultado_global["varianza"],
                resultado_global["z_score"],
                resultado_global["p_valor"],
                resultado_global["p_valor_sim"],
                resultado_global["n_sims"],
                interp["patron"],
                interp["descripcion"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        })
        df_global.to_excel(writer, sheet_name="Moran Global", index=False)
        
        # ── Hoja 3: LISA ──────────────────────────────────────────────────
        if resultado_lisa:
            df_lisa = pd.DataFrame({
                "Unidad":   range(len(resultado_lisa["Is"])),
                "LISA_I":   resultado_lisa["Is"],
                "p_valor":  resultado_lisa["p_sim"],
                "z_score":  resultado_lisa["z_sim"],
                "Cluster":  resultado_lisa["cuadrante"],
                "y_std":    resultado_lisa["y_std"],
                "Lag_std":  resultado_lisa["lag_std"],
            })
            df_lisa.to_excel(writer, sheet_name="LISA", index=False)
            
            # Sub-hoja de resumen LISA
            resumen_lisa = pd.DataFrame({
                "Tipo Cluster": ["High-High (HH)", "Low-Low (LL)", 
                                  "High-Low (HL)", "Low-High (LH)", 
                                  "No Significativo (NS)", "TOTAL"],
                "N Unidades": [
                    resultado_lisa["n_HH"], resultado_lisa["n_LL"],
                    resultado_lisa["n_HL"], resultado_lisa["n_LH"],
                    resultado_lisa["n_NS"], resultado_lisa["n_total"]
                ],
                "% del Total": [
                    f"{resultado_lisa['n_HH']/resultado_lisa['n_total']*100:.1f}%",
                    f"{resultado_lisa['n_LL']/resultado_lisa['n_total']*100:.1f}%",
                    f"{resultado_lisa['n_HL']/resultado_lisa['n_total']*100:.1f}%",
                    f"{resultado_lisa['n_LH']/resultado_lisa['n_total']*100:.1f}%",
                    f"{resultado_lisa['n_NS']/resultado_lisa['n_total']*100:.1f}%",
                    "100.0%"
                ],
            })
            resumen_lisa.to_excel(writer, sheet_name="Resumen LISA", index=False)
        
        # ── Hoja 4: Estadísticas descriptivas ─────────────────────────────
        columna = resultado_global["columna"]
        if columna in gdf.columns:
            stats = calcular_estadisticas(gdf[columna])
            df_stats = pd.DataFrame({
                "Estadístico": [
                    "N (válidos)", "Media", "Mediana", "Moda",
                    "Desviación estándar", "Varianza", "Mínimo", "Máximo",
                    "Rango", "Q1 (25%)", "Q3 (75%)", "IQR",
                    "Coef. asimetría", "Curtosis", "Coef. variación (%)", "Nulos"
                ],
                "Valor": [
                    stats["n"], stats["media"], stats["mediana"], stats["moda"],
                    stats["desv_std"], stats["varianza"], stats["minimo"], stats["maximo"],
                    stats["rango"], stats["q1"], stats["q3"], stats["iqr"],
                    stats["asimetria"], stats["curtosis"], stats["cv"], stats["nulos"]
                ]
            })
            df_stats.to_excel(writer, sheet_name="Estadísticas", index=False)
        
        # ── Formato visual básico ──────────────────────────────────────────
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            # Ajustar ancho de columnas automáticamente
            for col_cells in ws.columns:
                max_len = max((len(str(cell.value or "")) for cell in col_cells), default=10)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 50)
    
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# EXPORTAR A CSV
# -----------------------------------------------------------------------------

def exportar_csv_datos(gdf: gpd.GeoDataFrame, 
                        resultado_lisa: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Exporta los datos con los resultados LISA adjuntos como CSV.
    """
    df = gdf[[c for c in gdf.columns if c != "geometry"]].copy()
    
    if resultado_lisa:
        n = min(len(df), len(resultado_lisa["Is"]))
        df.loc[:n-1, "LISA_I"]       = resultado_lisa["Is"][:n]
        df.loc[:n-1, "LISA_p"]       = resultado_lisa["p_sim"][:n]
        df.loc[:n-1, "LISA_Cluster"] = resultado_lisa["cuadrante"][:n]
    
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def exportar_csv_resumen(resultados: Dict[str, Dict[str, Any]]) -> bytes:
    """
    Exporta la tabla de comparación entre datasets como CSV.
    """
    from modules.mapas import tabla_comparacion
    df = tabla_comparacion(resultados)
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


# -----------------------------------------------------------------------------
# EXPORTAR A PDF (Reporte estadístico)
# -----------------------------------------------------------------------------

def exportar_pdf(resultado_global: Dict[str, Any],
                  resultado_lisa: Optional[Dict[str, Any]],
                  nombre_dataset: str,
                  stats_descriptivas: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Genera un reporte PDF profesional con los resultados del análisis.
    
    El reporte incluye:
        - Portada con metadatos
        - Resumen ejecutivo
        - Estadísticas descriptivas
        - Resultados Moran Global
        - Resultados LISA
        - Interpretación y conclusiones
    """
    try:
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable,
                                         PageBreak, KeepTogether)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from modules.utils import interpretar_moran, formatear_numero
        
    except ImportError:
        st.error("❌ reportlab no está instalado. Ejecute: pip install reportlab")
        return b""
    
    buffer = io.BytesIO()
    
    # ── Configuración del documento ────────────────────────────────────────
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        title=f"Análisis Moran - {nombre_dataset}",
        author="Sistema de Análisis de Autocorrelación Espacial"
    )
    
    # ── Estilos ───────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    
    style_titulo = ParagraphStyle(
        "Titulo", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#1B4F72"),
        spaceAfter=6, alignment=TA_CENTER,
    )
    style_subtitulo = ParagraphStyle(
        "Subtitulo", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#2E86AB"),
        spaceAfter=4, spaceBefore=12,
    )
    style_normal = ParagraphStyle(
        "Normal2", parent=styles["Normal"],
        fontSize=10, spaceAfter=4, alignment=TA_JUSTIFY,
    )
    style_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=colors.gray,
        spaceAfter=2, alignment=TA_CENTER,
    )
    
    # ── Colores de tabla ──────────────────────────────────────────────────
    AZUL_ENCAB  = colors.HexColor("#1B4F72")
    AZUL_CLARO  = colors.HexColor("#D6EAF8")
    
    def tabla_estilo_base(n_cols: int) -> TableStyle:
        return TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  AZUL_ENCAB),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0),  9),
            ("ALIGN",       (0,0), (-1,0),  "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, AZUL_CLARO]),
            ("FONTSIZE",    (0,1), (-1,-1), 9),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING",(0,0), (-1,-1), 6),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ])
    
    # ── Contenido del documento ────────────────────────────────────────────
    story = []
    fecha = datetime.now().strftime("%d de %B de %Y, %H:%M")
    interp = interpretar_moran(
        resultado_global["indice_i"],
        resultado_global["p_valor_sim"],
        resultado_global["z_score"]
    )
    
    # --- PORTADA ---
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("REPORTE DE ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL", style_titulo))
    story.append(Paragraph("Índice de Moran Global y Local (LISA)", style_subtitulo))
    story.append(HRFlowable(width="100%", thickness=2, color=AZUL_ENCAB))
    story.append(Spacer(1, 0.5*cm))
    
    meta = [
        ["Dataset:", nombre_dataset],
        ["Variable analizada:", resultado_global["columna"]],
        ["N unidades:", str(resultado_global["n"])],
        ["Permutaciones:", str(resultado_global["n_sims"])],
        ["Fecha:", fecha],
    ]
    t_meta = Table(meta, colWidths=[5*cm, 11*cm])
    t_meta.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t_meta)
    story.append(PageBreak())
    
    # --- SECCIÓN 1: RESULTADOS MORAN GLOBAL ---
    story.append(Paragraph("1. Índice de Moran Global", style_subtitulo))
    story.append(HRFlowable(width="100%", thickness=1, color=AZUL_ENCAB))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph(
        f"<b>Patrón detectado:</b> {interp['icono']} {interp['patron']}",
        style_normal
    ))
    story.append(Paragraph(interp["descripcion"], style_normal))
    story.append(Spacer(1, 0.4*cm))
    
    datos_global = [
        ["Estadístico", "Valor", "Interpretación"],
        ["Índice de Moran I", formatear_numero(resultado_global["indice_i"]),
         "> 0: Agrupam. | < 0: Dispersión | ≈ 0: Aleatorio"],
        ["Valor esperado E[I]", formatear_numero(resultado_global["valor_esperado"]),
         f"Teórico: {formatear_numero(resultado_global['ei_teorico'])}"],
        ["Z-score", formatear_numero(resultado_global["z_score"]),
         "|Z| > 1.96 → Significativo (α=0.05)"],
        ["p-valor (normalidad)", formatear_numero(resultado_global["p_valor"]),
         "< 0.05 → Significativo"],
        ["p-valor (simulación)", formatear_numero(resultado_global["p_valor_sim"]),
         "Más robusto que el teórico"],
        ["Varianza", formatear_numero(resultado_global["varianza"]), ""],
        ["N unidades", str(resultado_global["n"]), ""],
    ]
    
    t_global = Table(datos_global, colWidths=[5.5*cm, 3.5*cm, 7*cm])
    t_global.setStyle(tabla_estilo_base(3))
    story.append(t_global)
    story.append(Spacer(1, 0.5*cm))
    
    # --- SECCIÓN 2: RESULTADOS LISA ---
    if resultado_lisa:
        story.append(Paragraph("2. Análisis LISA (Moran Local)", style_subtitulo))
        story.append(HRFlowable(width="100%", thickness=1, color=AZUL_ENCAB))
        story.append(Spacer(1, 0.3*cm))
        
        n_sig = resultado_lisa["n_significativo"]
        pct_sig = n_sig / resultado_lisa["n_total"] * 100
        
        story.append(Paragraph(
            f"Se identificaron <b>{n_sig} unidades significativas</b> "
            f"({pct_sig:.1f}% del total) distribuidas en los siguientes clusters:",
            style_normal
        ))
        story.append(Spacer(1, 0.3*cm))
        
        datos_lisa = [
            ["Cluster", "N Unidades", "% del Total", "Descripción"],
            ["HH (Alto-Alto)", resultado_lisa["n_HH"],
             f"{resultado_lisa['n_HH']/resultado_lisa['n_total']*100:.1f}%",
             "Hot Spot: valores altos rodeados de valores altos"],
            ["LL (Bajo-Bajo)", resultado_lisa["n_LL"],
             f"{resultado_lisa['n_LL']/resultado_lisa['n_total']*100:.1f}%",
             "Cold Spot: valores bajos rodeados de valores bajos"],
            ["HL (Alto-Bajo)", resultado_lisa["n_HL"],
             f"{resultado_lisa['n_HL']/resultado_lisa['n_total']*100:.1f}%",
             "Outlier: valor alto rodeado de valores bajos"],
            ["LH (Bajo-Alto)", resultado_lisa["n_LH"],
             f"{resultado_lisa['n_LH']/resultado_lisa['n_total']*100:.1f}%",
             "Outlier: valor bajo rodeado de valores altos"],
            ["NS (No signif.)", resultado_lisa["n_NS"],
             f"{resultado_lisa['n_NS']/resultado_lisa['n_total']*100:.1f}%",
             "Sin significancia estadística"],
            ["TOTAL", resultado_lisa["n_total"], "100%", ""],
        ]
        
        t_lisa = Table(datos_lisa, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 7.5*cm])
        t_lisa.setStyle(tabla_estilo_base(4))
        story.append(t_lisa)
        story.append(Spacer(1, 0.5*cm))
    
    # --- SECCIÓN 3: ESTADÍSTICAS DESCRIPTIVAS ---
    if stats_descriptivas:
        story.append(Paragraph("3. Estadísticas Descriptivas", style_subtitulo))
        story.append(HRFlowable(width="100%", thickness=1, color=AZUL_ENCAB))
        story.append(Spacer(1, 0.3*cm))
        
        datos_stats = [
            ["Estadístico", "Valor"],
            ["N válidos",          str(stats_descriptivas.get("n", "N/A"))],
            ["Media",              formatear_numero(stats_descriptivas.get("media"))],
            ["Mediana",            formatear_numero(stats_descriptivas.get("mediana"))],
            ["Desviación estándar",formatear_numero(stats_descriptivas.get("desv_std"))],
            ["Mínimo",             formatear_numero(stats_descriptivas.get("minimo"))],
            ["Máximo",             formatear_numero(stats_descriptivas.get("maximo"))],
            ["Q1 (25%)",           formatear_numero(stats_descriptivas.get("q1"))],
            ["Q3 (75%)",           formatear_numero(stats_descriptivas.get("q3"))],
            ["Coef. asimetría",    formatear_numero(stats_descriptivas.get("asimetria"))],
            ["Curtosis",           formatear_numero(stats_descriptivas.get("curtosis"))],
        ]
        
        t_stats = Table(datos_stats, colWidths=[7*cm, 9*cm])
        t_stats.setStyle(tabla_estilo_base(2))
        story.append(t_stats)
    
    # --- PIE DE PÁGINA ---
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        f"Reporte generado automáticamente el {fecha} | "
        "Sistema de Análisis de Autocorrelación Espacial — Índice de Moran",
        style_caption
    ))
    
    # Construir el PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# BOTONES DE DESCARGA EN STREAMLIT
# -----------------------------------------------------------------------------

def botones_descarga(gdf: gpd.GeoDataFrame, resultado_global: Dict[str, Any],
                      resultado_lisa: Optional[Dict[str, Any]],
                      nombre_dataset: str):
    """
    Renderiza los botones de descarga de Excel, CSV y PDF en Streamlit.
    """
    from modules.utils import calcular_estadisticas
    
    st.markdown("### 📥 Exportar Resultados")
    col1, col2, col3 = st.columns(3)
    
    columna = resultado_global["columna"]
    stats = calcular_estadisticas(gdf[columna]) if columna in gdf.columns else None
    
    # ── Excel ────────────────────────────────────────────────────────────
    with col1:
        try:
            xlsx_bytes = exportar_excel(gdf, resultado_global, resultado_lisa, nombre_dataset)
            st.download_button(
                label="📊 Descargar Excel",
                data=xlsx_bytes,
                file_name=f"moran_{nombre_dataset.replace(' ','_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error Excel: {e}")
    
    # ── CSV ───────────────────────────────────────────────────────────────
    with col2:
        try:
            csv_bytes = exportar_csv_datos(gdf, resultado_lisa)
            st.download_button(
                label="📄 Descargar CSV",
                data=csv_bytes,
                file_name=f"moran_{nombre_dataset.replace(' ','_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error CSV: {e}")
    
    # ── PDF ───────────────────────────────────────────────────────────────
    with col3:
        try:
            pdf_bytes = exportar_pdf(resultado_global, resultado_lisa, nombre_dataset, stats)
            if pdf_bytes:
                st.download_button(
                    label="📑 Descargar PDF",
                    data=pdf_bytes,
                    file_name=f"reporte_moran_{nombre_dataset.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error PDF: {e}")
