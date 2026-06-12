# =============================================================
#  app.py — HCVpredict · Dashboard de Triagem Preditiva
#  Comando para rodar: streamlit run app.py
# =============================================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (roc_curve, confusion_matrix,
                             roc_auc_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

# ── Configuração da página ─────────────────────────────────────
st.set_page_config(
    page_title="HCVpredict · HEMOPA",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estilo CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0B1120; color: #E8EDF5; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: #162032;
        border: 1px solid #1E3050;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.75rem; color: #6B7A99;
                    text-transform: uppercase; letter-spacing: 0.08em; }
    .risk-box {
        border-radius: 10px;
        padding: 14px 18px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-testid="stTabs"] button {
        color: #6B7A99 !important;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #E05252 !important;
        border-bottom-color: #E05252 !important;
    }
    .section-title {
        font-size: 0.72rem;
        color: #6B7A99;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Carregar modelo e arquivos ─────────────────────────────────
@st.cache_resource
def carregar_modelo():
    with open("modelo_hcv.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("metricas.pkl", "rb") as f:
        metricas = pickle.load(f)
    df = pd.read_csv("dados_simulados_hcv.csv")
    return modelo, encoders, metricas, df

try:
    modelo, encoders, metricas, df = carregar_modelo()
    modelo_ok = True
except FileNotFoundError:
    modelo_ok = False

# ── Header ─────────────────────────────────────────────────────
col_logo, col_title, col_info = st.columns([1, 6, 3])
with col_logo:
    st.markdown("## 🩸")
with col_title:
    st.markdown("## HCV**predict** · HEMOPA")
    st.caption("Triagem preditiva de resultado NAT-HCV em doadores de sangue")
with col_info:
    if modelo_ok:
        st.markdown(f"""
        <div style='text-align:right; color:#6B7A99; font-size:0.78rem; margin-top:10px'>
        Modelo: Gradient Boosting &nbsp;|&nbsp;
        AUC: <b style='color:#2DD4BF'>{metricas['auc']}</b> &nbsp;|&nbsp;
        Dados: simulados (protótipo)
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

if not modelo_ok:
    st.error("⚠️ Arquivos do modelo não encontrados. Execute primeiro: `python gerar_modelo.py`")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Predição Individual",
    "📊  Desempenho dos Modelos",
    "📋  Análise de Lote"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDIÇÃO INDIVIDUAL
# ══════════════════════════════════════════════════════════════
with tab1:

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:

        # Epidemiológicos
        st.markdown('<div class="section-title">Dados Epidemiológicos</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            idade = st.slider("Idade", 18, 70, 35)
        with c2:
            sexo = st.selectbox("Sexo biológico", ["M", "F"],
                                format_func=lambda x: "Masculino" if x == "M" else "Feminino")

        c3, c4 = st.columns(2)
        with c3:
            tipo_doador = st.selectbox("Tipo de doador",
                ["Espontaneo", "Reposicao", "Autolog"],
                format_func=lambda x: {
                    "Espontaneo": "Espontâneo",
                    "Reposicao":  "Reposição",
                    "Autolog":    "Autólogo"
                }[x])
        with c4:
            escolaridade = st.selectbox("Escolaridade",
                ["Fundamental", "Médio", "Superior"])

        raca = st.selectbox("Raça autodeclarada",
            ["Parda", "Branca", "Preta", "Outras"])

        st.markdown("---")

        # Fatores de risco
        st.markdown('<div class="section-title">Fatores de Risco</div>',
                    unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5:
            drogas_iv  = st.toggle("Drogas injetáveis", value=False)
        with c6:
            tatuagem   = st.toggle("Tatuagem / piercing", value=False)
        with c7:
            transfusao = st.toggle("Transfusão prévia", value=False)

        st.markdown("---")

        # Sorológicos
        st.markdown('<div class="section-title">Resultados Sorológicos e Moleculares</div>',
                    unsafe_allow_html=True)
        c8, c9 = st.columns(2)
        with c8:
            anti_hcv = st.selectbox("Anti-HCV (ECLIA/CLIA)",
                ["Nao_Reagente", "Indeterminado", "Reagente"],
                format_func=lambda x: {
                    "Nao_Reagente":  "Não reagente",
                    "Indeterminado": "Indeterminado",
                    "Reagente":      "Reagente"
                }[x])
        with c9:
            western_blot = st.selectbox("Western Blot / Imunoblot",
                ["Negativo", "Indeterminado", "Positivo"])

        pcr = st.selectbox("PCR em tempo real",
            ["Nao_Detectado", "Detectado"],
            format_func=lambda x: "Não detectado" if x == "Nao_Detectado" else "Detectado")

        st.markdown("")
        predizer = st.button("🔬 Predizer NAT-HCV", use_container_width=True,
                             type="primary")

    # ── Resultado ──────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-title">Resultado da Predição</div>',
                    unsafe_allow_html=True)

        if predizer:
            # Montar entrada
            entrada = pd.DataFrame([{
                "idade":        idade,
                "sexo":         encoders["sexo"].transform([sexo])[0],
                "escolaridade": encoders["escolaridade"].transform([escolaridade])[0],
                "tipo_doador":  encoders["tipo_doador"].transform([tipo_doador])[0],
                "raca":         encoders["raca"].transform([raca])[0],
                "drogas_iv":    int(drogas_iv),
                "tatuagem":     int(tatuagem),
                "transfusao":   int(transfusao),
                "anti_hcv":     encoders["anti_hcv"].transform([anti_hcv])[0],
                "western_blot": encoders["western_blot"].transform([western_blot])[0],
                "pcr":          encoders["pcr"].transform([pcr])[0],
            }])

            prob = modelo.predict_proba(entrada)[0][1]
            pct  = round(prob * 100, 1)

            # Classificação
            if pct < 25:
                risco, cor, emoji = "BAIXO",  "#22C55E", "✅"
                msg = "Perfil compatível com doador apto. Seguir protocolo padrão de triagem."
                bg  = "#052E16"
            elif pct < 60:
                risco, cor, emoji = "MÉDIO",  "#F59E0B", "⚡"
                msg = "Perfil de risco moderado. Considerar repetição da triagem sorológica e monitoramento."
                bg  = "#2D1F00"
            else:
                risco, cor, emoji = "ALTO",   "#EF4444", "⚠️"
                msg = "Alto risco de NAT positivo. Encaminhar para teste confirmatório e avaliação clínica especializada."
                bg  = "#3B1A1A"

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 40, "color": cor}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6B7A99",
                             "tickfont": {"color": "#6B7A99"}},
                    "bar":  {"color": cor, "thickness": 0.25},
                    "bgcolor": "#162032",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  25], "color": "#052E16"},
                        {"range": [25, 60], "color": "#2D1F00"},
                        {"range": [60, 100],"color": "#3B1A1A"},
                    ],
                    "threshold": {
                        "line": {"color": cor, "width": 4},
                        "thickness": 0.8,
                        "value": pct
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                paper_bgcolor="#0B1120",
                font_color="#E8EDF5",
                margin=dict(t=20, b=10, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risco badge
            st.markdown(f"""
            <div style='text-align:center; margin: -10px 0 16px'>
                <span style='background:{bg}; color:{cor}; border-radius:8px;
                    padding:6px 20px; font-weight:800; font-size:1rem;
                    letter-spacing:0.1em'>{emoji} RISCO {risco}</span>
            </div>
            """, unsafe_allow_html=True)

            # Probabilidade
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                background:#162032; border:1px solid #1E3050;
                border-radius:10px; padding:12px 18px; margin-bottom:12px'>
                <span style='color:#6B7A99'>Probabilidade de NAT positivo</span>
                <span style='color:{cor}; font-weight:800; font-size:1.1rem'>{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Barra de progresso
            st.progress(int(pct))

            # Recomendação
            st.markdown(f"""
            <div class='risk-box' style='background:{bg}; color:{cor};
                border-left: 3px solid {cor}; margin-top:12px'>
                {msg}
            </div>
            """, unsafe_allow_html=True)

            # Importância das variáveis
            st.markdown("---")
            st.markdown('<div class="section-title">Variáveis mais influentes</div>',
                        unsafe_allow_html=True)
            feat_imp = pd.Series(
                modelo.feature_importances_,
                index=metricas["feature_names"]
            ).sort_values(ascending=True).tail(6)

            fig_imp = go.Figure(go.Bar(
                x=feat_imp.values,
                y=feat_imp.index,
                orientation="h",
                marker_color="#E05252",
                marker_line_width=0,
            ))
            fig_imp.update_layout(
                height=200,
                paper_bgcolor="#0B1120",
                plot_bgcolor="#162032",
                font_color="#E8EDF5",
                margin=dict(t=5, b=5, l=5, r=5),
                xaxis=dict(showgrid=False, color="#6B7A99"),
                yaxis=dict(color="#6B7A99"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:80px 20px; color:#6B7A99'>
                <div style='font-size:3rem'>🩸</div>
                <div style='margin-top:12px; font-size:0.9rem'>
                    Preencha os dados do doador<br>e clique em <b>Predizer NAT-HCV</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DESEMPENHO DOS MODELOS
# ══════════════════════════════════════════════════════════════
with tab2:

    # Recalcular métricas com o conjunto de teste
    @st.cache_data
    def calcular_curvas(_modelo, _df):
        from sklearn.preprocessing import LabelEncoder
        df2 = _df.copy()
        cat_cols = ["sexo","escolaridade","tipo_doador",
                    "raca","anti_hcv","western_blot","pcr"]
        for col in cat_cols:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col])
        X = df2.drop("nat_positivo", axis=1)
        y = df2["nat_positivo"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        y_prob = _modelo.predict_proba(X_test)[:, 1]
        y_pred = _modelo.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        return fpr, tpr, cm, auc, f1, prec, rec

    fpr, tpr, cm, auc, f1, prec, rec = calcular_curvas(modelo, df)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, cor in [
        (k1, f"{auc:.3f}", "AUC-ROC",       "#2DD4BF"),
        (k2, f"{f1:.3f}",  "F1-Score",      "#E05252"),
        (k3, f"{rec:.1%}", "Sensibilidade",  "#F59E0B"),
        (k4, f"{prec:.1%}","Precisão",       "#22C55E"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{cor}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    c_roc, c_cm = st.columns(2, gap="large")

    with c_roc:
        st.markdown('<div class="section-title">Curva AUROC</div>',
                    unsafe_allow_html=True)

        # Linha de referência aleatória
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="#6B7A99", dash="dash", width=1))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Gradient Boosting (AUC = {auc:.3f})",
            line=dict(color="#2DD4BF", width=2.5),
            fill="tozeroy", fillcolor="rgba(45,212,191,0.08)"
        ))
        fig_roc.update_layout(
            height=320,
            paper_bgcolor="#0B1120", plot_bgcolor="#162032",
            font_color="#E8EDF5",
            xaxis=dict(title="1 - Especificidade (FPR)",
                       color="#6B7A99", gridcolor="#1E3050"),
            yaxis=dict(title="Sensibilidade (TPR)",
                       color="#6B7A99", gridcolor="#1E3050"),
            legend=dict(bgcolor="#162032", bordercolor="#1E3050",
                        borderwidth=1, font=dict(size=11)),
            margin=dict(t=10, b=40, l=50, r=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_cm:
        st.markdown('<div class="section-title">Matriz de Confusão</div>',
                    unsafe_allow_html=True)

        labels = ["NAT Negativo", "NAT Positivo"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels, y=labels,
            colorscale=[[0, "#162032"], [1, "#E05252"]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22, "color": "white"}
        ))
        fig_cm.update_layout(
            height=320,
            paper_bgcolor="#0B1120", plot_bgcolor="#162032",
            font_color="#E8EDF5",
            xaxis=dict(title="Predito", color="#6B7A99"),
            yaxis=dict(title="Real",    color="#6B7A99"),
            margin=dict(t=10, b=50, l=70, r=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Importância global das variáveis
    st.markdown('<div class="section-title">Importância Global das Variáveis</div>',
                unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "variavel": metricas["feature_names"],
        "importancia": modelo.feature_importances_
    }).sort_values("importancia", ascending=False)

    fig_feat = px.bar(feat_df, x="variavel", y="importancia",
                      color="importancia",
                      color_continuous_scale=["#1E3050", "#E05252"])
    fig_feat.update_layout(
        height=260,
        paper_bgcolor="#0B1120", plot_bgcolor="#162032",
        font_color="#E8EDF5",
        coloraxis_showscale=False,
        xaxis=dict(color="#6B7A99", gridcolor="#1E3050"),
        yaxis=dict(color="#6B7A99", gridcolor="#1E3050"),
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISE DE LOTE
# ══════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div style='background:#162032; border:1px dashed #1E3050;
        border-radius:12px; padding:20px; margin-bottom:20px'>
        <b>📂 Carregar planilha de doadores</b><br>
        <span style='color:#6B7A99; font-size:0.85rem'>
        Formato aceito: CSV com as colunas do modelo.
        Os dados são processados localmente e não são armazenados.
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Selecionar arquivo CSV",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded:
        try:
            df_lote = pd.read_csv(uploaded)

            # Codificar e predizer
            df_pred = df_lote.copy()
            cat_cols = ["sexo","escolaridade","tipo_doador",
                        "raca","anti_hcv","western_blot","pcr"]
            for col in cat_cols:
                if col in df_pred.columns:
                    df_pred[col] = encoders[col].transform(df_pred[col])

            X_lote = df_pred[metricas["feature_names"]]
            probs   = modelo.predict_proba(X_lote)[:, 1]

            df_lote["prob_nat_positivo"] = (probs * 100).round(1)
            df_lote["risco"] = pd.cut(probs,
                bins=[0, 0.25, 0.60, 1.01],
                labels=["BAIXO", "MÉDIO", "ALTO"])

            # Resumo
            r1, r2, r3, r4 = st.columns(4)
            for col, val, label, cor in [
                (r1, len(df_lote), "Total de doadores", "#6B7A99"),
                (r2, (df_lote["risco"]=="BAIXO").sum(), "Risco Baixo",  "#22C55E"),
                (r3, (df_lote["risco"]=="MÉDIO").sum(), "Risco Médio",  "#F59E0B"),
                (r4, (df_lote["risco"]=="ALTO").sum(),  "Risco Alto",   "#EF4444"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{cor}'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Tabela
            st.markdown('<div class="section-title">Resultado por doador</div>',
                        unsafe_allow_html=True)

            def colorir(val):
                cores = {"ALTO":  "background-color:#3B1A1A; color:#EF4444",
                         "MÉDIO": "background-color:#2D1F00; color:#F59E0B",
                         "BAIXO": "background-color:#052E16; color:#22C55E"}
                return cores.get(val, "")

            st.dataframe(
                df_lote.style.applymap(colorir, subset=["risco"]),
                use_container_width=True, height=350
            )

            # Download
            csv_out = df_lote.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Exportar resultados (.csv)",
                data=csv_out,
                file_name="predicoes_nat_hcv.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

    else:
        # Demo com amostra do dataset simulado
        st.info("Nenhum arquivo carregado. Exibindo amostra dos dados simulados como demonstração.")
        amostra = df.sample(10, random_state=7).copy()

        df_demo = amostra.copy()
        from sklearn.preprocessing import LabelEncoder
        cat_cols = ["sexo","escolaridade","tipo_doador",
                    "raca","anti_hcv","western_blot","pcr"]
        df_enc = df_demo.copy()
        for col in cat_cols:
            df_enc[col] = encoders[col].transform(df_enc[col])

        X_demo = df_enc[metricas["feature_names"]]
        probs_demo = modelo.predict_proba(X_demo)[:, 1]

        amostra["prob_nat_positivo (%)"] = (probs_demo * 100).round(1)
        amostra["risco"] = pd.cut(probs_demo,
            bins=[0, 0.25, 0.60, 1.01],
            labels=["BAIXO", "MÉDIO", "ALTO"])

        def colorir(val):
            cores = {"ALTO":  "background-color:#3B1A1A; color:#EF4444",
                     "MÉDIO": "background-color:#2D1F00; color:#F59E0B",
                     "BAIXO": "background-color:#052E16; color:#22C55E"}
            return cores.get(val, "")

        st.dataframe(
            amostra.style.applymap(colorir, subset=["risco"]),
            use_container_width=True, height=350
        )
