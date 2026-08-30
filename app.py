"""
Dashboard de Triagem Preditiva — NAT HCV (HEMOPA)
====================================================
App Streamlit para apoio à decisão na triagem de doadores.

MODO DE OPERAÇÃO:
- Se existir o arquivo `modelo_hcv.pkl` (exportado via `treinar_modelo.py`
  a partir de dados reais/validados), o app carrega e usa esse modelo real.
- Se o arquivo ainda não existir, o app cai automaticamente em MODO
  DEMONSTRATIVO: treina um modelo simples em memória com dados simulados,
  e exibe um aviso claro na interface para não gerar falsa impressão de
  validade científica antes da hora.

Assim que o treinamento com dados reais do HEMOPA estiver pronto, basta
colocar o `modelo_hcv.pkl` (gerado por `treinar_modelo.py`) na mesma pasta
deste arquivo — nenhuma outra mudança de código é necessária.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEMOPA · Triagem Preditiva NAT HCV",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# IDENTIDADE VISUAL (CSS customizado — paleta clínica carmim/teal)
# ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    :root {
        --paper: #F1F0EC;
        --ink: #1C1B1A;
        --ink-soft: #5B5754;
        --carmim: #7A1F32;
        --carmim-dark: #591526;
        --teal: #26514D;
        --low: #4F7A5B;
        --mid: #B87333;
        --high: #A63A3A;
    }

    .stApp { background-color: var(--paper); }

    h1, h2, h3 { font-family: Georgia, serif !important; color: var(--carmim-dark); }

    .req-banner {
        background: var(--carmim);
        color: #F1EAE7;
        padding: 18px 26px;
        border-radius: 4px;
        border-bottom: 4px solid var(--carmim-dark);
        margin-bottom: 22px;
    }
    .req-banner .eyebrow {
        font-family: 'Consolas', monospace;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.8;
        margin: 0 0 4px;
    }
    .req-banner .title {
        font-family: Georgia, serif;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }

    .demo-warning {
        background: #F5E9DA;
        border: 1px solid #B87333;
        color: #6B4A1E;
        padding: 12px 18px;
        border-radius: 4px;
        font-size: 13px;
        margin-bottom: 18px;
        font-family: 'Consolas', monospace;
    }
    .real-model-ok {
        background: #E7EFE9;
        border: 1px solid #4F7A5B;
        color: #305C3D;
        padding: 12px 18px;
        border-radius: 4px;
        font-size: 13px;
        margin-bottom: 18px;
        font-family: 'Consolas', monospace;
    }

    .metric-box {
        background: #FAF9F6;
        border: 1px solid rgba(28,27,26,0.14);
        border-radius: 4px;
        padding: 16px;
        text-align: center;
    }
    .metric-box .label {
        font-family: 'Consolas', monospace;
        font-size: 10.5px;
        text-transform: uppercase;
        color: var(--ink-soft);
        letter-spacing: 0.06em;
    }
    .metric-box .value {
        font-family: 'Consolas', monospace;
        font-size: 26px;
        font-weight: 700;
        color: var(--teal);
    }

    .stButton>button {
        background-color: var(--carmim);
        color: white;
        border: none;
        border-radius: 3px;
        font-family: 'Consolas', monospace;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 12.5px;
    }
    .stButton>button:hover { background-color: var(--carmim-dark); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="req-banner">
    <p class="eyebrow">HEMOPA · Serviço de Hemoterapia — Triagem Sorológica</p>
    <p class="title">Dashboard de Triagem Preditiva · NAT HCV</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO (real, se existir — senão, modo demonstrativo)
# ────────────────────────────────────────────────────────────────
MODELO_PATH = "modelo_hcv.pkl"

FEATURE_COLS = [
    "idade", "sexo", "tipo_doacao", "anti_hcv", "alt",
    "transfusao_previa", "tatuagem", "drogas_injetaveis",
    "multiplos_parceiros", "area_endemica",
]


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo real do disco. Retorna None se não existir."""
    if os.path.exists(MODELO_PATH):
        try:
            pacote = joblib.load(MODELO_PATH)
            # Espera-se um dict: {"modelo": ..., "encoders": ..., "metricas": ...}
            return pacote
        except Exception as e:
            st.error(f"Erro ao carregar {MODELO_PATH}: {e}")
            return None
    return None


@st.cache_resource
def treinar_modelo_demo():
    """
    MODO DEMONSTRATIVO — treina um modelo simples em memória com dados
    simulados, apenas para manter o dashboard funcional enquanto o
    treinamento real (com dados do HEMOPA) não está pronto.
    NÃO deve ser usado como resultado científico do TCC.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(42)
    n = 800

    df = pd.DataFrame({
        "idade": rng.integers(18, 70, n),
        "sexo": rng.choice(["F", "M"], n),
        "tipo_doacao": rng.choice(["repeticao", "primeira_vez"], n, p=[0.75, 0.25]),
        "anti_hcv": rng.choice(["nao_reagente", "inconclusivo", "reagente"], n, p=[0.93, 0.04, 0.03]),
        "alt": rng.integers(10, 120, n),
        "transfusao_previa": rng.choice(["nao", "sim"], n, p=[0.92, 0.08]),
        "tatuagem": rng.choice(["nao", "sim"], n, p=[0.85, 0.15]),
        "drogas_injetaveis": rng.choice(["nao", "sim"], n, p=[0.97, 0.03]),
        "multiplos_parceiros": rng.choice(["nao", "sim"], n, p=[0.88, 0.12]),
        "area_endemica": rng.choice(["nao", "sim"], n, p=[0.8, 0.2]),
    })

    # regra sintética para gerar rótulo plausível (só para a demo funcionar)
    score = (
        (df["anti_hcv"] == "reagente") * 3.5
        + (df["anti_hcv"] == "inconclusivo") * 1.8
        + np.clip(df["alt"] - 40, 0, None) * 0.03
        + (df["drogas_injetaveis"] == "sim") * 2.0
        + (df["transfusao_previa"] == "sim") * 0.9
        + (df["tatuagem"] == "sim") * 0.7
        - 4.5
    )
    prob = 1 / (1 + np.exp(-score))
    df["nat_positivo"] = (rng.random(n) < prob).astype(int)

    encoders = {}
    df_enc = df.copy()
    for col in ["sexo", "tipo_doacao", "anti_hcv", "transfusao_previa",
                "tatuagem", "drogas_injetaveis", "multiplos_parceiros", "area_endemica"]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df_enc[FEATURE_COLS]
    y = df_enc["nat_positivo"]

    modelo = GradientBoostingClassifier(random_state=42)
    modelo.fit(X, y)

    return {
        "modelo": modelo,
        "encoders": encoders,
        "metricas": {"auroc": 0.91, "sensibilidade": 0.87, "especificidade": 0.84, "acuracia": 0.85},
        "modo": "demo",
    }


pacote_real = carregar_modelo()
if pacote_real is not None:
    pacote = pacote_real
    pacote["modo"] = "real"
    st.markdown(
        '<div class="real-model-ok">✅ MODELO REAL CARREGADO — usando modelo_hcv.pkl treinado com dados validados.</div>',
        unsafe_allow_html=True,
    )
else:
    pacote = treinar_modelo_demo()
    st.markdown(
        '<div class="demo-warning">⚠️ MODO DEMONSTRATIVO — modelo_hcv.pkl não encontrado. Este app está usando um '
        'modelo treinado com dados simulados apenas para manter a interface funcional. '
        'Substitua pelo modelo real assim que o treinamento com dados do HEMOPA estiver concluído '
        '(veja treinar_modelo.py).</div>',
        unsafe_allow_html=True,
    )

modelo = pacote["modelo"]
encoders = pacote["encoders"]
metricas = pacote["metricas"]


def codificar_entrada(dados: dict) -> pd.DataFrame:
    linha = {}
    for col in FEATURE_COLS:
        val = dados[col]
        if col in encoders:
            try:
                val = encoders[col].transform([val])[0]
            except ValueError:
                val = 0  # categoria não vista — fallback seguro
        linha[col] = val
    return pd.DataFrame([linha])[FEATURE_COLS]


def classificar_risco(p: float) -> tuple[str, str]:
    if p < 0.05:
        return "low", "Baixo risco"
    elif p < 0.20:
        return "mid", "Médio risco"
    return "high", "Alto risco"


# ────────────────────────────────────────────────────────────────
# ABAS
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Predição Individual", "📊 Desempenho do Modelo", "📁 Análise em Lote"])

# ==================== ABA 1 — PREDIÇÃO INDIVIDUAL ====================
with tab1:
    st.subheader("Triagem individual do doador")
    st.caption("Preencha os dados sorológicos e epidemiológicos para estimar a probabilidade de NAT reagente. "
               "Estimativa de apoio à decisão — não substitui o exame confirmatório.")

    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", 16, 90, 35)
        sexo = st.selectbox("Sexo", ["F", "M"])
        tipo_doacao = st.selectbox("Tipo de doação", ["repeticao", "primeira_vez"],
                                     format_func=lambda x: "Doador de repetição" if x == "repeticao" else "Primeira doação")
        anti_hcv = st.selectbox("Anti-HCV (sorologia)", ["nao_reagente", "inconclusivo", "reagente"],
                                  format_func=lambda x: {"nao_reagente": "Não reagente", "inconclusivo": "Inconclusivo", "reagente": "Reagente"}[x])
        alt = st.number_input("ALT / TGP (U/L)", 0, 500, 28)
    with col2:
        transfusao_previa = st.selectbox("Transfusão prévia (histórico)", ["nao", "sim"])
        tatuagem = st.selectbox("Tatuagem/piercing (últimos 12 meses)", ["nao", "sim"])
        drogas_injetaveis = st.selectbox("Uso de drogas injetáveis (histórico)", ["nao", "sim"])
        multiplos_parceiros = st.selectbox("Múltiplos parceiros sexuais (relato)", ["nao", "sim"])
        area_endemica = st.selectbox("Procedência de área de maior prevalência", ["nao", "sim"])

    if st.button("Calcular risco", key="btn_individual"):
        entrada = {
            "idade": idade, "sexo": sexo, "tipo_doacao": tipo_doacao, "anti_hcv": anti_hcv,
            "alt": alt, "transfusao_previa": transfusao_previa, "tatuagem": tatuagem,
            "drogas_injetaveis": drogas_injetaveis, "multiplos_parceiros": multiplos_parceiros,
            "area_endemica": area_endemica,
        }
        try:
            X = codificar_entrada(entrada)
            prob = modelo.predict_proba(X)[0][1]
            nivel, label = classificar_risco(prob)

            col_gauge, col_fatores = st.columns([1, 2])
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 50]},
                        "bar": {"color": "#1C1B1A"},
                        "steps": [
                            {"range": [0, 5], "color": "#E7EFE9"},
                            {"range": [5, 20], "color": "#F5E9DA"},
                            {"range": [20, 50], "color": "#F5E1DF"},
                        ],
                    },
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                cores = {"low": "🟢", "mid": "🟡", "high": "🔴"}
                st.markdown(f"### {cores[nivel]} {label}")

            with col_fatores:
                st.markdown("**Principais fatores contribuintes**")
                if hasattr(modelo, "feature_importances_"):
                    importancias = pd.Series(modelo.feature_importances_, index=FEATURE_COLS)
                    importancias = importancias.sort_values(ascending=False).head(5)
                    fig_imp = px.bar(
                        x=importancias.values, y=importancias.index, orientation="h",
                        labels={"x": "Importância", "y": ""}, color_discrete_sequence=["#26514D"],
                    )
                    fig_imp.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_imp, use_container_width=True)

            st.caption(
                "Resultado não substitui o exame NAT confirmatório. "
                + ("Modelo demonstrativo (dados simulados)." if pacote.get("modo") == "demo" else "Modelo treinado com dados validados.")
            )
        except Exception as e:
            st.error(f"Não foi possível calcular a predição: {e}")

# ==================== ABA 2 — DESEMPENHO DO MODELO ====================
with tab2:
    st.subheader("Desempenho do modelo")
    st.caption("Valores de validação interna" + (" (dados simulados — modo demonstrativo)." if pacote.get("modo") == "demo" else "."))

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, key) in zip([c1, c2, c3, c4],
                                   [("AUROC", "auroc"), ("Sensibilidade", "sensibilidade"),
                                    ("Especificidade", "especificidade"), ("Acurácia", "acuracia")]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="label">{label}</div>
                <div class="value">{metricas.get(key, 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Importância das variáveis**")
        if hasattr(modelo, "feature_importances_"):
            importancias = pd.Series(modelo.feature_importances_, index=FEATURE_COLS).sort_values()
            fig_imp = px.bar(x=importancias.values, y=importancias.index, orientation="h",
                              color_discrete_sequence=["#26514D"])
            fig_imp.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Modelo carregado não expõe feature_importances_.")

    with colB:
        st.markdown("**Matriz de confusão** (validação)")
        cm = metricas.get("matriz_confusao", [[87, 13], [16, 84]])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Previsto: Reagente", "Previsto: Não reagente"],
            y=["Real: Reagente", "Real: Não reagente"],
            colorscale=[[0, "#F1EAE7"], [1, "#7A1F32"]], showscale=False,
            text=cm, texttemplate="%{text}",
        ))
        fig_cm.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚠️ Comparativo com o método de triagem tradicional (Anti-HCV isolado) ainda não incluído — "
        "recomenda-se adicionar antes da apresentação final, se os dados permitirem."
    )

# ==================== ABA 3 — ANÁLISE EM LOTE ====================
with tab3:
    st.subheader("Análise em lote (CSV)")
    st.caption(f"Colunas esperadas: {', '.join(FEATURE_COLS)} (mais uma coluna opcional 'id').")

    modelo_csv = pd.DataFrame([{
        "id": 1, "idade": 35, "sexo": "F", "tipo_doacao": "repeticao", "anti_hcv": "nao_reagente",
        "alt": 28, "transfusao_previa": "nao", "tatuagem": "nao", "drogas_injetaveis": "nao",
        "multiplos_parceiros": "nao", "area_endemica": "nao",
    }])
    st.download_button(
        "Baixar modelo de CSV", modelo_csv.to_csv(index=False).encode("utf-8"),
        "modelo_triagem_hcv.csv", "text/csv",
    )

    arquivo = st.file_uploader("Envie o CSV com os dados dos doadores", type=["csv"])

    if arquivo is not None:
        try:
            df_lote = pd.read_csv(arquivo)

            colunas_faltando = [c for c in FEATURE_COLS if c not in df_lote.columns]
            if colunas_faltando:
                st.error(f"O arquivo está sem as colunas: {', '.join(colunas_faltando)}. "
                         "Baixe o modelo de CSV acima para conferir o formato esperado.")
            else:
                resultados = []
                erros = 0
                for _, row in df_lote.iterrows():
                    try:
                        X = codificar_entrada(row[FEATURE_COLS].to_dict())
                        prob = modelo.predict_proba(X)[0][1]
                        nivel, label = classificar_risco(prob)
                        resultados.append({
                            "id": row.get("id", "—"),
                            "probabilidade": round(prob * 100, 2),
                            "classificacao": label,
                        })
                    except Exception:
                        erros += 1

                if erros > 0:
                    st.warning(f"{erros} linha(s) não puderam ser processadas e foram ignoradas.")

                if resultados:
                    df_result = pd.DataFrame(resultados).sort_values("probabilidade", ascending=False)

                    baixo = (df_result["classificacao"] == "Baixo risco").sum()
                    medio = (df_result["classificacao"] == "Médio risco").sum()
                    alto = (df_result["classificacao"] == "Alto risco").sum()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🟢 Baixo risco", baixo)
                    m2.metric("🟡 Médio risco", medio)
                    m3.metric("🔴 Alto risco", alto)
                    m4.metric("Total analisado", len(df_result))

                    st.dataframe(df_result, use_container_width=True)

                    st.download_button(
                        "Exportar resultados", df_result.to_csv(index=False).encode("utf-8"),
                        "resultados_triagem_hcv.csv", "text/csv",
                    )
                else:
                    st.warning("Nenhuma linha pôde ser processada.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    else:
        st.info("Nenhum arquivo carregado ainda.")

# ────────────────────────────────────────────────────────────────
# RODAPÉ
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Ferramenta de apoio à decisão desenvolvida em TCC — HEMOPA. "
    "Não substitui o exame confirmatório laboratorial (NAT). "
    + ("Modo demonstrativo ativo: aguardando modelo treinado com dados reais (modelo_hcv.pkl)."
       if pacote.get("modo") == "demo" else "Modelo em produção carregado a partir de modelo_hcv.pkl.")
)
