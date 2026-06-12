# =============================================================
#  gerar_modelo.py
#  Gera dados simulados baseados na literatura e treina o modelo
#  Execute este arquivo UMA VEZ antes de rodar o dashboard
#  Comando: python gerar_modelo.py
# =============================================================

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N = 2000  # número de doadores simulados

print("=" * 55)
print("  HCVpredict · Gerador de Dados Simulados + Modelo")
print("=" * 55)

# -------------------------------------------------------------
# 1. GERAR DADOS SIMULADOS
#    Distribuições baseadas na literatura (HEMOPA / Frazão 2025)
# -------------------------------------------------------------

print("\n[1/4] Gerando dados simulados...")

# Dados epidemiológicos
idade       = np.random.normal(loc=38, scale=12, size=N).clip(18, 70).astype(int)
sexo        = np.random.choice(["M", "F"], size=N, p=[0.62, 0.38])
escolaridade= np.random.choice(
                ["Fundamental", "Médio", "Superior"],
                size=N, p=[0.25, 0.50, 0.25])
tipo_doador = np.random.choice(
                ["Espontaneo", "Reposicao", "Autolog"],
                size=N, p=[0.70, 0.25, 0.05])
raca        = np.random.choice(
                ["Parda", "Branca", "Preta", "Outras"],
                size=N, p=[0.55, 0.25, 0.15, 0.05])

# Fatores de risco
drogas_iv   = np.random.choice([0, 1], size=N, p=[0.92, 0.08])
tatuagem    = np.random.choice([0, 1], size=N, p=[0.75, 0.25])
transfusao  = np.random.choice([0, 1], size=N, p=[0.88, 0.12])

# Resultados sorológicos — influenciados pelos fatores de risco
# Probabilidade base de anti-HCV reagente: ~3% (literatura HEMOPA)
prob_reagente = (
    0.03
    + drogas_iv   * 0.30
    + tatuagem    * 0.05
    + transfusao  * 0.08
    + (idade > 45).astype(int) * 0.04
    + (sexo == "M").astype(int) * 0.02
    + (tipo_doador == "Reposicao").astype(int) * 0.04
).clip(0, 0.95)

anti_hcv_num = np.random.binomial(1, prob_reagente)
anti_hcv = np.where(
    anti_hcv_num == 1,
    np.random.choice(["Reagente", "Indeterminado"],
                     size=N, p=[0.80, 0.20]),
    "Nao_Reagente"
)

# Western Blot — positivo quando anti-HCV reagente
prob_wb = np.where(anti_hcv == "Reagente", 0.75,
          np.where(anti_hcv == "Indeterminado", 0.20, 0.01))
wb = np.where(
    np.random.binomial(1, prob_wb),
    np.random.choice(["Positivo", "Indeterminado"],
                     size=N, p=[0.85, 0.15]),
    "Negativo"
)

# PCR — detectado quando WB positivo
prob_pcr = np.where(wb == "Positivo", 0.85,
           np.where(wb == "Indeterminado", 0.25, 0.005))
pcr = np.where(np.random.binomial(1, prob_pcr),
               "Detectado", "Nao_Detectado")

# TARGET: NAT positivo
# Altamente correlacionado com PCR detectado + anti-HCV reagente
prob_nat = (
    (pcr == "Detectado").astype(float)     * 0.90
    + (anti_hcv == "Reagente").astype(float) * 0.30
    + (wb == "Positivo").astype(float)       * 0.25
    + drogas_iv  * 0.10
    + transfusao * 0.05
).clip(0, 0.98)

nat_positivo = np.random.binomial(1, prob_nat)

# Montar DataFrame
df = pd.DataFrame({
    "idade":        idade,
    "sexo":         sexo,
    "escolaridade": escolaridade,
    "tipo_doador":  tipo_doador,
    "raca":         raca,
    "drogas_iv":    drogas_iv,
    "tatuagem":     tatuagem,
    "transfusao":   transfusao,
    "anti_hcv":     anti_hcv,
    "western_blot": wb,
    "pcr":          pcr,
    "nat_positivo": nat_positivo,
})

# Salvar dataset simulado
df.to_csv("dados_simulados_hcv.csv", index=False)
print(f"   ✓ {N} doadores gerados")
print(f"   ✓ NAT positivos: {nat_positivo.sum()} ({nat_positivo.mean():.1%})")
print(f"   ✓ Arquivo salvo: dados_simulados_hcv.csv")

# -------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO
# -------------------------------------------------------------

print("\n[2/4] Pré-processando dados...")

df_model = df.copy()

# Codificar variáveis categóricas
encoders = {}
cat_cols = ["sexo", "escolaridade", "tipo_doador", "raca",
            "anti_hcv", "western_blot", "pcr"]

for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# Salvar encoders para uso no dashboard
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

X = df_model.drop("nat_positivo", axis=1)
y = df_model["nat_positivo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   ✓ Treino: {len(X_train)} amostras")
print(f"   ✓ Teste:  {len(X_test)} amostras")
print(f"   ✓ Encoders salvos: encoders.pkl")

# -------------------------------------------------------------
# 3. TREINAR MODELO
# -------------------------------------------------------------

print("\n[3/4] Treinando Gradient Boosting...")

modelo = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
modelo.fit(X_train, y_train)

# Salvar modelo
with open("modelo_hcv.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("   ✓ Modelo treinado e salvo: modelo_hcv.pkl")

# -------------------------------------------------------------
# 4. AVALIAR MÉTRICAS
# -------------------------------------------------------------

print("\n[4/4] Avaliando desempenho...")

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
f1  = f1_score(y_test, y_pred)

print(f"\n   {'─'*35}")
print(f"   AUC-ROC  : {auc:.4f}")
print(f"   F1-Score : {f1:.4f}")
print(f"   {'─'*35}")
print(f"\n{classification_report(y_test, y_pred, target_names=['NAT-', 'NAT+'])}")

# Salvar métricas para o dashboard
metricas = {
    "auc":   round(auc, 4),
    "f1":    round(f1, 4),
    "n_train": len(X_train),
    "n_test":  len(X_test),
    "n_positivos": int(nat_positivo.sum()),
    "feature_names": list(X.columns),
}
with open("metricas.pkl", "wb") as f:
    pickle.dump(metricas, f)

print("=" * 55)
print("  Tudo pronto! Agora rode: streamlit run app.py")
print("=" * 55)
