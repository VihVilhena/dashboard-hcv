"""
Treinamento e exportação do modelo — NAT HCV (HEMOPA)
========================================================
Rode este script QUANDO os dados reais e validados do HEMOPA estiverem
disponíveis. Ele treina o modelo, avalia o desempenho e salva um único
arquivo `modelo_hcv.pkl` no formato que o app.py (dashboard) espera.

Depois de rodar este script com sucesso, basta colocar o `modelo_hcv.pkl`
gerado na mesma pasta do app.py — o dashboard passa a usar o modelo real
automaticamente, sem precisar mudar nenhuma linha do app.py.

USO:
    python treinar_modelo.py --dados caminho/para/dados_hemopa.csv

O CSV de entrada deve ter as colunas:
    idade, sexo, tipo_doacao, anti_hcv, alt, transfusao_previa,
    tatuagem, drogas_injetaveis, multiplos_parceiros, area_endemica,
    nat_positivo   <- coluna alvo (0 = não reagente, 1 = reagente)
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    recall_score, precision_score, roc_curve,
)

FEATURE_COLS = [
    "idade", "sexo", "tipo_doacao", "anti_hcv", "alt",
    "transfusao_previa", "tatuagem", "drogas_injetaveis",
    "multiplos_parceiros", "area_endemica",
]
CATEGORICAL_COLS = [
    "sexo", "tipo_doacao", "anti_hcv", "transfusao_previa",
    "tatuagem", "drogas_injetaveis", "multiplos_parceiros", "area_endemica",
]
TARGET_COL = "nat_positivo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", required=True, help="Caminho para o CSV com os dados reais do HEMOPA")
    parser.add_argument("--saida", default="modelo_hcv.pkl", help="Caminho de saída do modelo treinado")
    parser.add_argument("--test-size", type=float, default=0.25, help="Proporção para conjunto de teste")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Carregando dados de: {args.dados}")
    df = pd.read_csv(args.dados)

    faltando = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if faltando:
        raise ValueError(f"O CSV está sem as colunas: {faltando}")

    print(f"Total de registros: {len(df)}")
    print(f"Prevalência de NAT positivo: {df[TARGET_COL].mean():.4f}")

    # ---------- Codificação de variáveis categóricas ----------
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    # ---------- Divisão treino/teste ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ---------- Treinamento ----------
    # NOTA: os hiperparâmetros abaixo são um ponto de partida razoável.
    # Se o Orange indicou hiperparâmetros diferentes como "melhor modelo",
    # ajuste aqui para reproduzir o mesmo resultado reportado no TCC.
    modelo = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=args.seed,
    )

    print("Treinando modelo...")
    modelo.fit(X_train, y_train)

    # ---------- Validação cruzada (checagem de estabilidade) ----------
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"AUROC (validação cruzada, 5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ---------- Avaliação no conjunto de teste ----------
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = modelo.predict(X_test)

    auroc = roc_auc_score(y_test, y_proba)
    sensibilidade = recall_score(y_test, y_pred)
    especificidade = recall_score(y_test, y_pred, pos_label=0)
    acuracia = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Métricas no conjunto de teste ---")
    print(f"AUROC:          {auroc:.3f}")
    print(f"Sensibilidade:  {sensibilidade:.3f}")
    print(f"Especificidade: {especificidade:.3f}")
    print(f"Acurácia:       {acuracia:.3f}")
    print(f"Matriz de confusão: {cm}")

    # ---------- Exportação ----------
    pacote = {
        "modelo": modelo,
        "encoders": encoders,
        "metricas": {
            "auroc": round(float(auroc), 3),
            "sensibilidade": round(float(sensibilidade), 3),
            "especificidade": round(float(especificidade), 3),
            "acuracia": round(float(acuracia), 3),
            "matriz_confusao": cm,
        },
    }

    joblib.dump(pacote, args.saida)
    print(f"\nModelo salvo em: {args.saida}")
    print("Copie este arquivo para a pasta do app.py para o dashboard passar a usar o modelo real.")


if __name__ == "__main__":
    main()
