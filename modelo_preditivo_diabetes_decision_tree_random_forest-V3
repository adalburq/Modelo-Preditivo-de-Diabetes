# ============================================================
# MODELO PREDITIVO DE DIABETES
# Decision Tree e Random Forest com SHAP
# Projeto Final – Pós-Graduação
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    RocCurveDisplay
)

sns.set(style="whitegrid")

# -----------------------------
# Funções utilitárias
# -----------------------------

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o dataset CSV."""
    return pd.read_csv(caminho)

def tratar_valores_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui zeros inválidos por medianas nas colunas clínicas."""
    df_clean = df.copy()
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in invalid_zero_cols:
        mediana = df_clean[col].median()
        df_clean[col] = df_clean[col].replace(0, mediana)
    return df_clean

def dividir_dados(df: pd.DataFrame):
    """Divide em treino e teste (80/20)."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def treinar_modelos(X_train, y_train):
    """Treina Decision Tree e Random Forest."""
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    for mdl in models.values():
        mdl.fit(X_train, y_train)
    return models

def avaliar_modelos(models, X_test, y_test):
    """Avalia modelos e retorna DataFrame com acurácia."""
    resultados = []
    for name, mdl in models.items():
        acc = accuracy_score(y_test, mdl.predict(X_test))
        resultados.append({"Modelo": name, "Accuracy": acc})
        print(f"\nRelatório de Classificação – {name}")
        print(classification_report(y_test, mdl.predict(X_test)))
    return pd.DataFrame(resultados)

def matriz_confusao_df(y_true, y_pred):
    """Cria DataFrame detalhado da matriz de confusão."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame({
        "Classe Real": ["Não Diabetes (0)", "Não Diabetes (0)", "Diabetes (1)", "Diabetes (1)"],
        "Classe Predita": ["Não Diabetes (0)", "Diabetes (1)", "Não Diabetes (0)", "Diabetes (1)"],
        "Resultado": ["VN", "FP", "FN", "VP"],
        "Quantidade": [tn, fp, fn, tp]
    })

def ajustar_random_forest(X_train, y_train):
    """Busca hiperparâmetros com RandomizedSearchCV."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring="accuracy",
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def importancia_features(model, X):
    """Retorna DataFrame com importância das features."""
    importances = model.feature_importances_
    return pd.DataFrame({
        "Feature": X.columns,
        "Coeficiente (Importância)": importances
    }).sort_values(by="Coeficiente (Importância)", ascending=False)

def explicabilidade_shap(model, X_test, X_columns):
    """Gera gráfico SHAP para explicabilidade."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    X_test_df = pd.DataFrame(X_test, columns=X_columns)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test_df)
    else:
        shap.summary_plot(shap_values, X_test_df)

def correlacao_outcome(df_clean):
    """Analisa correlação das features com Outcome."""
    correlations = df_clean.corr(numeric_only=True)["Outcome"].drop("Outcome")
    correlation_table = pd.DataFrame({
        "Feature": correlations.index,
        "Correlação com Outcome": correlations.values,
        "Conclusão": correlations.apply(
            lambda x: "Alta" if abs(x) > 0.30 else "Moderada" if abs(x) > 0.15 else "Baixa"
        )
    }).sort_values(by="Correlação com Outcome", ascending=False)
    return correlation_table

# -----------------------------
# Fluxo principal
# -----------------------------

def main():
    # Caminho do dataset
    caminho = "diabetes.csv"  # ajuste conforme necessário
    
    # 1. Carregar dados
    df = carregar_dados(caminho)
    df_clean = tratar_valores_invalidos(df)
    
    # 2. Dividir dados
    X_train, X_test, y_train, y_test = dividir_dados(df_clean)
    
    # 3. Treinar modelos
    models = treinar_modelos(X_train, y_train)
    
    # 4. Avaliar modelos
    resultados_df = avaliar_modelos(models, X_test, y_test)
    print("\nComparação de Acurácia:")
    print(resultados_df)
    
    # 5. Ajuste hiperparâmetros Random Forest
    rf_final = ajustar_random_forest(X_train, y_train)
    
    # 6. Importância das features
    feature_importance_df = importancia_features(rf_final, X_train)
    print("\nImportância das Features:")
    print(feature_importance_df)
    
    # 7. SHAP
    explicabilidade_shap(rf_final, X_test, X_train.columns)
    
    # 8. Correlação
    correlation_table = correlacao_outcome(df_clean)
    print("\nCorrelação com Outcome:")
    print(correlation_table)

