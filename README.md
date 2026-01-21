# 📊 Modelo Preditivo de Diabetes com Decision Tree e Random Forest (SHAP)

Trabalho de entrega da Fase 1 da Pós Graduação IA para Devs da FIAP

---

## 🎯 Objetivo
Desenvolver e avaliar modelos de **aprendizado de máquina supervisionado** para **auxílio ao diagnóstico de diabetes**, comparando os algoritmos **Decision Tree** e **Random Forest**, com análise detalhada de desempenho e **explicabilidade via SHAP**.

> ⚠️ O modelo não substitui diagnóstico médico. Atua como ferramenta de **apoio à decisão**.

---

## 📂 Dataset
- **Nome:** Pima Indians Diabetes Database
- **Fonte:** Kaggle
- **Link:** https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data
- **Descrição:** Dados clínicos e demográficos utilizados para prever a presença de diabetes (Outcome).

Observação importante:
- O dataset **não possui valores NaN explícitos**.
- Algumas variáveis clínicas utilizam **zero como valor inválido**, tratado adequadamente no pré-processamento.

---

## 🧪 Metodologia

### 1. Análise Exploratória
- Verificação de valores ausentes (NaN)
- Análise de distribuição das classes (desbalanceamento)
- Avaliação exploratória de outliers

### 2. Pré-processamento
- Substituição de zeros inválidos pela **mediana** (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Não foi aplicada normalização, pois **Decision Tree e Random Forest não dependem de escala**

### 3. Modelagem
- **Decision Tree Classifier**
- **Random Forest Classifier**

Motivação:
- Modelos baseados em árvores oferecem **boa performance** e **alta interpretabilidade**, especialmente relevantes em contextos de saúde.

### 4. Treinamento e Validação
- Divisão dos dados:
  - 80% treino
  - 20% teste
- Amostragem estratificada para preservação da proporção das classes

### 5. Avaliação de Desempenho
- Accuracy
- Precision
- Recall
- F1-score
- Curva ROC
- Matrizes de confusão (tabelas e heatmaps)

### 6. Ajuste de Hiperparâmetros
- Random Forest otimizado com **RandomizedSearchCV**

### 7. Explicabilidade (SHAP)
- Utilização do **TreeExplainer**
- Análise da contribuição individual das features
- Visualização global com **SHAP summary plot**

### 8. Análises Complementares
- Importância das features (Random Forest)
- Correlação entre cada feature e o Outcome
- Identificação dos principais fatores de risco

---

## 📈 Principais Resultados
- O **Random Forest apresentou melhor desempenho geral**, superando a Decision Tree em acurácia e estabilidade.
- As features mais relevantes para a predição incluem variáveis clínicas como **Glucose**, **BMI** e **Age**.
- A análise SHAP confirmou a coerência clínica das decisões do modelo.

---

## 📌 Estrutura do Notebook
- Análise de valores ausentes (NaN)
- Tratamento de dados inválidos
- Modelagem e comparação de algoritmos
- Relatórios de classificação (texto + gráficos)
- Matrizes de confusão (tabelas + heatmaps)
- Curvas ROC
- Importância das features
- SHAP e interpretabilidade
- Análise de correlação com Outcome

---

## 🛠️ Tecnologias Utilizadas
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SHAP

---




