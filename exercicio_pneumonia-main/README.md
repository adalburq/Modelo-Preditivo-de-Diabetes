# Detecção de Pneumonia em Imagens de Raio-X

Este projeto implementa um pipeline completo de classificação de imagens médicas utilizando Deep Learning e Transfer Learning, com o objetivo de identificar casos de pneumonia a partir de radiografias de tórax (Chest X-Rays).

O modelo foi desenvolvido em TensorFlow/Keras, utilizando a arquitetura ResNet50V2 pré-treinada no ImageNet, e faz uso do dataset público Chest X-Ray Pneumonia, disponibilizado no Kaggle.

O projeto aborda desde a aquisição segura dos dados até o treinamento, avaliação e análise dos resultados do modelo.


## ⚙️ Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Kaggle API
- Google Colab


## 🚀 Como Executar o Projeto

O notebook pode ser aberto diretamente no Google Colab utilizando o badge Open in Colab abaixo ou acessando o arquivo ExercicioExtra_PosFIAP.ipynb via GitHub.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/SEU_REPOSITORIO/blob/main/ExercicioExtra_PosFIAP.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## 🔐 Gerenciamento Seguro de Credenciais

O download do dataset é realizado por meio da API oficial do Kaggle.  
Para evitar a exposição de credenciais sensíveis no código-fonte, este projeto utiliza os Secrets do Google Colab.

As credenciais:
- ❌ Não ficam hardcoded no notebook  
- ❌ Não são versionadas no GitHub  
- ✅ São carregadas apenas em tempo de execução  

Essa abordagem permite o compartilhamento público do repositório sem riscos de segurança.


## ▶️ Executar o Notebook

Após configurar os Secrets, execute as células do notebook em ordem.
O pipeline irá automaticamente:

- Baixar o dataset do Kaggle
- Realizar o pré-processamento das imagens
- Treinar o modelo de Deep Learning
- Avaliar o desempenho nos dados de teste


## 📊 Sobre o Dataset

- Fonte: Kaggle  
- Nome: Chest X-Ray Pneumonia  
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  

O dataset está organizado em três subconjuntos:
- `train/` — dados de treinamento  
- `val/` — dados de validação  
- `test/` — dados de teste  

Cada subconjunto contém duas classes:
- `NORMAL`
- `PNEUMONIA`

Os rótulos são inferidos automaticamente a partir da estrutura de diretórios.


## ⚙️ Arquitetura do Modelo

- Backbone: ResNet50V2 (pré-treinada no ImageNet)
- Estratégia: Transfer Learning
- Camadas finais:
    - Global Average Pooling
    - Dropout
    - Dense com ativação sigmoide

Durante o treinamento, o backbone convolucional é mantido congelado, permitindo que o modelo aprenda apenas os pesos das camadas finais, reduzindo o risco de overfitting e o custo computacional.

## 📊 Métricas de Avaliação

O desempenho do modelo é avaliado utilizando as seguintes métricas:

- Accuracy
- Precision
- Recall
- Loss

Essas métricas são particularmente relevantes em aplicações médicas, onde erros de classificação — especialmente falsos negativos — podem gerar impactos significativos.


## 📌 Resultados

De modo geral, o modelo apresenta bom desempenho na distinção entre radiografias normais e casos de pneumonia, demonstrando a eficácia do uso de Transfer Learning em conjuntos de dados médicos.

## 📌 Contexto Acadêmico

Este projeto foi desenvolvido com fins educacionais e acadêmicos, no contexto de estudos em:

- Visão Computacional
- Deep Learning
- Inteligência Artificial aplicada à Saúde


Este projeto não deve ser utilizado para diagnóstico médico real.
Os resultados apresentados possuem caráter experimental e educacional.
