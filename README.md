# Credit Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-purple)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Status](https://img.shields.io/badge/Project-Active-success)

Sistema de previsão de risco de crédito baseado em Machine Learning com pipeline completo de ETL, treinamento, tracking com MLflow e dashboard interativo em Streamlit.

O objetivo do projeto é estimar a probabilidade de inadimplência de clientes com base em variáveis financeiras e comportamentais.

------------------------------------------------------------
## TECNOLOGIAS UTILIZADAS

- Python 3.12
- Scikit-learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Streamlit
- Matplotlib
- SHAP
- MLflow
- Docker
- Joblib

------------------------------------------------------------
## COMO RODAR O PROJETO (DOCKER)

### 1. Clonar o repositório

```bash
git clone https://github.com/hertzrafael/credit-risk
cd credit-risk
```

### 2. Subir a aplicação

```bash
docker compose up --build
```

### 3. Acessar no navegador

Streamlit:
http://localhost:8501

------------------------------------------------------------
## ESTRUTURA DO PROJETO

```
credit-risk/
├── model/        -> pipeline de treino e ML
├── view/         -> dashboard Streamlit
├── temp/         -> modelos serializados (pickle)
├── docker/       -> configuração de deploy
├── pyproject.toml
└── docker-compose.yml
```

------------------------------------------------------------
## PROBLEMAS IDENTIFICADOS NO DATASET

- 895 nulos em person_emp_length
- 3116 nulos em loan_int_rate
- Outliers em person_emp_length
- 35 registros com idade > percentil 99.9% (66 anos)

------------------------------------------------------------
## TRATAMENTO DE NULOS

### person_emp_length
- Aplicado filtro por faixa de renda (person_income)
- Lower/upper bound de 1500
- Mediana por grupo de renda
- Fallback: mediana global do dataset

### loan_int_rate
- Preenchido com mediana por loan_grade

------------------------------------------------------------
## TRATAMENTO DE OUTLIERS

- Outliers não foram removidos (considerados erros de input)
- Idades > 100 anos:
  - Ajustadas para age + person_emp_length

------------------------------------------------------------
## MODELOS UTILIZADOS

- Logistic Regression
- Random Forest
- XGBoost (melhor performance)
- LightGBM

------------------------------------------------------------
## MÉTRICA PRINCIPAL

- ROC AUC (GridSearchCV)
- Avaliação final com:
  - Precision
  - Recall
  - F1-score
  - Accuracy

------------------------------------------------------------
## MLFLOW

Utilizado para:

- Tracking de experimentos
- Log de parâmetros
- Log de métricas
- Registro de modelos

------------------------------------------------------------
## MODELO FINAL

Salvo como:

```
temp/credit_risk_model.pkl
```

Estrutura:

```
{
  "model": pipeline_treinado,
  "threshold": melhor_threshold
}
```

------------------------------------------------------------
## OBSERVAÇÕES

- Pipeline inclui encoding e scaling automático
- Threshold otimizado por F1-score
- XGBoost apresentou melhor performance geral

------------------------------------------------------------
## AUTOR

Hertz Rafael  
GitHub: https://github.com/hertzrafael