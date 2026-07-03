import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import dotenv


def _config():
    st.set_page_config(
        page_title="Predição de Risco de Crédito",
        page_icon="💳",
        layout="wide"
    )
    
    
def _check_cache():
    if 'model' in st.session_state:
        return
    
    artifact = joblib.load("temp/credit_risk_model.pkl")
    
    pipeline = artifact["model"]
    
    st.session_state.pipeline = pipeline
    st.session_state.preprocess_model = pipeline.named_steps['preprocess']
    st.session_state.model = pipeline.named_steps['model']
    st.session_state.best_threshold = artifact["threshold"]


def _sidebar():
    st.sidebar.title("Menu")


def dashboard():
    st.title("Dashboard")


def predict():
    st.title("💳 Predição de Risco de Crédito Individual")
    st.markdown("Informe os dados do cliente para realizar a previsão de risco de crédito.")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input(
            "Idade",
            min_value=18,
            max_value=100,
            value=30
        )

        person_income = st.number_input(
            "Renda Anual (US$)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        person_home_ownership = st.selectbox(
            "Tipo de Moradia",
            [
                "RENT",
                "OWN",
                "MORTGAGE",
                "OTHER"
            ]
        )

        person_emp_length = st.number_input(
            "Tempo de Emprego (anos)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5
        )

        loan_intent = st.selectbox(
            "Objetivo do Empréstimo",
            [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

    with col2:

        loan_grade = st.selectbox(
            "Classificação de Crédito",
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G"
            ]
        )

        loan_amnt = st.number_input(
            "Valor do Empréstimo (US$)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        loan_int_rate = st.number_input(
            "Taxa de Juros (%)",
            min_value=0.0,
            max_value=100.0,
            value=12.5,
            step=0.1
        )

        cb_person_default_on_file = st.selectbox(
            "Possui histórico de inadimplência?",
            [
                "N",
                "Y"
            ]
        )

        cb_preson_cred_hist_length = st.number_input(
            "Tempo de Histórico de Crédito (anos)",
            min_value=0.0,
            max_value=50.0,
            value=8.0,
            step=1.0
        )

    if st.button("Realizar Previsão", width='stretch'):

        input_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": (loan_amnt / (person_income + 1e-8)),
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_preson_cred_hist_length
        }])

        st.subheader("Dados enviados")
        st.dataframe(input_df, width='stretch')
        
        pipeline = st.session_state.pipeline
        preprocess_model = st.session_state.preprocess_model
        model = st.session_state.model
        threshold = st.session_state.best_threshold
        
        feature_names = preprocess_model.get_feature_names_out()
        X = preprocess_model.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        
        y_proba = pipeline.predict_proba(input_df)[0, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        st.subheader("Resultado da Análise")
        st.metric(
            "Probabilidade de Inadimplência",
            f"{y_proba:.1%}"
        )
        
        if y_pred:
            st.error("🚨 Cliente classificado como **ALTO RISCO**")
        else:
            st.success("✅ Cliente classificado como **BAIXO RISCO**")
        
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X[0],
            feature_names=feature_names
        )

        shap.plots.waterfall(explanation, show=False)
        st.pyplot(plt.gcf())
        

def main():
    dotenv.load_dotenv()
    
    _check_cache()
    
    _config()
    _sidebar()
    
    dashboard_tab, predict_tab = st.tabs(["Dashboard", "Predição"])
    with dashboard_tab:
        dashboard()
    with predict_tab:
        predict()
    

if __name__ == "__main__":
    main()