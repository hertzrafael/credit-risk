from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import shap
import dotenv
import os


@st.cache_data
def load_data():
    return pd.read_csv(Path(os.getenv("TEMP_DIR")) / "credit_risk_data_gold.csv")

df = load_data()


def _config():
    st.set_page_config(
        page_title="Predição de Risco de Crédito",
        page_icon="💳",
        layout="wide"
    )
    
    
def _check_cache():
    if 'pipeline' not in st.session_state:
        artifact = joblib.load(Path(os.getenv("TEMP_DIR")) / "credit_risk_model.pkl")
        
        pipeline = artifact["model"]
        
        st.session_state.pipeline = pipeline
        st.session_state.preprocess_model = pipeline.named_steps['preprocess']
        st.session_state.model = pipeline.named_steps['model']
        st.session_state.best_threshold = artifact["threshold"]


def _sidebar():
    st.sidebar.title("Filtros")

    age_min, age_max = int(df["person_age"].min()), int(df["person_age"].max())
    age_filter = st.sidebar.slider("Idade", age_min, age_max, (age_min, age_max))

    home_options = df["person_home_ownership"].dropna().unique().tolist()
    home_filter = st.sidebar.multiselect("Tipo de casa", home_options, default=home_options)

    income_max = float(df["person_income"].max())
    income_min = float(df["person_income"].min())

    income_filter = st.sidebar.number_input(
        "Renda máxima",
        min_value=income_min,
        max_value=income_max,
        value=income_max,
        step=(income_max - income_min) / 100,
    )
    
    return {"age": age_filter, "home": home_filter, "income": income_filter}


def dashboard(age_filter, home_filter, income_filter):
    st.title("📊 Dashboard")
 
    # =========================
    # FILTROS
    # =========================
    df_filtered = df[
        (df["person_age"].between(age_filter[0], age_filter[1]))
        & (df["person_home_ownership"].isin(home_filter))
        & (df["person_income"] <= income_filter)
    ].copy()
 
    if df_filtered.empty:
        st.warning("Nenhum registro encontrado para os filtros selecionados.")
        return
 
    # =========================
    # KPIs
    # =========================
    default_rate = df_filtered["loan_status"].mean() * 100
    avg_hist = df_filtered["cb_person_cred_hist_length"].mean()
    avg_income = df_filtered["person_income"].mean()
    avg_loan = df_filtered["loan_amnt"].mean()
    avg_rate = df_filtered["loan_int_rate"].mean()
    avg_pct_income = df_filtered["loan_percent_income"].mean() * 100
 
    col1, col2, col3, col4, col5, col6 = st.columns(6)
 
    col1.metric("Taxa de Inadimplência", f"{default_rate:.2f}%")
    col2.metric("Renda Média", f"R$ {avg_income:,.0f}")
    col3.metric("Histórico Médio (anos)", f"{avg_hist:.1f}")
    col4.metric("Empréstimo Médio", f"R$ {avg_loan:,.0f}")
    col5.metric("Juros Médio", f"{avg_rate:.1f}%")
    col6.metric("% Renda Comprometida", f"{avg_pct_income:.1f}%")
 
    st.caption(f"Base filtrada: {len(df_filtered):,} registros")
 
    st.divider()
 
    # =========================
    # GRÁFICO 1 - DISTRIBUIÇÃO
    # =========================
    st.subheader("Distribuição de inadimplência")
 
    status_counts = (
        df_filtered["loan_status"]
        .value_counts()
        .sort_index()
        .rename({0: "Adimplente", 1: "Inadimplente"})
        .reset_index()
    )
    status_counts.columns = ["status", "quantidade"]
 
    fig1 = px.pie(
        status_counts,
        names="status",
        values="quantidade",
        hole=0.5,
        color="status",
        color_discrete_map={"Adimplente": "#2E86AB", "Inadimplente": "#E63946"},
    )
    fig1.update_traces(textinfo="percent+value")
    st.plotly_chart(fig1, use_container_width=True)
 
    st.divider()
 
    # =========================
    # GRÁFICO 3 - HISTÓRICO DE CRÉDITO VS DEFAULT (boxplot)
    # =========================
    st.subheader("Histórico de crédito vs inadimplência")
 
    df_plot3 = df_filtered.copy()
    df_plot3["status_label"] = df_plot3["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})
 
    fig3 = px.box(
        df_plot3,
        x="status_label",
        y="cb_person_cred_hist_length",
        color="status_label",
        color_discrete_map={"Não default": "#2E86AB", "Default": "#E63946"},
        labels={"status_label": "Status", "cb_person_cred_hist_length": "Anos de histórico"},
        points="outliers",
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
 
    st.divider()
 
    # =========================
    # GRÁFICO 4 - INADIMPLÊNCIA POR TIPO DE POSSE DE IMÓVEL
    # =========================
    st.subheader("Inadimplência por posse de imóvel")
 
    home_risk = (
        df_filtered.groupby("person_home_ownership")["loan_status"]
        .agg(taxa_inadimplencia="mean", quantidade="size")
        .reset_index()
    )
    home_risk["taxa_inadimplencia"] *= 100
 
    fig4 = px.bar(
        home_risk,
        x="person_home_ownership",
        y="taxa_inadimplencia",
        text_auto=".1f",
        hover_data=["quantidade"],
        labels={
            "person_home_ownership": "Posse de imóvel",
            "taxa_inadimplencia": "Taxa de inadimplência (%)",
        },
        color="taxa_inadimplencia",
        color_continuous_scale="Reds",
    )
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)
 
    st.divider()
 
    # =========================
    # GRÁFICO 5 - INADIMPLÊNCIA POR FINALIDADE DO EMPRÉSTIMO
    # =========================
    if "loan_intent" in df_filtered.columns:
        st.subheader("Inadimplência por finalidade do empréstimo")
 
        intent_risk = (
            df_filtered.groupby("loan_intent")["loan_status"]
            .agg(taxa_inadimplencia="mean", quantidade="size")
            .reset_index()
            .sort_values("taxa_inadimplencia", ascending=False)
        )
        intent_risk["taxa_inadimplencia"] *= 100
 
        fig5 = px.bar(
            intent_risk,
            x="taxa_inadimplencia",
            y="loan_intent",
            orientation="h",
            text_auto=".1f",
            hover_data=["quantidade"],
            labels={
                "loan_intent": "Finalidade",
                "taxa_inadimplencia": "Taxa de inadimplência (%)",
            },
            color="taxa_inadimplencia",
            color_continuous_scale="Reds",
        )
        fig5.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)
 
        st.divider()
 
    # =========================
    # GRÁFICO 6 - DISPERSÃO RENDA x VALOR DO EMPRÉSTIMO
    # =========================
    st.subheader("Renda vs valor do empréstimo")
 
    df_plot6 = df_filtered.copy()
    df_plot6["status_label"] = df_plot6["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})
 
    fig6 = px.scatter(
        df_plot6,
        x="person_income",
        y="loan_amnt",
        color="status_label",
        opacity=0.5,
        color_discrete_map={"Adimplente": "#2E86AB", "Inadimplente": "#E63946"},
        labels={
            "person_income": "Renda",
            "loan_amnt": "Valor do empréstimo",
            "status_label": "Status",
        },
    )
    st.plotly_chart(fig6, use_container_width=True)
 
    st.divider()
 
    # =========================
    # GRÁFICO 7 - MAPA DE CORRELAÇÃO DAS VARIÁVEIS NUMÉRICAS
    # =========================
    st.subheader("Correlação entre variáveis numéricas")
 
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "loan_status",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
 
    corr = df_filtered[numeric_cols].corr()
 
    fig7 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig7, use_container_width=True)
 
    st.divider()
 
    # =========================
    # TABELA + DOWNLOAD
    # =========================
    st.subheader("Amostra dos dados filtrados")
    st.dataframe(df_filtered.head(50))
 
    csv = df_filtered.drop(columns=["income_bucket", "income_bucket_str"], errors="ignore").to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        "⬇️ Baixar dados filtrados (CSV)",
        data=csv,
        file_name="credit_risk_filtrado.csv",
        mime="text/csv",
    )


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
    filters = _sidebar()
    
    dashboard_tab, predict_tab = st.tabs(["Dashboard", "Predição"])
    with dashboard_tab:
        dashboard(
            age_filter=filters["age"],
            home_filter=filters["home"],
            income_filter=filters["income"]
        )
    with predict_tab:
        predict()
    

if __name__ == "__main__":
    main()