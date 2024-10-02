import pickle
import sys
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load Model
@st.cache_resource
def load_model():
    filehandler = open('model_rf.pkl', 'rb')
    model = pickle.load(filehandler)
    return model

def simple_pie_chart(prediction, df, new_data, col_name):
    df = df.drop(df.tail(1).index)
    val = df[col_name].value_counts()
    colors = ['#00567A', '#00567A']
    
    if col_name == "loan_status":
        highlight_label = " Rejected"
        if prediction == 1:
            highlight_label = " Approved"
        for i, label in enumerate(val.index):
            if label == highlight_label:
                colors[i] = '#05dde0'
    elif col_name == "education":
        education = new_data["education"][0]
        highlight_label = " Not Graduate"
        if education == 1:
            highlight_label = " Graduate"
        for i, label in enumerate(val.index):
            if label == highlight_label:
                colors[i] = '#05dde0'
    else:
        self_employment = new_data["self_employed"][0]
        highlight_label = " No"
        if self_employment == 1:
            highlight_label = " Yes"
        for i, label in enumerate(val.index):
            if label == highlight_label:
                colors[i] = '#05dde0'

    
    fig = go.Figure(data=[go.Pie(labels=val.index, values=val)])
    fig.update_traces(marker=dict(colors=colors))
    return fig

def hue_pie_chart(prediction, df, new_data, col_name):
    colors = ['#05DDE0', '#003952', '#F8E6AE', '#F61F0C']
    labels = 0
    if col_name == "education":
        labels = ['Graduate + Approved', 'Graduate + Rejected', 'Not graduate + Approved', 'Not graduate + Rejected']
    else:
        labels = ['Self employed + Approved', 'Self employed + Rejected', 'Not self employed + Approved', 'Not self employed + Rejected']
    values = [
        df[((df[col_name] == " Graduate") | (df[col_name] == " Yes")) & (df["loan_status"] == " Approved")].size,
        df[((df[col_name] == " Graduate") | (df[col_name] == " Yes")) & (df["loan_status"] == " Rejected")].size,
        df[((df[col_name] == " Not Graduate") | (df[col_name] == " No")) & (df["loan_status"] == " Approved")].size,
        df[((df[col_name] == " Not Graduate") | (df[col_name] == " No")) & (df["loan_status"] == " Rejected")].size
    ]
    idx = 0
    pull_values = [0, 0, 0, 0]

    if len(new_data[new_data[col_name] == 1]) == 1:
        if prediction == 1:
            idx = 0
        else:
            idx = 1
    else:
        if prediction == 1:
            idx = 2
        else:
            idx = 3
    pull_values[idx] += 0.2

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull_values)])
    fig.update_traces(marker=dict(colors=colors))
    return fig

def sunburst_chart(prediction, df, new_data, col_name):
    fig = px.sunburst(df, path=[col_name, 'loan_status'], color='loan_status', color_discrete_map={'(?)':'#F8E6AE', ' Approved': '#003952', ' Rejected':'#F61F0C'})
    fig.update_traces(textinfo="label+percent parent", insidetextorientation='horizontal')
    return fig

def show_depend(prediction, df, new_data):
    data = df["no_of_dependents"].value_counts()
    no_of_dependent = new_data["no_of_dependents"][0]
    colors = ['lightslategray',] * data.size
    idx = 0
    for i, j in enumerate(data.index):
        if(j == no_of_dependent):
            idx = i
    colors[idx] = '#F61F0C'
    fig = go.Figure(data=[go.Bar(
        x=data.index,
        y=data,
        marker_color=colors
    )])
    return fig

def show_income(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "income_annum", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.income_annum[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

def show_loan_am(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "loan_amount", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.loan_amount[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

def show_loan_year(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "loan_term", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.loan_term[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

def show_cred_sc(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "cibil_score", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.cibil_score[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

def show_res_val(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "residential_assets_value", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.residential_assets_value[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

def show_com_val(prediction, df, dist, new_data):
    fig = px.histogram(df, x = "commercial_assets_value", color="loan_status", marginal=dist, hover_data=df.columns)
    fig.add_vline(x=new_data.commercial_assets_value[0], line_dash = 'dash', line_color = '#F61F0C')
    return fig

@st.cache_data
def get_train_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = ['loan_id', 'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value', 'loan_status']
    return df

@st.cache_data
def get_new_data(no_of_dependent, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value):
    column_name =  ['no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value']
    df = pd.DataFrame([[no_of_dependent, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value]])
    df.columns = column_name
    return df

def get_prediction(df):
    pred = model.predict(df)
    return pred

def draw_pie(res, prediction, df, new_data, col_name):
    if(res == "Simple pie chart"):
        fig = simple_pie_chart(prediction, df, new_data, col_name)
    elif(res == "Hue pie chart"):
        fig = hue_pie_chart(prediction, df, new_data, col_name)
    else:
        fig = sunburst_chart(prediction, df, new_data, col_name)

    return fig


def output_result(submit_button, no_of_dependent, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value):
    if(state.submitted):
        if(name == ""):
            st.warning("Please input your name")
        else:
            new_data = get_new_data(no_of_dependent, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value)

            y = get_prediction(new_data)

            first_name = name.split()[0]
            st.markdown(f"## Hello {first_name}!")

            if y[0] == 1:
                st.markdown("✅ Your loan proposal is more likely to be **accepted**.")
            else:
                st.markdown("❌ Your loan proposal is more likely to be **rejected**.")
            st.markdown("Find out more by analyzing the graphs that we have provided below.")
            
            # load data
            train_data = get_train_data()
            # append new data to avoid outbound errors
            new_df = pd.concat([train_data, new_data], axis=0)

            # make tabs
            tab_loan_st, tab_edu, tab_self_emp, tab_depend, tab_income, tab_loan_am, tab_loan_year, tab_cred_sc, tab_res_val, tab_com_val = st.tabs(["Loan status", "Education", "Self employed", "Number of dependents", "Income", "Loan amount", "Loan term", "Credit score", "Residential asset", "Commercial asset"])

            with tab_loan_st:
                # predict loan_status
                fig = simple_pie_chart(y[0], new_df, new_data, "loan_status")
                st.plotly_chart(fig)

            with tab_edu:
                # plot education
                st.selectbox(
                    "Choose chart",
                    ("Simple pie chart", "Hue pie chart", "Sunburst"),
                    key="edu"
                )
                fig = draw_pie(state.edu, y[0], new_df, new_data, "education")
                st.plotly_chart(fig)

            with tab_self_emp:
                # predict self_employed
                st.selectbox(
                    "Choose chart",
                    ("Simple pie chart", "Hue pie chart", "Sunburst"),
                    key="self_emp"
                )
                fig = draw_pie(state.self_emp, y[0], new_df, new_data, "self_employed")
                st.plotly_chart(fig)

            with tab_depend:
                # plot no_of_dependents
                fig = show_depend(y[0], new_df, new_data)
                st.plotly_chart(fig)

            with tab_income:
                # plot income_annum
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="income"
                )
                fig = show_income(y[0], new_df, state.income.lower(), new_data)
                st.plotly_chart(fig)

            with tab_loan_am:
                # plot loan_amount
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="loan_am"
                )
                fig = show_loan_am(y[0], new_df, state.loan_am.lower(), new_data)
                st.plotly_chart(fig)

            with tab_loan_year:
                # plot loan_term
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="loan_year"
                )
                fig = show_loan_year(y[0], new_df, state.loan_year.lower(), new_data)
                st.plotly_chart(fig)

            with tab_cred_sc:
                # plot cibil_score
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="cred_sc"
                )
                fig = show_cred_sc(y[0], new_df, state.cred_sc.lower(), new_data)
                st.plotly_chart(fig)

            with tab_res_val:
                # plot residential_assets_value
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="res_val"
                )
                fig = show_res_val(y[0], new_df, state.res_val.lower(), new_data)
                st.plotly_chart(fig)

            with tab_com_val:
                # plot commercial_assets_value
                st.selectbox(
                    "Choose distribution",
                    ("Box", "Rug", "Violin"),
                    key="com_val"
                )
                fig = show_com_val(y[0], new_df, state.com_val.lower(), new_data)
                st.plotly_chart(fig)
    
    return True

def form_callback():
    state.submitted = True

# start session state
state = st.session_state

#set display into wide mode
st.set_page_config(layout='wide')

#set color into web page 
base="light"
primaryColor="#7edabd"
secondaryBackgroundColor="#f7e5b2"
textColor="#003b54"

if "submitted" not in state:
    state.submitted = False

if "edu" not in state:
    state.edu = "Simple pie chart"

if "self_emp" not in state:
    state.self_emp = "Simple pie chart"

if "income" not in state:
    state.income = "Box"

if "loan_am" not in state:
    state.loan_am = "Box"

if "loan_year" not in state:
    state.loan_year = "Box"

if "cred_sc" not in state:
    state.cred_sc = "Box"

if "res_val" not in state:
    state.res_val = "Box"

if "com_val" not in state:
    state.com_val = "Box"

# load model
model = load_model()


# Welcome section
st.markdown("# 🪙 LoanPredictor")
st.divider()

col1, col2 = st.columns([0.72, 0.28])
with col1:
    st.markdown(
        "### Welcome to LoanPredictor!"
    )
    st.markdown(
        "Do you have any difficulties in your bank loan application process❓"
    )
    st.markdown(
        "No worry! LoanPredictor is here to help you in predicting whether your bank loan application is going to be accepted or rejected. LoanPredictor leverages a machine learning algorithm that can provide you with an accurate result. The main goal is to help you understand your chances and prepare you better for the application process 🤗"
    )
    st.markdown(
        "📃 You can simply fill the form below and get your result in just short amount of time."
    )
with col2:
    st.image('image_1.jpg')


# Streamlit Form
with st.form("Information Form"):
    st.markdown("### Information Form")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe")
        no_of_dependent = st.slider("Number of dependents", 0, 20, 0)
        # st.number_input("Number of Dependents", min_value=0)
        col11, col12 = st.columns(2)
        with col11:
            education = st.radio("Education", ["Graduated", "Not graduated"])
        # st.selectbox("Education", ("Graduated", "Not Graduated"))
        with col12:
            self_employed = st.radio("Self Employed", ["Yes", "No"])
        # st.selectbox("Self Employed", ("Yes", "No"))
    with col2:
        income_annum = st.number_input("Annual Income ($)", min_value=0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0)
        loan_term = st.slider("Loan Term in Year", 0, 20, 0)
        # st.number_input("Loan Term in Year", min_value=0)
    with col3:
        cibil_score = st.number_input("Credit Score", min_value=0)
        residential_assets_value = st.number_input("Residential Asset Value ($)", min_value=0)
        commercial_assets_value = st.number_input("Commercial Asset Value ($)", min_value=0)

    st.write("")
    submit_button = st.form_submit_button(label="Submit", on_click=form_callback)


# Encode Education
if education == "Graduated":
    education = 1
else:
    education = 0

# Encode Self Employed
if self_employed == "Yes":
    self_employed = 1
else:
    self_employed = 0

output_result(submit_button, no_of_dependent, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value)