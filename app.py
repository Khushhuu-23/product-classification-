# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("pricerunner_aggregate.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# Navbar
st.sidebar.title("🔍 PricePredictor App")
page = st.sidebar.radio("Go to", ["Home", "Data Preview", "Graphs", "Predictor", "Model Report"])

# Home
if page == "Home":
    st.title("📦 PricePredictor App")
    st.markdown("""
    Welcome to the **PricePredictor** app!  
    This app lets you:
    - Preview product data
    - Visualize trends
    - Predict a product's cluster
    - See model accuracy report  
    *Built using Streamlit.*
    """)

# Data Preview
elif page == "Data Preview":
    st.title("🗃️ Data Preview")
    st.write(df.head(100))

# Graphs
elif page == "Graphs":
    st.title("📈 Graphical Representations")

    st.subheader("Top Categories")
    top_cats = df['Category Label'].value_counts().head(10)
    st.bar_chart(top_cats)

    st.subheader("Top Clusters")
    top_clusters = df['Cluster Label'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_clusters.values, y=top_clusters.index, ax=ax)
    st.pyplot(fig)

# Predictor
elif page == "Predictor":
    st.title("🔮 Product Cluster Predictor")

    user_input = st.text_input("Enter Product Title")
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a product title.")
        else:
            prediction = model.predict([user_input])[0]
            st.success(f"Predicted Cluster: **{prediction}**")

# Model Report
elif page == "Model Report":
    st.title("📊 Model Report")

    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X = df["Product Title"]
    y = df["Cluster Label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)

    st.text(report)
