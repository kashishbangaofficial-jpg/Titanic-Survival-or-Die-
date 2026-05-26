import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(page_title="Titanic ", layout="wide")

st.markdown('<div class="title">🚢 Titanic Survival Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive ML + Data Visualization</div>', unsafe_allow_html=True)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic_train.csv")

df = load_data()

# ---------------------------
# Clean Data
# ---------------------------
def clean_data(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

    df['Age'] = SimpleImputer(strategy='median').fit_transform(df[['Age']])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    return df

df = clean_data(df)

# ---------------------------
# Encode Features
# ---------------------------
def encode(df):
    df = df.copy()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df

encoded_df = encode(df)

# ---------------------------
# Train Models
# ---------------------------
X = encoded_df.drop("Survived", axis=1)
y = encoded_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Logistic Regression
# ---------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))

# ---------------------------
# Decision Tree Classifier
# ---------------------------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# ---------------------------
# Linear Regression
# ---------------------------
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

lin_pred = lin_model.predict(X_test_scaled)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))

# ---------------------------
# KMeans Clustering
# ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
encoded_df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------------------
# PCA Dimensionality Reduction
# ---------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=pca_components,
    columns=['PC1', 'PC2']
)

pca_df['Survived'] = y.values
pca_df['Cluster'] = encoded_df['Cluster']

# ---------------------------
# Accuracy Display
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.success(f"Logistic Regression Accuracy: {log_acc:.2f}")

with col2:
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    st.info(f"Decision Tree Accuracy: {dt_acc:.2f}")

with col3:
    st.warning(f"Linear Regression RMSE: {lin_rmse:.2f}")

# ---------------------------
# Logo
# ---------------------------
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2bg0jHc0BUO9WYKADCLN4UEIMb7-bKjQKJ3wM0MuKwlOHpmSB5HiG1Rk&s", width=150) 
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2bg0jHc0BUO9WYKADCLN4UEIMb7-bKjQKJ3wM0MuKwlOHpmSB5HiG1Rk&s", width=120)
st.sidebar.title("Titanic App")

# ---------------------------
# Visualization Data
# ---------------------------
viz_df = df.copy()

# Ensure clean values for charts
viz_df['Age'] = viz_df['Age'].fillna(viz_df['Age'].median())
viz_df['Fare'] = viz_df['Fare'].fillna(viz_df['Fare'].median())
viz_df['Embarked'] = viz_df['Embarked'].fillna(viz_df['Embarked'].mode()[0])

# Ensure Sex is string (important for filters + charts)
viz_df['Sex'] = viz_df['Sex'].astype(str)

# ---------------------------
# Sidebar Input Features
# ---------------------------
st.sidebar.header("🎛️ Input Features")

sex_filter = st.sidebar.selectbox("Gender", ["All", "male", "female"])
class_filter = st.sidebar.selectbox("Class", ["All", 1, 2, 3])
age_filter = st.sidebar.slider("Age", 0, 90, (0, 90))
fare_filter = st.sidebar.slider("Fare", 0, 500, (0, 500))

# ---------------------------
# Apply Filters
# ---------------------------
filtered_df = viz_df.copy()

if sex_filter != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == sex_filter]

if class_filter != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == class_filter]

filtered_df = filtered_df[
    (filtered_df["Age"] >= age_filter[0]) &
    (filtered_df["Age"] <= age_filter[1]) &
    (filtered_df["Fare"] >= fare_filter[0]) &
    (filtered_df["Fare"] <= fare_filter[1])
]

# ---------------------------
# Metrics
# ---------------------------
st.metric("Survival Rate", f"{filtered_df['Survived'].mean()*100:.2f}%")

# ---------------------------
# Charts
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x='Survived', title="Survival Count")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.pie(filtered_df, names='Survived', title="Survival Ratio")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.histogram(filtered_df, x='Sex', color='Survived',
                    barmode='group', title="Survival by Gender")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.histogram(filtered_df, x='Age', color='Survived',
                    nbins=30, title="Age Distribution")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.scatter(filtered_df, x='Age', y='Fare',
                  color='Survived', title="Fare vs Age")
st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Supervised Learning Models
# ---------------------------
st.subheader("📚 Supervised Learning Models")

col_sup1, col_sup2 = st.columns(2)

with col_sup1:
    fig_sup1 = px.bar(
        x=['Logistic Regression', 'Decision Tree'],
        y=[log_acc, dt_acc],
        title='Model Accuracy Comparison',
        labels={'x': 'Models', 'y': 'Accuracy'}
    )

    st.plotly_chart(fig_sup1, use_container_width=True, key="sup_chart_1")

with col_sup2:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig_sup2 = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Decision Tree Feature Importance'
    )

    st.plotly_chart(fig_sup2, use_container_width=True, key="sup_chart_2")


# ---------------------------
# Unsupervised Learning
# ---------------------------
st.subheader("🧠 Unsupervised Learning")

col_unsup1, col_unsup2 = st.columns(2)

# ---------------------------
# Clustering Visualization
# ---------------------------
with col_unsup1:
    fig_cluster = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=pca_df['Cluster'].astype(str),
        title='KMeans Clustering using PCA'
    )

    st.plotly_chart(fig_cluster, use_container_width=True, key="cluster_chart")

# ---------------------------
# PCA Visualization
# ---------------------------
with col_unsup2:
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=pca_df['Survived'].astype(str),
        title='PCA Dimensionality Reduction'
    )

    st.plotly_chart(fig_pca, use_container_width=True, key="pca_chart")

# ---------------------------
# Prediction Section
# ---------------------------
st.subheader("🎯 Predict Survival")

col_left, col_right = st.columns([2, 1])

# ---------------------------
# Left Column (Inputs)
# ---------------------------
with col_left:

    selected_model = st.selectbox(
        "🤖 Select ML Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Linear Regression",
            "KMeans Clustering",
            "PCA Dimensionality Reduction"
        ],
        key="model_selector"
    )

    pclass = st.selectbox("Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    fare = st.slider("Fare", 0, 500, 50)
    sibsp = st.slider("Siblings", 0, 5, 0)
    parch = st.slider("Parents", 0, 5, 0)
    embarked = st.selectbox(
        "Embarked",
        ["Cherbourg", "Queenstown", "Southampton"]
    )

# ---------------------------
# Right Column
# ---------------------------
with col_right:

    st.markdown("### 🧠 Prediction Result")

    model_scores = {
        "Logistic Regression": log_acc,
        "Decision Tree": dt_acc
    }

    best_model = max(model_scores, key=model_scores.get)

    st.info(f"🏆 Best Model: {best_model}")

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict"):

    # Encode Inputs
    sex_val = 1 if sex == "male" else 0

    embarked_map = {
        "Cherbourg": 0,
        "Queenstown": 1,
        "Southampton": 2
    }

    # Create Input Array
    input_data = np.array([[
        pclass,
        sex_val,
        age,
        sibsp,
        parch,
        fare,
        embarked_map[embarked]
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # ---------------------------
    # Logistic Regression
    # ---------------------------
    if selected_model == "Logistic Regression":

        pred = log_model.predict(input_scaled)[0]
        prob = log_model.predict_proba(input_scaled)[0][1]
        percentage = prob * 100

        st.progress(int(percentage))
        st.write(f"Survival Probability: {percentage:.2f}%")

        st.success(f"Using: {selected_model}")

        if pred == 1:
            st.success("🎉 Survived")
        else:
            st.error("❌ Did Not Survive")

    # ---------------------------
    # Decision Tree
    # ---------------------------
    elif selected_model == "Decision Tree":

        pred = dt_model.predict(input_data)[0]
        prob = dt_model.predict_proba(input_data)[0][1]
        percentage = prob * 100

        st.progress(int(percentage))
        st.write(f"Survival Probability: {percentage:.2f}%")

        st.success(f"Using: {selected_model}")

        if pred == 1:
            st.success("🎉 Survived")
        else:
            st.error("❌ Did Not Survive")

    # ---------------------------
    # Linear Regression
    # ---------------------------
    elif selected_model == "Linear Regression":

        pred = lin_model.predict(input_scaled)[0]

        percentage = max(0, min(pred * 100, 100))

        st.progress(int(percentage))
        st.write(f"Prediction Score: {pred:.2f}")

        st.success(f"Using: {selected_model}")

        if pred >= 0.5:
            st.success("🎉 Survived")
        else:
            st.error("❌ Did Not Survive")

    # ---------------------------
    # KMeans Clustering
    # ---------------------------
    elif selected_model == "KMeans Clustering":

        cluster = kmeans.predict(input_scaled)[0]

        distances = kmeans.transform(input_scaled)[0]

        confidence = 100 / (1 + min(distances))

        st.success(f"Using: {selected_model}")

        st.progress(int(confidence))

        st.write(f"Cluster Confidence: {confidence:.2f}%")

        st.info(f"Passenger belongs to Cluster: {cluster}")

        if cluster == 0:
            st.write("Cluster 0 → Higher survival tendency")

        elif cluster == 1:
            st.write("Cluster 1 → Medium survival tendency")

        else:
            st.write("Cluster 2 → Lower survival tendency")

    # ---------------------------
    # PCA
    # ---------------------------
    elif selected_model == "PCA Dimensionality Reduction":

        pca_result = pca.transform(input_scaled)

        explained = sum(pca.explained_variance_ratio_) * 100

        st.success(f"Using: {selected_model}")

        st.progress(int(explained))

        st.write(f"Variance Explained: {explained:.2f}%")

        st.write(f"PC1: {pca_result[0][0]:.2f}")
        st.write(f"PC2: {pca_result[0][1]:.2f}")

        st.info("PCA reduces dimensions for analysis.")