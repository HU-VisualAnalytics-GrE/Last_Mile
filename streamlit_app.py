# -------------- Imports --------------
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_plotly_events import plotly_events
from sklearn.decomposition import PCA

# -------------- Constants --------------
CLASS_MAPPING = {
    0: 'very_high',
    1: 'high',
    2: 'moderate',
    3: 'low',
    4: 'very_low'
}


# -------------- Helper Functions --------------
def get_label_from_index(index):
    return CLASS_MAPPING[index]


# -------------- Data Loading and Preparation --------------
@st.cache_resource
def load_and_prepare_data(df_test_path, df_target_path):
    df_target = pd.read_csv(df_target_path)
    df_test = pd.read_csv(df_test_path)

    X = df_test  # Features
    y = df_target['x']  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
    return X_train, X_test, y_train, y_test, df_test, df_target


# -------------- Model Training --------------
@st.cache_resource
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model


# -------------- Confusion Matrix Functions --------------
def analyze_misclassifications(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm, cmn


def confussion_matrix(cm, unique_key):
    class_names = ['very high', 'high', 'moderate', 'low', 'very low']
    fig = px.imshow(cm, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        xaxis_title="Actual Class",
        yaxis_title="Predicted Class",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )
    fig.update_traces(
        hovertemplate='<b>Predicted Class: %{y}</b><br>Actual Class: %{x}<br>Wert: %{z}<extra></extra>',
        hoverlabel=dict(bgcolor="grey", font_size=16, font_family="Rockwell"),
        showscale=True,
        colorscale="Blues",
        colorbar=dict(title="Normierte Häufigkeit"),
    )

    # st.plotly_chart(fig)
    selected_points = plotly_events(fig, key=f"plotly_events_{unique_key}")

    if selected_points:
        point = selected_points[0]
        true_label = get_label_from_index(int(point['x']))
        pred_label = get_label_from_index(int(point['y']))
        show_pca_for_labels(true_label, pred_label, unique_key)


def confussion_matrix_normalized(cmn, unique_key):
    class_names = ['very high', 'high', 'medium', 'low', 'very low']
    fig = px.imshow(cmn, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        legend_title="Classification",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100),
        xaxis_title="Actual Class",
        yaxis_title="Predicted Class",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )

    fig.update_traces(
        hovertemplate='<b>Predicted Class: %{y}</b><br>Actual Class: %{x}<br>Wert: %{z}<extra></extra>',
        hoverlabel=dict(bgcolor="lightgrey", font_size=16, font_family="Rockwell"),
        showscale=True,
        colorscale="Blues",
        colorbar=dict(title="Normierte Häufigkeit"),
    )

    # st.plotly_chart(fig)
    selected_points = plotly_events(fig, key=f"plotly_events_{unique_key}")

    if selected_points:
        point = selected_points[0]
        true_label = get_label_from_index(int(point['x']))
        pred_label = get_label_from_index(int(point['y']))
        show_pca_for_labels(true_label, pred_label, unique_key)


# -------------- PCA Functions --------------
def prepare_pca_data(X_test, y_test, predictions):
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(X_test)

    return pd.DataFrame({
        'PC1': pca_features[:, 0],
        'PC2': pca_features[:, 1],
        'TrueLabel': y_test,
        'PredictedLabel': predictions
    })


def adjust_df_for_plot(df, true_label, predicted_label):
    filtered_df = df[(df['TrueLabel'] == true_label) | (df['PredictedLabel'] == predicted_label)].copy()
    filtered_df['classification'] = np.select(
        condlist=[
            (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] == predicted_label),
            (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] != predicted_label),
            (filtered_df['TrueLabel'] != true_label) & (filtered_df['PredictedLabel'] == predicted_label),
        ],
        choicelist=['True Positives', 'False Negatives', 'False Positives'],
        default='Other'
    )
    return filtered_df


def scatter_plot_df(df, true_label, predicted_label):
    color_map = {
        'True Positives': '#00FF00',
        'False Negatives': '#FF0000',
        'False Positives': '#0000FF'
    }

    hover_data = {
        'PC1': True,
        'PC2': True,
        'TrueLabel': True,
        'PredictedLabel': True,
        'classification': True
    }

    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='classification',
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=f'True: {true_label}, Predicted: {predicted_label}'
    )

    fig.update_layout(
        legend_title="Classification",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )

    return fig


def show_pca_for_labels(true_label, pred_label, unique_key):
    pca_df = prepare_pca_data(X_test, y_test, predictions)
    filtered_df = adjust_df_for_plot(pca_df, true_label, pred_label)
    fig = scatter_plot_df(filtered_df, true_label, pred_label)
    st.plotly_chart(fig, use_container_width=True, key=f"pca_{unique_key}")

    st.subheader("Classification Statistics")
    stats = filtered_df['classification'].value_counts()
    st.write(pd.DataFrame({
        'Classification': stats.index,
        'Count': stats.values
    }))


def train_or_load_model(path, x_train, y_train):
    """
    Used to either train or load the model. If a valid path is provided, the model will be loaded from that path.
    If there is no model existing at the path, it will be trained.

    :param path: str path to the model file e.g. models/RandomForestClassifier.pkl
    :return: Model Instance
    """
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError:
        # Load and prepare data
        # Train model and make predictions
        model  = train_model(x_train, y_train)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        return model


# -------------- Main Execution --------------
if __name__ == "__main__":

    df_test_path = "lucas_organic_carbon_training_and_test_data.csv"
    df_target_path = "lucas_organic_carbon_target.csv"
    X_train, X_test, y_train, y_test, df_test, df_target = load_and_prepare_data("lucas_organic_carbon_training_and_test_data.csv", "lucas_organic_carbon_target.csv")

    # Setup Streamlit interface
    st.sidebar.title("Options")
    option_cmn = st.sidebar.checkbox("Normalize Confusion Matrix", value=True)

    st.sidebar.header("Select Models for Comparison")
    option_model_selection_1 = st.sidebar.selectbox("Select Model 1", ["Unoptimized Model", "Optimized Model"])
    option_model_selection_2 = st.sidebar.selectbox("Select Model 2", ["Optimized Model", "Unoptimized Model"])

    st.sidebar.header("Select number of displayed top features")
    option_number_top_features = st.sidebar.slider("Top Features to display", min_value=1, max_value=50, value=5)

    rf_model_1 = train_or_load_model("models/RandomForestClassifier1.pkl", X_train, y_train)
    rf_model_2 = train_or_load_model("models/RandomForestClassifier2.pkl", X_train, y_train)

    st.title("Lucas Organic Carbon Dataset")
    st.header("Overview of Misclassifications")

    misclasification_model_1, misclassification_model_2 = st.columns(2, gap="small")

    with misclasification_model_1:
        
        st.subheader(option_model_selection_1)

        predictions = rf_model_1.predict(X_test)

        # Create confusion matrix
        cm, cmn = analyze_misclassifications(y_test, predictions)

        # Display appropriate matrix
        if option_cmn:
            confussion_matrix_normalized(cmn, 1)
        else:
            confussion_matrix(cm, 1)

    with misclassification_model_2:

        st.subheader(option_model_selection_2)

        predictions = rf_model_2.predict(X_test)

        # Create confusion matrix
        cm, cmn = analyze_misclassifications(y_test, predictions)

        # Display appropriate matrix
        if option_cmn:
            confussion_matrix_normalized(cmn, 2)
        else:
            confussion_matrix(cm, 2)

    st.header("Feature Importances")
    top_n = option_number_top_features
    feature_importance_1 = rf_model_1.feature_importances_
    feature_importance_2 = rf_model_2.feature_importances_

    feature_importance_model_1, feature_importance_model_2 = st.columns(2, gap="small")

    with feature_importance_model_1:
        st.subheader(option_model_selection_1)

        sorted_idx = np.argsort(feature_importance_1)[::-1][:top_n]  # Korrekte Sortierung
        sorted_importance = feature_importance_1[sorted_idx]
        sorted_features = df_test.columns[sorted_idx]

        top_features_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importance
        })

        fig = px.bar(
            top_features_df,
            x='Feature',
            y='Importance',
            orientation='v',
            title=f'Top {top_n} Feature Importance',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'}
        )

        st.plotly_chart(fig, key="feature_importance_chart_1")

        selected_feature = st.selectbox(
            "Wähle ein Feature aus:",
            options=top_features_df['Feature'],
            key="feature_selection_1"
        )

        if selected_feature:
            # Anzahl der Intervalle festlegen
            num_intervals = 10
            intervals = pd.cut(X_test.iloc[0], bins=num_intervals, labels=['Intervall ' + str(i+1) for i in range(num_intervals)])

            # Intervall für das ausgewählte Feature bestimmen
            selected_value = X_test[selected_feature].iloc[0]  # Beispiel: Erster Wert im Testdatensatz
            selected_interval = intervals.iloc[0]  # Intervall für den ausgewählten Wert

            # Ergebnisse anzeigen
            st.write(f"Du hast das Feature **{selected_feature}** ausgewählt!")
            st.write(f"Der Wert des Features im Testdatensatz: {selected_value:.8f}")
            st.write(f"Intervall für das ausgewählte Feature: {selected_interval}")
    with feature_importance_model_2:
        st.subheader(option_model_selection_2)
        
        sorted_idx = np.argsort(feature_importance_2)[::-1][:top_n]  # Korrekte Sortierung
        sorted_importance = feature_importance_2[sorted_idx]
        sorted_features = df_test.columns[sorted_idx]

        top_features_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importance
        })

        fig = px.bar(
            top_features_df,
            x='Feature',
            y='Importance',
            orientation='v',
            title=f'Top {top_n} Feature Importance',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'}
        )

        st.plotly_chart(fig, key="feature_importance_chart_2")

        selected_feature = st.selectbox(
            "Wähle ein Feature aus:",
            options=top_features_df['Feature'],
            key="feature_selection_2"
        )

        if selected_feature:
            importance_value = top_features_df[top_features_df['Feature'] == selected_feature]['Importance'].values[0]
            st.write(f"Du hast das Feature **{selected_feature}** ausgewählt!")
            st.write(f"Importance-Wert: {importance_value:.4f}")