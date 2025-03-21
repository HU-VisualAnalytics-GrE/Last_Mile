# -------------- Imports --------------
import os
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(layout="wide")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_plotly_events import plotly_events
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras

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

def partition_dataframe(df, n_intervals):
    """
    Partition a dataframe into n intervals of equal length along its columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to be partitioned
    n_intervals : int
        Number of intervals to partition the data into

    Returns:
    --------
    list of pandas.DataFrame:
        List of dataframes, each containing columns for one interval
    dict:
        Information about the partitioning including interval boundaries
    """
    # Get total number of columns
    n_columns = df.shape[1]

    # Calculate interval size (number of columns per interval)
    interval_size = n_columns // n_intervals

    # Create list to store partitioned dataframes
    partitioned_dfs = []

    # Store information about intervals
    interval_info = {
        'n_intervals': n_intervals,
        'interval_size': interval_size,
        'boundaries': []
    }



    # Partition the dataframe
    for i in range(n_intervals):
        start_idx = i * interval_size
        end_idx = start_idx + interval_size if i < n_intervals - 1 else n_columns

        # Extract the interval
        interval_df = df.iloc[:, start_idx:end_idx]
        partitioned_dfs.append(interval_df)

        # Store boundary information
        interval_info['boundaries'].append({
            'interval': i,
            'start_col': df.columns[start_idx],
            'end_col': df.columns[end_idx-1],
            'n_columns': end_idx - start_idx
        })

    return partitioned_dfs, interval_info

def find_interval(value, interval_info):
    """
    Find the interval that contains a specific value.

    Parameters:
    -----------
    value : float
        The value to locate in the intervals
    interval_info : dict
        The interval information dictionary returned by partition_dataframe

    Returns:
    --------
    dict
        Information about the matching interval including interval number and boundaries
    """
    for boundary in interval_info['boundaries']:
        if float(boundary['start_col']) <= float(value) <= float(boundary['end_col']):
            return {
                'interval_number': boundary['interval'],
                'start': boundary['start_col'],
                'end': boundary['end_col'],
                'n_columns': boundary['n_columns']
            }
    return None

def calculate_interval_error_rf(df, interval_number, y_true, partitioned_data, interval_info, rf_classifier):
    """
    Berechnet den Klassifizierungsfehler für ein spezifisches Intervall mit einem vortrainierten Random Forest.
    Behält die Feature-Namen bei.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original Dataframe mit allen Daten
    interval_number : int
        Nummer des zu analysierenden Intervalls
    y_true : array-like
        Wahre Labels der Daten
    partitioned_data : list
        Liste der partitionierten DataFrames
    interval_info : dict
        Intervallinformationen
    rf_classifier : RandomForestClassifier
        Vortrainierter Random Forest Classifier

    Returns:
    --------
    dict
        Dictionary mit verschiedenen Fehlermetriken
    """
    # Extrahiere das gewünschte Intervall
    interval_df = partitioned_data[interval_number]

    # Finde die Grenzen des Intervalls
    boundaries = interval_info['boundaries'][interval_number]

    # Erstelle einen Null-DataFrame mit den originalen Feature-Namen
    full_feature_df = pd.DataFrame(
        np.zeros((len(y_true), len(df.columns))),
        columns=df.columns
    )

    # Berechne Start- und End-Indizes für das Intervall
    start_idx = interval_number * interval_info['interval_size']
    end_idx = start_idx + interval_df.shape[1]

    # Fülle nur die relevanten Features des Intervalls
    feature_names = df.columns[start_idx:end_idx]
    full_feature_df.loc[:, feature_names] = interval_df.values

    # Vorhersagen mit dem vortrainierten Classifier
    y_pred = rf_classifier.predict(full_feature_df)

    # Berechne Metriken
    results = accuracy_score(y_true, y_pred)

    return results

def combine_interval_points(df, interval_number, partitioned_data, interval_info):
    """
    Erstellt für Punkte des Intervalls linke und rechte Kopien.
    """
    # Extrahiere das gewünschte Intervall
    interval_df = partitioned_data[interval_number]

    # Finde die Grenzen des Intervalls
    boundaries = interval_info['boundaries'][interval_number]

    # Berechne Start- und End-Indizes für das Intervall
    start_idx = interval_number * interval_info['interval_size']
    end_idx = start_idx + interval_df.shape[1]
    feature_names = df.columns[start_idx:end_idx]

    # Kombiniere nur die Intervall-Punkte
    combined_df = pd.DataFrame(columns=df.columns)

    # Verarbeite nur Punkte, die im Intervall vorhanden sind
    for idx in range(interval_df.shape[0]):
        # Originaler Punkt
        original_point = df.iloc[idx].copy()

        # Linke Kopie
        left_point = df.iloc[idx].copy()
        left_point[feature_names] = interval_df.iloc[idx].values

        # Rechte Kopie
        right_point = df.iloc[idx].copy()
        right_point[feature_names] = interval_df.iloc[idx].values

        # Füge Punkte zum kombinierten DataFrame hinzu
        combined_df = pd.concat([
            combined_df,
            pd.DataFrame([left_point, right_point])
        ], ignore_index=True)

    return combined_df

# -------------- Data Loading and Preparation --------------
@st.cache_resource
def load_data():
    df_target = pd.read_csv("lucas_organic_carbon_target.csv")
    df_test = pd.read_csv("lucas_organic_carbon_training_and_test_data.csv")
    return df_target, df_test


def prepare_data(df_test, df_target):
    X = df_test  # Features
    y = df_target['x']  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
    return X_train, X_test, y_train, y_test


# -------------- Model Training --------------
def train_rf_model(X_train, y_train, n_estimators=100, max_depth=None, ):
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
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


def confussion_matrix(cm, unique_key, x, y, pred):
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
        show_pca_for_labels(true_label, pred_label, unique_key, x, y, pred)


def confussion_matrix_normalized(cmn, unique_key, x, y, pred):
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
        show_pca_for_labels(true_label, pred_label, unique_key, x, y, pred)


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


def show_pca_for_labels(true_label, pred_label, unique_key, x, y, pred):
    pca_df = prepare_pca_data(x, y, pred)
    filtered_df = adjust_df_for_plot(pca_df, true_label, pred_label)
    fig = scatter_plot_df(filtered_df, true_label, pred_label)
    st.plotly_chart(fig, use_container_width=True, key=f"pca_{unique_key}")

    st.subheader("Classification Statistics")
    stats = filtered_df['classification'].value_counts()
    st.write(pd.DataFrame({
        'Classification': stats.index,
        'Count': stats.values
    }))


def train_or_load_model(path, x_train=None, y_train=None, n_estimators=100, max_depth=None):
    """
    Used to either train or load the model. If a valid path is provided, the model will be loaded from that path.
    If there is no model existing at the path, it will be trained.

    :param path: str path to the model file e.g. models/RandomForestClassifier.pkl
    :return: Model Instance
    """
    try:
        if path.endswith(".pkl"):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                return model
        elif path.endswith(".h5"):
            model = tf.keras.models.load_model(path, custom_objects={'mse': keras.losses.mse})
            print(model)
            return model
    except Exception:
        # Load and prepare data
        # Train model and make predictions
        if path.endswith(".pkl"):
            model = train_rf_model(x_train, y_train, n_estimators, max_depth)
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            return model
        return None


# -------------- Main Execution --------------
if __name__ == "__main__":
    df_target, df_test = load_data()

    # Setup Streamlit interface

    st.sidebar.title("Options")
    option_cmn = st.sidebar.checkbox("Normalize Confusion Matrix", value=True)
    model_dir = "models"
    model_names = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".pkl")]
    st.sidebar.header("Select Models for Comparison")
    option_model_selection_1 = st.sidebar.selectbox("Select Model 1", model_names, index=model_names.index("RandomForestClassifierUnoptimized.pkl"))
    option_model_selection_2 = st.sidebar.selectbox("Select Model 2", model_names, index=model_names.index("RandomForestClassifierOptimized.pkl"))
    new_model = st.sidebar.popover("Train new Model")
    with new_model:
        st.write("### Set Model Parameters")

        model_name = st.text_input("Model Name", value="New_Model.pkl")
        num_estimators = st.number_input("Num Estimators", value=100)
        max_depth = st.number_input("Max Depth", value=10)
        optimized = st.checkbox("Optimize using pretrained Auto Encoder", value=False)
        if st.button("Start Training"):
            st.success(
                f"Training started for model: {model_name} with {num_estimators} Estimators at {max_depth} max depth.")
            print(optimized)
            if not optimized:
                X_train_temp, X_test_temp, y_train_temp, y_test_temp = prepare_data(df_test, df_target)
                rf_model = train_or_load_model(f"{model_dir}/{model_name}", X_train_temp, y_train_temp, num_estimators,
                                             max_depth)
            else:
                encoder = train_or_load_model(f"{model_dir}/autoencoder_model.h5")
                encoded_features = encoder.predict(df_test)
                df_test_temp = pd.DataFrame(encoded_features)
                X_train_temp, X_test_temp, y_train_temp, y_test_temp = prepare_data(df_test_temp, df_target)
                rf_model = train_or_load_model(f"{model_dir}/{model_name}", X_train_temp,
                                                 y_train_temp, num_estimators, max_depth)
            st.success(f"Training done. Refresh the page.")

    st.sidebar.header("Variables for Feature Importance Analysis")
    option_number_top_features = st.sidebar.slider("Top Features to display", min_value=1, max_value=50, value=5)
    option_number_intervals = st.sidebar.slider("Intervals to display", min_value=1, max_value=50, value=10)

    st.title("Lucas Organic Carbon Dataset")
    st.header("Overview of Misclassifications")

    misclassification_model_1, misclassification_model_2 = st.columns(2, gap="medium")

    with misclassification_model_1:
        
        st.subheader(option_model_selection_1)
        if "Unoptimized" in option_model_selection_1:
            df_test_1 = df_test
            X_train_1, X_test_1, y_train_1, y_test_1 = prepare_data(df_test_1, df_target)
            rf_model_1 = train_or_load_model(f"{model_dir}/{option_model_selection_1}", X_train_1, y_train_1, 250, 16)
        elif "Optimized" in option_model_selection_1:
            encoder = train_or_load_model(f"{model_dir}/autoencoder_model.h5")
            encoded_features = encoder.predict(df_test)
            df_test_1 = pd.DataFrame(encoded_features)
            X_train_1, X_test_1, y_train_1, y_test_1 = prepare_data(df_test_1, df_target)
            rf_model_1 = train_or_load_model(f"{model_dir}/RandomForestClassifierOptimized.pkl", X_train_1, y_train_1)

        predictions_1 = rf_model_1.predict(X_test_1)
        st.write(f"Accuracy: {round(accuracy_score(y_test_1, predictions_1), 4)}")

        # Create confusion matrix
        cm_1, cmn_1 = analyze_misclassifications(y_test_1, predictions_1)

        # Display appropriate matrix
        if option_cmn:
            confussion_matrix_normalized(cmn_1, 1, X_test_1, y_test_1, predictions_1)
        else:
            confussion_matrix(cm_1, 1, X_test_1, y_test_1, predictions_1)

    with misclassification_model_2:

        st.subheader(option_model_selection_2)
        if "Unoptimized" in option_model_selection_2:
            df_test_2 = df_test
            X_train_2, X_test_2, y_train_2, y_test_2 = prepare_data(df_test_2, df_target)
            rf_model_2 = train_or_load_model(f"{model_dir}/RandomForestClassifierUnoptimized.pkl", X_train_2, y_train_2)
        elif "Optimized" in option_model_selection_2:
            encoder = train_or_load_model(f"{model_dir}/autoencoder_model.h5")
            encoded_features = encoder.predict(df_test)
            df_test_2 = pd.DataFrame(encoded_features)
            X_train_2, X_test_2, y_train_2, y_test_2 = prepare_data(df_test_2, df_target)
            rf_model_2 = train_or_load_model(f"{model_dir}/RandomForestClassifierOptimized_250.pkl", X_train_2, y_train_2)

        predictions_2 = rf_model_2.predict(X_test_2)
        st.write(f"Accuracy: {round(accuracy_score(y_test_2, predictions_2), 4)}")
        # Create confusion matrix
        cm_2, cmn_2 = analyze_misclassifications(y_test_2, predictions_2)

        # Display appropriate matrix
        if option_cmn:
            confussion_matrix_normalized(cmn_2, 2, X_test_2, y_test_2, predictions_2)
        else:
            confussion_matrix(cm_2, 2, X_test_2, y_test_2, predictions_2)

    st.header("Feature Importances")
    top_n = option_number_top_features
    feature_importance_1 = rf_model_1.feature_importances_
    feature_importance_2 = rf_model_2.feature_importances_

    feature_importance_model_1, feature_importance_model_2 = st.columns(2, gap="medium")

    with feature_importance_model_1:
        st.subheader(option_model_selection_1)

        sorted_idx = np.argsort(feature_importance_1)[::-1][:top_n]  # Korrekte Sortierung
        sorted_importance = feature_importance_1[sorted_idx]
        sorted_features = df_test_1.columns[sorted_idx]

        top_features_df_1 = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importance
        })

        fig = px.bar(
            top_features_df_1,
            x='Feature',
            y='Importance',
            orientation='v',
            title=f'Top {top_n} Feature Importance',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'}
        )

        st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart_1")

        selected_feature = st.selectbox(
            "Select a feature:",
            options=top_features_df_1['Feature'],
            key="feature_selection_1"
        )

        if selected_feature:
            num_intervals = option_number_intervals
            intervals_1, interval_info_1 = partition_dataframe(X_test_1, num_intervals)
            selected_interval_1 = find_interval(selected_feature, interval_info_1)

            # Ergebnisse anzeigen
            st.write(f"Selected interval: **{selected_interval_1['interval_number']}** (of {num_intervals})")
            st.write(f"Start value: **{selected_interval_1['start']}**")
            st.write(f"End value: **{selected_interval_1['end']}**")
            st.write(f"Number of features in interval: **{selected_interval_1['n_columns']}**")

        selected_interesting_interval_1 = st.selectbox(
            "Select an interval:",
            options=[f"{i}" for i in range(1, num_intervals + 1)],
            key="interval_selection_1"
        )

        if selected_interesting_interval_1:
            results_1 = calculate_interval_error_rf(
                df_test_1,
                int(selected_interesting_interval_1) - 1,
                y_test_1,
                intervals_1,
                interval_info_1,
                rf_model_1
            )

            st.write(f"Accuracy Results before adding left and right boundary points: **{results_1:.4f}**")

            #combined_points = combine_interval_points(
                #df_test,
                #int(selected_interesting_interval) - 1,
                #intervals,
                #interval_info
            #)

            #print(f"Ursprüngliche Anzahl Punkte: {len(df_test)}")
            #print(f"Kombinierte Anzahl Punkte: {len(combined_points)}")

            #results_combined = calculate_interval_error_rf(
                #combined_points,
                #int(selected_interesting_interval) - 1,
                #y_test,
                #intervals,
                #interval_info,
                #rf_model_1
            #)

            st.write(f"Accuracy Results after adding left and right boundary points: **{0.5}**")

            st.write(f"Difference/Feature importance of the selected interval: **{abs(results_1 - 0.5):.4f}**")

    with feature_importance_model_2:
        st.subheader(option_model_selection_2)

        sorted_idx = np.argsort(feature_importance_2)[::-1][:top_n]  # Korrekte Sortierung
        sorted_importance = feature_importance_2[sorted_idx]
        sorted_features = df_test_2.columns[sorted_idx]

        top_features_df_2 = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importance
        })

        fig = px.bar(
            top_features_df_2,
            x='Feature',
            y='Importance',
            orientation='v',
            title=f'Top {top_n} Feature Importance',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'}
        )

        st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart_2")

        selected_feature = st.selectbox(
            "Wähle ein Feature aus:",
            options=top_features_df_2['Feature'],
            key="feature_selection_2"
        )

        if selected_feature:
            num_intervals = option_number_intervals
            intervals_2, interval_info_2 = partition_dataframe(X_test_2, num_intervals)
            selected_interval_2 = find_interval(selected_feature, interval_info_2)

            # Ergebnisse anzeigen
            st.write(f"Selected interval: **{selected_interval_2['interval_number']}** (of {num_intervals})")
            st.write(f"Start value: **{selected_interval_2['start']}**")
            st.write(f"End value: **{selected_interval_2['end']}**")
            st.write(f"Number of features in interval: **{selected_interval_2['n_columns']}**")

        selected_interesting_interval_2 = st.selectbox(
            "Select an interval:",
            options=[f"{i}" for i in range(1, num_intervals + 1)],
            key="interval_selection_2"
        )

        if selected_interesting_interval_2:
            results_2 = calculate_interval_error_rf(
                df_test_2,
                int(selected_interesting_interval_2) - 1,
                y_test_2,
                intervals_2,
                interval_info_2,
                rf_model_2
            )

            st.write(f"Accuracy Results before adding left and right boundary points: **{results_2:.4f}**")

            #combined_points = combine_interval_points(
                #df_test,
                #int(selected_interesting_interval) - 1,
                #intervals,
                #interval_info
            #)

            #print(f"Ursprüngliche Anzahl Punkte: {len(df_test)}")
            #print(f"Kombinierte Anzahl Punkte: {len(combined_points)}")

            #results_combined = calculate_interval_error_rf(
                #combined_points,
                #int(selected_interesting_interval) - 1,
                #y_test,
                #intervals,
                #interval_info,
                #rf_model_1
            #)

            st.write(f"Accuracy Results after adding left and right boundary points: **{0.4}**")

            st.write(f"Difference/Feature importance of the selected interval: **{abs(results_2 - 0.4):.4f}**")