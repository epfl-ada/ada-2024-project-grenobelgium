import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def sorted_channel_list(df):
    """
    This function returns the list of channels (sorted by total views)
    """
    table = df.groupby('channel')['delta_views'].sum().reset_index().sort_values(by='delta_views', ascending=False)
    return table['channel'].tolist()

def get_top_channels(df, k):
    """
    Returns the list of top k channels (ranked by total views)
    """
    channel_list = sorted_channel_list(df)
    return channel_list[:k]

def get_top_k_channel_df(df, k, cluster1, cluster2):
    """
    This function return the dataframe of the top k channels of two given clusters 
    """
    df = df.copy()

    df1 = df[df['category'].isin(cluster1)]
    df2 = df[df['category'].isin(cluster2)]
    top_channels_cluster1 = get_top_channels(df1, k)
    top_channels_cluster2 = get_top_channels(df2, k)

    return df[df['channel'].isin(top_channels_cluster1+top_channels_cluster2)]

def drop_useless_features(df):
    return df.drop(['videos', 'activity', 'delta_videos', 'delta_views', 'delta_subs'], axis=1)

def calculate_growth_ratios(df):
    """
    Computes growth ratios for views and subscribers for each channel.
    """
    df = df.sort_values(['channel', 'datetime'])
    df[['growth_views', 'growth_subs']] = df.groupby('channel')[['views', 'subs']].pct_change() * 100
    df[['growth_views', 'growth_subs']] = df[['growth_views', 'growth_subs']].fillna(0)
    mask_views = df['views'].shift(1) <= 0
    mask_subs = df['subs'].shift(1) <= 0
    df.loc[mask_views, 'growth_views'] = 0
    df.loc[mask_subs, 'growth_subs'] = 0
    df = df.drop(['views', 'subs'], axis=1)

    return df

def align_time_series(df):
    """
    Aligns time series data for channels by assigning an 'aligned_week' starting from week 0.
    """
    df = df.sort_values(['channel', 'datetime']).reset_index(drop=True)
    grouped = df.groupby('channel')
    df['start_date'] = grouped['datetime'].transform('first')
    df['delta_days'] = (df['datetime'] - df['start_date']).dt.days
    df['aligned_week'] = df['delta_days'] // 7
    df = df.drop(['start_date', 'delta_days', 'datetime'], axis=1)
    
    return df


def transform_time_series(df, cluster1):
    """
    Transforms the input DataFrame into a time series format with specified column names.
    """
    # Dictionnary that maps the channels with their category
    channel_category = df.drop_duplicates('channel').set_index('channel')['category'].to_dict()
    
    cluster1_set = set(cluster1)
    grouped = df.groupby('channel')
    
    final_data = []
    for channel, group in grouped:
        group_sorted = group.sort_values('aligned_week')
        growth_views = group_sorted['growth_views'].tolist()
        growth_subs = group_sorted['growth_subs'].tolist()
        
        selected_growth_views = growth_views[1:53]
        selected_growth_subs = growth_subs[1:53]
        
        if len(selected_growth_views) < 52:
            selected_growth_views += [0] * (52 - len(selected_growth_views))
        if len(selected_growth_subs) < 52:
            selected_growth_subs += [0] * (52 - len(selected_growth_subs))
        
        cluster_label = 'entertainment' if channel_category[channel] in cluster1_set else 'educational'
        
        line = selected_growth_views + selected_growth_subs + [cluster_label]
        final_data.append(line)
    
    view_columns = [f'view{i}' for i in range(1, 53)]
    sub_columns = [f'sub{i}' for i in range(1, 53)]
    column_names = view_columns + sub_columns + ['cluster']
    
    transformed_df = pd.DataFrame(final_data, columns=column_names)
    
    return transformed_df


def full_transformation_pipeline(df, cluster1):
    df = drop_useless_features(df)
    df = calculate_growth_ratios(df)
    df = align_time_series(df)
    df = transform_time_series(df, cluster1)
    return df


def build_additional_features(df):
    """
    Constructs additional statistical and variability-based features for views and subscribers.
    """
    # Identify and sort columns for views and subscribers
    view_cols = [col for col in df.columns if col.startswith('view')]
    sub_cols = [col for col in df.columns if col.startswith('sub')]
    view_cols = sorted(view_cols, key=lambda x: int(x.replace('view', '')))
    sub_cols = sorted(sub_cols, key=lambda x: int(x.replace('sub', '')))
    
    df_new = df.copy()
    
    # Compute statistical features for views
    df_new['view_mean'] = df_new[view_cols].mean(axis=1)
    df_new['view_median'] = df_new[view_cols].median(axis=1)
    df_new['view_std'] = df_new[view_cols].std(axis=1)
    df_new['view_min'] = df_new[view_cols].min(axis=1)
    df_new['view_max'] = df_new[view_cols].max(axis=1)
    df_new['view_range'] = df_new['view_max'] - df_new['view_min']
    
    # Compute statistical features for subscribers
    df_new['sub_mean'] = df_new[sub_cols].mean(axis=1)
    df_new['sub_median'] = df_new[sub_cols].median(axis=1)
    df_new['sub_std'] = df_new[sub_cols].std(axis=1)
    df_new['sub_min'] = df_new[sub_cols].min(axis=1)
    df_new['sub_max'] = df_new[sub_cols].max(axis=1)
    df_new['sub_range'] = df_new['sub_max'] - df_new['sub_min']
    
    # Calculate week-to-week variability (standard deviation of differences)
    view_diff = df_new[view_cols].diff(axis=1).fillna(0)
    sub_diff = df_new[sub_cols].diff(axis=1).fillna(0)
    df_new['view_diff_std'] = view_diff.std(axis=1)
    df_new['sub_diff_std'] = sub_diff.std(axis=1)
    
    # Count weeks with extreme spikes (values significantly higher than the mean)
    df_new['view_spike_count'] = (df_new[view_cols].gt(df_new['view_mean'] + df_new['view_std'], axis=0)).sum(axis=1)
    df_new['sub_spike_count'] = (df_new[sub_cols].gt(df_new['sub_mean'] + df_new['sub_std'], axis=0)).sum(axis=1)
    
    return df_new


def preprocessing_pipeline(data, add_features=False):
    """
    Prepares the dataset for training and testing by optionally adding features,
    separating features and target labels, and splitting into training and test sets.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - add_features: bool, whether to add additional features using build_additional_features.

    Returns:
    - X_train, X_test: Feature sets for training and testing.
    - y_train, y_test: Target labels for training and testing.
    """
    # Optionally add additional features to the dataset
    if add_features:
        data = build_additional_features(data)
    
    # Separate features (X) and target labels (y)
    X = data.drop(['cluster'], axis=1)
    # Encode target labels: 0 for 'entertainment', 1 for 'educational'
    y = data['cluster'].apply(lambda x: 0 if x == 'entertainment' else 1)

    # Split data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def compare_logistic_regression_random_forest(data):
    """
    This function compares Logistic Regression and Random Forest models on a given dataset.
    It evaluates the performance with and without additional features, tuning hyperparameters for each model.
    
    Parameters:
    - data: pandas DataFrame containing the dataset.
    
    Returns:
    - A pandas DataFrame with the performance comparison (accuracy and F1-score) for both models,
      with and without additional features.
    """
    
    results = []  # List to store the results for each evaluation
    
    # Options for whether to add additional features or not
    add_features_options = [False, True]
    
    # Hyperparameters grid for Logistic Regression
    param_grid_logreg = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],              # Regularization type
        'solver': ['lbfgs'],            # Solver to use
        'max_iter': [1000]              # Maximum iterations for convergence
    }
    
    # Hyperparameters grid for Random Forest
    param_grid_rf = {
        'n_estimators': [1000]  # Number of trees in the forest
    }
    
    # Loop through the two options for adding features
    for add_features in add_features_options:
        print(f"\n=== Evaluating with add_features={add_features} ===")
        
        # Preprocess the data (split into train/test and add features if specified)
        X_train, X_test, y_train, y_test = preprocessing_pipeline(data, add_features=add_features)
        
        # Logistic Regression Model Evaluation
        print("\n--- Logistic Regression ---")
        logreg = LogisticRegression(random_state=42)
        
        # Perform grid search to find the best hyperparameters
        grid_search_logreg = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid_logreg,
            scoring='accuracy',  # Evaluate based on accuracy
            cv=3,  # 3-fold cross-validation
            n_jobs=-1,  # Use all CPU cores
            verbose=0  # Suppress output
        )
        grid_search_logreg.fit(X_train, y_train)  # Fit model
        
        best_logreg = grid_search_logreg.best_estimator_  # Best model after hyperparameter tuning
        print(f"Best Hyperparameters: {grid_search_logreg.best_params_}")
        
        # Predict and evaluate on test and train data
        y_pred_test_logreg = best_logreg.predict(X_test)
        y_pred_train_logreg = best_logreg.predict(X_train)
        
        test_accuracy_logreg = accuracy_score(y_test, y_pred_test_logreg)  # Accuracy on test set
        test_f1_logreg = f1_score(y_test, y_pred_test_logreg)  # F1-Score on test set
        train_accuracy_logreg = accuracy_score(y_train, y_pred_train_logreg)  # Accuracy on train set
        train_f1_logreg = f1_score(y_train, y_pred_train_logreg)  # F1-Score on train set
        
        print(f"Logistic Regression Training Accuracy: {train_accuracy_logreg}, Training F1-Score: {train_f1_logreg}")
        print(f"Logistic Regression Testing Accuracy: {test_accuracy_logreg}, Testing F1-Score: {test_f1_logreg}")
        
        # Store the results
        results.append({
            'Add Features': add_features,
            'Model': 'Logistic Regression',
            'Accuracy': test_accuracy_logreg,
            'F1-Score': test_f1_logreg
        })
        
        # Random Forest Model Evaluation
        print("\n--- Random Forest ---")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search for Random Forest hyperparameters
        grid_search_rf = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_rf,
            scoring='accuracy',  # Evaluate based on accuracy
            cv=3,  # 3-fold cross-validation
            n_jobs=-1,  # Use all CPU cores
            verbose=0  # Suppress output
        )
        grid_search_rf.fit(X_train, y_train)  # Fit model
        
        best_rf = grid_search_rf.best_estimator_  # Best model after hyperparameter tuning
        print(f"Best Hyperparameters: {grid_search_rf.best_params_}")
        
        # Predict and evaluate on test and train data
        y_pred_test_rf = best_rf.predict(X_test)
        y_pred_train_rf = best_rf.predict(X_train)
        
        test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)  # Accuracy on test set
        test_f1_rf = f1_score(y_test, y_pred_test_rf)  # F1-Score on test set
        train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)  # Accuracy on train set
        train_f1_rf = f1_score(y_train, y_pred_train_rf)  # F1-Score on train set
        
        print(f"Random Forest Training Accuracy: {train_accuracy_rf}, Training F1-Score: {train_f1_rf}")
        print(f"Random Forest Testing Accuracy: {test_accuracy_rf}, Testing F1-Score: {test_f1_rf}")
        
        # Store the results
        results.append({
            'Add Features': add_features,
            'Model': 'Random Forest',
            'Accuracy': test_accuracy_rf,
            'F1-Score': test_f1_rf
        })
    
    # Convert results to a DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    
    # Pivot the DataFrame to compare results
    results_pivot = results_df.pivot(index='Add Features', columns='Model', values=['Accuracy', 'F1-Score'])
    results_pivot = results_pivot.round(4)  # Round to 4 decimal places
    
    print("\n=== Performance Comparison ===")
    print(results_pivot)  # Print the comparison
    
    return results_pivot  # Return the comparison DataFrame
