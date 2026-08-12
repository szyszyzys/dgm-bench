import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_inputs(X, y):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be NumPy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be equal.")


def train_linear_regression(X, y):
    validate_inputs(X, y)
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X, y, n_estimators=100, random_state=42):
    validate_inputs(X, y)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model


def train_neural_network(X, y, hidden_layer_sizes=(100,), max_iter=500, random_state=42):
    validate_inputs(X, y)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": mean_squared_error(y, y_pred, squared=False),
        "MAE": mean_absolute_error(y, y_pred),
        "R2_Score": r2_score(y, y_pred),
        "Explained_Variance": explained_variance_score(y, y_pred)
    }
    return metrics


def cross_validated_metrics(model, X, y, cv=5):
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)

    metrics = {
        "CV_MSE": -mse_scores.mean(),
        "CV_MAE": -mae_scores.mean(),
        "CV_R2_Score": r2_scores.mean()
    }
    return metrics


def evaluate_attack_trained_model(
        X_sell, y_sell, X_buy, y_buy, data_indices, inverse_covariance=None, cv_folds=5
):
    """
    Evaluate the performance of a linear regression model trained on selected seller data
    against the buyer's dataset, including cross-validated metrics.

    Parameters:
    - X_sell (numpy.ndarray): Seller's data matrix (n_sell, n_features)
    - y_sell (numpy.ndarray): Seller's target values (n_sell,)
    - X_buy (numpy.ndarray): Buyer's data matrix (n_buy, n_features)
    - y_buy (numpy.ndarray): Buyer's target values (n_buy,)
    - data_indices (numpy.ndarray or list): Indices of selected data points from seller's data
    - inverse_covariance (numpy.ndarray, optional): Inverse covariance matrix (n_features, n_features)
    - cv_folds (int): Number of cross-validation folds

    Returns:
    - dict: Dictionary containing evaluation metrics
    """
    # Select data
    X_selected = X_sell[data_indices]
    y_selected = y_sell[data_indices]

    # Train multiple models
    models = {
        "LinearRegression": train_linear_regression(X_selected, y_selected),
        "RandomForest": train_random_forest(X_selected, y_selected),
        "NeuralNetwork": train_neural_network(X_selected, y_selected)
    }

    # Evaluate on buyer's data
    buyer_metrics = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_buy, y_buy)
        buyer_metrics[f"{name}_MSE"] = metrics["MSE"]
        buyer_metrics[f"{name}_RMSE"] = metrics["RMSE"]
        buyer_metrics[f"{name}_MAE"] = metrics["MAE"]
        buyer_metrics[f"{name}_R2_Score"] = metrics["R2_Score"]
        buyer_metrics[f"{name}_Explained_Variance"] = metrics["Explained_Variance"]

    # Cross-validated metrics on seller's data
    cv_metrics = {}
    for name, model in models.items():
        cv_m = cross_validated_metrics(model, X_selected, y_selected, cv=cv_folds)
        cv_metrics.update({f"{name}_CV_MSE": cv_m["CV_MSE"],
                           f"{name}_CV_MAE": cv_m["CV_MAE"],
                           f"{name}_CV_R2_Score": cv_m["CV_R2_Score"]})

    return {
        "Buyer_Metrics": buyer_metrics,
        "Cross_Validated_Metrics": cv_metrics
    }


def comparative_evaluation(
        X_sell_original, y_sell_original,
        X_sell_modified, y_sell_modified,
        X_buy, y_buy,
        data_indices_original, data_indices_modified,
        cv_folds=5
):
    """
    Compare the performance of models trained on original and modified seller datasets.

    Parameters:
    - X_sell_original (numpy.ndarray): Original seller's data matrix
    - y_sell_original (numpy.ndarray): Original seller's target values
    - X_sell_modified (numpy.ndarray): Modified seller's data matrix
    - y_sell_modified (numpy.ndarray): Modified seller's target values
    - X_buy (numpy.ndarray): Buyer's data matrix
    - y_buy (numpy.ndarray): Buyer's target values
    - data_indices_original (numpy.ndarray or list): Indices for original data selection
    - data_indices_modified (numpy.ndarray or list): Indices for modified data selection
    - cv_folds (int): Number of cross-validation folds

    Returns:
    - dict: Dictionary containing evaluation metrics for both models
    """
    # Evaluate original model
    original_results = evaluate_attack_trained_model(
        X_sell_original, y_sell_original,
        X_buy, y_buy,
        data_indices_original,
        cv_folds=cv_folds
    )

    # Evaluate modified model
    modified_results = evaluate_attack_trained_model(
        X_sell_modified, y_sell_modified,
        X_buy, y_buy,
        data_indices_modified,
        cv_folds=cv_folds
    )

    # Prepare comparison
    comparison_metrics = {
        "Original_Model": original_results["Buyer_Metrics"],
        "Modified_Model": modified_results["Buyer_Metrics"],
        "Original_Model_CV": original_results["Cross_Validated_Metrics"],
        "Modified_Model_CV": modified_results["Cross_Validated_Metrics"]
    }

    return comparison_metrics


def plot_evaluation_metrics(comparison_metrics):
    """
    Plot evaluation metrics for original and modified models across different algorithms.

    Parameters:
    - comparison_metrics (dict): Dictionary containing metrics for both models
    """
    # Extract unique metric names
    metrics = list(next(iter(comparison_metrics.values())).keys())

    # Prepare data for plotting
    original = [comparison_metrics["Original_Model"][metric] for metric in metrics]
    modified = [comparison_metrics["Modified_Model"][metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width / 2, original, width, label='Original Model')
    rects2 = ax.bar(x + width / 2, modified, width, label='Modified Model')

    # Add labels and title
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Evaluation Metrics Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    # Add text labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.close()


def plot_residuals(model, X, y, title="Residuals Plot"):
    """
    Plot residuals of a regression model.

    Parameters:
    - model (sklearn.base.RegressorMixin): Trained regression model
    - X (numpy.ndarray): Input feature matrix
    - y (numpy.ndarray): Actual target values
    - title (str): Title of the plot
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.close()


def comprehensive_evaluation(
        X_sell_original, y_sell_original,
        X_sell_modified, y_sell_modified,
        X_buy, y_buy,
        data_indices_original, data_indices_modified,
        cv_folds=5
):
    """
    Perform a comprehensive evaluation comparing models trained on original and modified datasets.

    Parameters:
    - X_sell_original (numpy.ndarray): Original seller's data matrix
    - y_sell_original (numpy.ndarray): Original seller's target values
    - X_sell_modified (numpy.ndarray): Modified seller's data matrix
    - y_sell_modified (numpy.ndarray): Modified seller's target values
    - X_buy (numpy.ndarray): Buyer's data matrix
    - y_buy (numpy.ndarray): Buyer's target values
    - data_indices_original (numpy.ndarray or list): Indices for original data selection
    - data_indices_modified (numpy.ndarray or list): Indices for modified data selection
    - cv_folds (int): Number of cross-validation folds

    Returns:
    - None: Displays comparison plots and prints metrics
    """
    # Evaluate original model
    original_results = evaluate_attack_trained_model(
        X_sell_original, y_sell_original,
        X_buy, y_buy,
        data_indices_original,
        cv_folds=cv_folds
    )

    # Evaluate modified model
    modified_results = evaluate_attack_trained_model(
        X_sell_modified, y_sell_modified,
        X_buy, y_buy,
        data_indices_modified,
        cv_folds=cv_folds
    )

    # Prepare comparison
    comparison_metrics = {
        "Original_Model": original_results["Buyer_Metrics"],
        "Modified_Model": modified_results["Buyer_Metrics"],
        "Original_Model_CV": original_results["Cross_Validated_Metrics"],
        "Modified_Model_CV": modified_results["Cross_Validated_Metrics"]
    }

    # Log comparison
    logging.info("Comparison of Evaluation Metrics:")
    for model_name, metrics in comparison_metrics.items():
        logging.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

    # Plot metrics
    plot_evaluation_metrics(comparison_metrics)

    # Residual Analysis for each model type
    # Re-train models to get residuals (ensure same training as in evaluation)
    models = {
        "Original_LinearRegression": train_linear_regression(X_sell_original[data_indices_original],
                                                             y_sell_original[data_indices_original]),
        "Original_RandomForest": train_random_forest(X_sell_original[data_indices_original],
                                                     y_sell_original[data_indices_original]),
        "Original_NeuralNetwork": train_neural_network(X_sell_original[data_indices_original],
                                                       y_sell_original[data_indices_original]),
        "Modified_LinearRegression": train_linear_regression(X_sell_modified[data_indices_modified],
                                                             y_sell_modified[data_indices_modified]),
        "Modified_RandomForest": train_random_forest(X_sell_modified[data_indices_modified],
                                                     y_sell_modified[data_indices_modified]),
        "Modified_NeuralNetwork": train_neural_network(X_sell_modified[data_indices_modified],
                                                       y_sell_modified[data_indices_modified])
    }

    for name, model in models.items():
        plot_residuals(model, X_buy, y_buy, title=f"Residuals - {name}")
