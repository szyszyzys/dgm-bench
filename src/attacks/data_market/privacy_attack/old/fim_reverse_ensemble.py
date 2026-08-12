import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def ensemble_reconstruction(method_results, weights=None):
    """
    Combines multiple inference results into a final estimate.

    Parameters:
    - method_results: List of dictionaries containing 'x_test_hat' from different methods
    - weights: List of weights for each method (optional)

    Returns:
    - x_test_final: Final reconstructed x_test vector
    """
    num_methods = len(method_results)
    n_features = method_results[0]['x_test_hat'].shape[0]

    if weights is None:
        weights = np.ones(num_methods)
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize weights

    x_final = np.zeros(n_features)
    for i, result in enumerate(method_results):
        x_final += weights[i] * result['x_test_hat']

    return x_final


def plot_final_reconstruction(x_true, x_final, title='Final Ensemble Reconstruction'):
    """
    Plots the true test data vs. final reconstructed test data.
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_true))
    plt.plot(indices, x_true, label='True x_test', marker='o')
    plt.plot(indices, x_final, label='Final Reconstructed x_test', marker='x')
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.close()


def run_ensemble_experiment(
        n_selected=100,
        n_unselected=200,
        n_features=10,
        epsilon=1e-3,
        lambda_reg=1e-5,
        gamma=0.1,
        n_components=5,
        top_k=2,
        n_clusters=3,
        num_shadow_models=10,
        verbose=True
):
    """
    Runs an ensemble of inference methods and combines their results.

    Parameters:
    - All parameters as defined in previous methods
    - verbose: If True, prints details

    Returns:
    - final_results: Dictionary containing the final inference results
    """
    # Generate synthetic data
    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # 1. Optimization-Based Reconstruction
    from scipy.optimize import minimize
    from sklearn.linear_model import Ridge

    def optimization_based_reconstruction(fim_inv, initial_guess, bounds=None):
        def loss(x):
            return x.T @ fim_inv @ x

        result = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')
        return result.x

    def compute_extended_fim(X_selected, X_unselected, epsilon=epsilon, lambda_reg=lambda_reg):
        W_selected = np.ones(X_selected.shape[0])
        I_selected = X_selected.T @ (W_selected[:, np.newaxis] * X_selected)

        W_unselected = epsilon * np.ones(X_unselected.shape[0])
        I_unselected = X_unselected.T @ (W_unselected[:, np.newaxis] * X_unselected)

        fim = I_selected + I_unselected + lambda_reg * np.eye(X_selected.shape[1])
        return fim

    fim = compute_extended_fim(X_selected, X_unselected, epsilon=epsilon, lambda_reg=lambda_reg)
    try:
        fim_inv = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        fim += 1e-3 * np.eye(fim.shape[0])
        fim_inv = np.linalg.inv(fim)

    initial_guess = np.mean(X_selected, axis=0)
    x_hat_opt = optimization_based_reconstruction(fim_inv, initial_guess)

    mse_opt = np.mean((x_test - x_hat_opt) ** 2)
    cosine_sim_opt = cosine_similarity([x_test], [x_hat_opt])[0, 0]
    if verbose:
        print(f"Optimization-Based MSE: {mse_opt:.6f}, Cosine Similarity: {cosine_sim_opt:.6f}")
        plot_reconstruction_comparison(x_test, x_hat_opt, title='Optimization-Based Reconstruction')

    opt_result = {
        'x_test_hat': x_hat_opt,
        'mse': mse_opt,
        'cosine_similarity': cosine_sim_opt
    }

    # 2. Bayesian Inference
    # (Assuming the bayesian_inference function is defined as in previous code)
    def bayesian_inference_simple(X_selected, X_unselected, x_test, epsilon=epsilon, lambda_reg=lambda_reg,
                                  verbose=True):
        mu_prior = np.zeros(n_features)
        Sigma_prior = np.eye(n_features)
        fim = compute_extended_fim(X_selected, X_unselected, epsilon=epsilon, lambda_reg=lambda_reg)
        try:
            fim_inv = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            fim += 1e-3 * np.eye(fim.shape[0])
            fim_inv = np.linalg.inv(fim)

        Sigma_posterior = np.linalg.inv(fim_inv + np.linalg.inv(Sigma_prior))
        x_posterior = Sigma_posterior @ (fim_inv @ np.mean(X_selected, axis=0) + np.linalg.inv(Sigma_prior) @ mu_prior)

        # Optimization to find mode
        def negative_log_posterior(x):
            nlp = 0.5 * (x - mu_prior).T @ np.linalg.inv(Sigma_prior) @ (x - mu_prior)
            nll = 0.5 * x.T @ fim_inv @ x
            return nlp + nll

        from scipy.optimize import minimize
        result = minimize(negative_log_posterior, np.mean(X_selected, axis=0), method='L-BFGS-B')
        x_posterior_opt = result.x

        mse = np.mean((x_test - x_posterior_opt) ** 2)
        cosine_sim = cosine_similarity([x_test], [x_posterior_opt])[0, 0]

        if verbose:
            print(f"Bayesian Inference MSE: {mse:.6f}, Cosine Similarity: {cosine_sim:.6f}")
            plot_reconstruction_comparison(x_test, x_posterior_opt, title='Bayesian Inference Reconstruction')

        return {
            'x_test_hat': x_posterior_opt,
            'mse': mse,
            'cosine_similarity': cosine_sim
        }

    bayesian_result = bayesian_inference_simple(X_selected, X_unselected, x_test, epsilon=epsilon,
                                                lambda_reg=lambda_reg, verbose=verbose)

    # 3. Kernel-Based Inference
    from sklearn.decomposition import KernelPCA
    from sklearn.linear_model import Ridge

    def kernel_based_reconstruction_simple(X_selected, X_unselected, x_test, gamma=0.1, n_components=5, lambda_reg=1e-5,
                                           verbose=True):
        X_all = np.vstack((X_selected, X_unselected))
        kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
        X_kpca = kpca.fit_transform(X_all)
        x_test_kpca = kpca.transform(x_test.reshape(1, -1))
        ridge = Ridge(alpha=lambda_reg)
        ridge.fit(X_kpca, X_all)
        x_hat = ridge.predict(x_test_kpca).flatten()
        mse = np.mean((x_test - x_hat) ** 2)
        cosine_sim = cosine_similarity([x_test], [x_hat])[0, 0]
        if verbose:
            print(f"Kernel-Based Reconstruction MSE: {mse:.6f}, Cosine Similarity: {cosine_sim:.6f}")
            plot_reconstruction_comparison(x_test, x_hat, title='Kernel-Based Reconstruction')
        return {
            'x_test_hat': x_hat,
            'mse': mse,
            'cosine_similarity': cosine_sim
        }

    kernel_result = kernel_based_reconstruction_simple(X_selected, X_unselected, x_test, gamma=gamma,
                                                       n_components=n_components, lambda_reg=lambda_reg,
                                                       verbose=verbose)

    # 4. Shadow Model Inference
    # Reusing the earlier shadow model code
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.cluster import KMeans

    def train_shadow_models_simple(X_selected, X_unselected, num_models=5, test_size=0.2, random_state=42):
        shadow_models = []
        combined_X = np.vstack((X_selected, X_unselected))
        labels = np.hstack((np.ones(len(X_selected)), np.zeros(len(X_unselected))))

        for i in range(num_models):
            X_train, X_val, y_train, y_val = train_test_split(combined_X, labels, test_size=test_size,
                                                              random_state=random_state + i)
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"Shadow Model {i + 1}/{num_models} - Validation Accuracy: {acc:.4f}")
            shadow_models.append(model)

        return shadow_models

    def infer_x_test_shadow_model_simple(X_selected, X_unselected, x_test, shadow_models, epsilon=1e-3, lambda_reg=1e-5,
                                         verbose=True):
        X_all = np.vstack((X_selected, X_unselected))
        avg_probs = np.zeros(X_all.shape[0])
        for model in shadow_models:
            avg_probs += model.predict_proba(X_all)[:, 1]
        avg_probs /= len(shadow_models)

        W = avg_probs
        W_unselected = epsilon * np.ones(X_unselected.shape[0])
        W[:X_selected.shape[0]] = W[:X_selected.shape[0]]

        fim = X_all.T @ (W[:, np.newaxis] * X_all) + lambda_reg * np.eye(X_all.shape[1])
        try:
            fim_inv = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            fim += 1e-3 * np.eye(fim.shape[0])
            fim_inv = np.linalg.inv(fim)

        inferred_x = fim_inv @ (X_all.T @ W)

        mse = np.mean((x_test - inferred_x) ** 2)
        cosine_sim = cosine_similarity([x_test], [inferred_x])[0, 0]

        if verbose:
            print(f"Shadow Model Inference MSE: {mse:.6f}, Cosine Similarity: {cosine_sim:.6f}")
            plot_reconstruction_comparison(x_test, inferred_x, title='Shadow Model Inference Reconstruction')

        return {
            'x_test_hat': inferred_x,
            'mse': mse,
            'cosine_similarity': cosine_sim
        }

    # Train shadow models
    shadow_models = train_shadow_models_simple(X_selected, X_unselected, num_models=5, test_size=0.2, random_state=42)

    # Perform Shadow Model Inference
    shadow_result = infer_x_test_shadow_model_simple(
        X_selected, X_unselected, x_test, shadow_models, epsilon=epsilon, lambda_reg=lambda_reg, verbose=verbose
    )

    # 5. Ensemble Combination
    method_results = [opt_result, bayesian_result, kernel_result, shadow_result]
    final_x_test = ensemble_reconstruction(method_results, weights=None)  # Equal weights

    # Evaluation
    mse_final = mean_squared_error(x_test, final_x_test)
    cosine_sim_final = cosine_similarity([x_test], [final_x_test])[0, 0]

    if verbose:
        print(f"\nEnsemble Reconstruction MSE: {mse_final:.6f}")
        print(f"Ensemble Cosine Similarity: {cosine_sim_final:.6f}")
        plot_reconstruction_comparison(x_test, final_x_test, title='Ensemble Reconstruction')

    return {
        'final_x_test': final_x_test,
        'mse': mse_final,
        'cosine_similarity': cosine_sim_final
    }


# Example Usage
if __name__ == "__main__":
    results = run_ensemble_experiment()
    print("\n--- Ensemble Inference Summary ---")
    print(f"Final Reconstructed x_test: {results['final_x_test']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.6f}")
