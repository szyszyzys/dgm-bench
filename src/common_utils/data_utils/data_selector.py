import time
from enum import Enum
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, Optional, Union

import numpy as np
from opendataval import dataval
from opendataval.dataloader import DataFetcher
from opendataval.model import RegressionSkLearnWrapper
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# TODO: daved module was removed — re-enable when a replacement is available
# from src.mechanism.data.discovery.daved.src import evaluate_indices


class SelectionStrategy(Enum):
    DAVED_SINGLE_STEP = "daved_single_step"
    DAVED_MULTI_STEP = "daved_multi_step"
    DATASHAPLEY = "DataShapley"
    BETASHAPLEY = "BetaShapley"
    BANZHAF = "banzhaf"
    RANDOM = "random"
    INFLUENCE = "influence"


def compute_exp_design_loss(X_buy, inverse_covariance):
    """
    Compute the experiment design loss.

    Parameters:
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)

    Returns:
    - float: loss value
    """

    # Compute the matrix product E[x_0^T P x_0]
    return np.mean((X_buy @ inverse_covariance) * X_buy) * X_buy.shape[-1]


def compute_neg_gradient(X_sell, X_buy, inverse_covariance):
    """
    Compute the negative gradient vector of the exp design loss.

    Parameters:
    - X_sell: Sellers data matrix of shape (n_sell, d)
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)

    Returns:
    - Gradient vector of shape (n_sell,)
    """

    # Compute the intermediate matrix product  x_i^T P x_0
    product_matrix = X_sell @ inverse_covariance @ X_buy.T

    # Calculate the squared norms of rows E(x_i^T P x_0)^2
    neg_gradient = np.mean(product_matrix ** 2, axis=1)

    return neg_gradient


def opt_step_size(X_sell_data, X_buy, inverse_covariance, old_loss, lower=1e-3):
    """
    Compute the optimal step size to minimize exp design loss along chosen coordinate .

    Parameters:
    - X_sell_data: Sellers data being updated (n_sell, d,)
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)
    - old_loss: previous value of loss
    - lower (float): Lower bound for the step size optimization


    Returns:
    - float: Optimal step size (value in [0,1])
    - float: New loss after applying the optimal step size
    """
    a = old_loss
    prod = (X_sell_data.T @ inverse_covariance) @ X_buy.T
    b = np.mean(prod ** 2)
    c = X_sell_data @ inverse_covariance @ X_sell_data

    loss = lambda x: (1 / (1 - x)) * (a - (x * b) / (1 - x * (1 - c)))
    result = minimize_scalar(loss, bounds=(0, 0.9), method='bounded')
    return result.x, result.fun


def sherman_morrison_update_inverse(A_inv, u, v):
    """
    Update the inverse of a matrix A_inv after a rank-one update (A + uv^T).

    Parameters:
    - A_inv: The inverse of the original matrix A (d,d)
    - u: Column vector u (d,)
    - v: Column vector v (d,)

    Returns:
    - The inverse of (A + uv^T)
    """

    # Calculate the denominator term (1 + v^T * A_inv * u)
    denominator = 1.0 + v.T @ A_inv @ u

    # Calculate the update term (A_inv * u * v^T * A_inv)
    update_term = (A_inv @ u)[:, None] @ (v.T @ A_inv)[None, :]

    # Update the inverse using the Sherman-Morrison formula
    updated_inverse = A_inv - (update_term / denominator)

    return updated_inverse


def daved_design_selection(
        X_sell,
        y_sell,
        X_buy,
        y_buy,
        num_select=10,
        num_iters=1000,
        alpha=0.01,
        line_search=True,
        recompute_interval=50,
        early_stop_threshold=None,
        sampling_selection_error=True,
        costs=None,
        return_grads=False,
        reg_lambda=0.0,
):
    """
    Select data points based on experimental design optimization.

    Parameters:
    - X_sell (numpy.ndarray): Sellers data matrix (n_sell, n_features)
    - y_sell (numpy.ndarray): Sellers target values (n_sell,)
    - X_buy (numpy.ndarray): Buyers data matrix (n_buy, n_features)
    - y_buy (numpy.ndarray): Buyers target values (n_buy,)
    - num_select (int): Number of seller data points to evaluate each step
    - num_iters (int): Number of step iterations
    - alpha (float, optional): Set manual step size for weight update
    - line_search (bool): Whether to use line search for optimizing step size
    - recompute_interval (int): Interval for recomputing the inverse covariance matrix
    - early_stop_threshold (float, optional): Early stopping threshold for step size
    - sampling_selection_error (bool): Whether to use sampling weights for evaluating seller indices
    - costs (numpy.ndarray, optional): Costs associated with each seller data point
    - use_identity (bool): Initialize inverse covariance to be the identity matrix

    Returns:
    - dict: Dictionary containing tracking information of the optimization process
    """

    # initialize seller weights
    n_sell = X_sell.shape[0]
    weights = np.ones(n_sell) / n_sell

    # Compute inverse covariance matrix
    if reg_lambda > 0:
        reg = np.eye(X_sell.shape[1]) @ np.diag(X_sell.std(0))
        cov = X_sell.T @ np.diag(weights) @ X_sell
        reg_cov = (1 - reg_lambda) * cov + reg_lambda * reg
        inv_cov = np.linalg.pinv(reg_cov)
    else:
        inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)

    # experimental design loss i.e. E[X_buy.T @ inv_cov @ X]
    loss = compute_exp_design_loss(X_buy, inv_cov)

    # track losses and errors
    losses = {}
    errors = {}
    coords = {}
    alphas = {}
    if return_grads:
        grads = {}

    if costs is not None:
        err_msg = f"cost vector should have same length as seller data"
        assert costs.shape[0] == n_sell, f"{err_msg}: should be {n_sell}"
        err_msg = f"cost vector should be strictly positive"
        assert (costs > 0).all(), f"{err_msg}"

    for i in tqdm(range(num_iters)):
        # Recomute actual inverse covariance to periodically recalibrate
        if recompute_interval > 0 and i % recompute_interval == 0:
            inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)

        # Pick coordinate with largest gradient to update
        neg_grad = compute_neg_gradient(X_sell, X_buy, inv_cov)

        if return_grads:
            grads[i] = neg_grad

        if costs is not None:
            neg_grad *= 1 / costs

        update_coord = np.argmax(neg_grad)

        coords[i] = update_coord

        # Optimize step size with line search
        if line_search or alpha is None:
            alpha, line_loss = opt_step_size(X_sell[update_coord], X_buy, inv_cov, loss)

        # Terminate early if step size is too small
        if early_stop_threshold is not None and alpha < early_stop_threshold:
            break

        alphas[i] = alpha

        # Update weight vector
        weights *= 1 - alpha  # shrink weights by 1 - alpha
        weights[update_coord] += alpha  # increase magnitude of picked coordinate

        # Update inverse covariance matrix
        inv_cov /= 1 - alpha  # Update with respect to weights shrinking

        # update with respect to picked coordinate increasing
        inv_cov = sherman_morrison_update_inverse(
            inv_cov,
            alpha * X_sell[update_coord, :],
            X_sell[update_coord, :],
            )

        if sampling_selection_error:
            selected_seller_indices = np.random.choice(
                np.arange(weights.shape[0]),
                size=num_select,
                p=weights / weights.sum(),
                replace=False,
            )
        else:
            selected_seller_indices = weights.argsort()[::-1][:num_select]

        results = evaluate_indices(
            X_sell,
            y_sell,
            X_buy,
            y_buy,
            selected_seller_indices,
            inverse_covariance=inv_cov,
        )
        losses[i] = compute_exp_design_loss(X_buy, inv_cov)
        errors[i] = results["mse_error"]

    ret = dict(
        losses=losses,
        errors=errors,
        weights=weights,
        coords=coords,
        alphas=alphas,
    )
    if return_grads:
        ret['grads'] = grads
    return ret


def daved_one_step(X_sell, X_buy):
    """
    Compute one-step baseline
    Parameters:
    - X_sell: Sellers data being updated (n_sell, d)
    - X_buy: Buyer data matrix of shape (n_buy, d)

    Returns:
    - numpy.ndarray: shape (n_sell)
    """
    inv_cov = np.linalg.pinv(np.dot(X_sell.T, X_sell))
    one_step_values = np.mean((X_sell @ inv_cov @ X_buy.T) ** 2, axis=1)
    return one_step_values


def get_selection_general(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        metric=mean_squared_error,
        random_state=0,
        selection_method="DataOob",
        baseline_kwargs={"num_models": 100},
        use_ridge=False,
):
    fetcher = DataFetcher.from_data_splits(
        train_x, train_y, val_x, val_y, test_x, test_y, one_hot=False
    )
    if use_ridge:
        model = RegressionSkLearnWrapper(Ridge)
    else:
        model = RegressionSkLearnWrapper(LinearRegression)

    baseline_kwargs["random_state"] = random_state

    start_time = time.perf_counter()
    print(selection_method.center(40, "-"))
    baseline_value = (
        getattr(dataval, selection_method)(**baseline_kwargs)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"\tTIME: {runtime:.0f}")
    return baseline_value, runtime


def get_error_fixed(
        x_test,
        y_test,
        x_s,
        y_s,
        w,
        eval_range=range(1, 10),
        use_sklearn=False,
        return_list=False,
):
    sorted_w = w.argsort()[::-1]

    errors = {}
    for k in eval_range:
        selected = sorted_w[:k]
        x_k = x_s[selected]
        y_k = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_k, y_k)
            y_hat = LR.predict(x_test)
        else:
            beta_k = np.linalg.pinv(x_k) @ y_k
            y_hat = x_test @ beta_k

        errors[k] = mean_squared_error(y_test, y_hat)

    return list(errors.values()) if return_list else errors


def get_error_under_budget(
        x_test,
        y_test,
        x_s,
        y_s,
        w,
        costs=None,
        eval_range=range(1, 10),
        use_sklearn=False,
        return_list=False,
):
    assert costs is not None, "Missing costs"
    sorted_w = w.argsort()[::-1]
    cum_cost = np.cumsum(costs[sorted_w])

    errors = {}
    for budget in eval_range:
        under_budget_index = np.searchsorted(cum_cost, budget, side="left")

        # Could not find any points under budget constraint
        if under_budget_index == 0:
            continue

        selected = sorted_w[:under_budget_index]
        x_budget = x_s[selected]
        y_budget = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_budget, y_budget)
            y_hat = LR.predict(x_test)
        else:
            beta_budget = np.linalg.pinv(x_budget) @ y_budget
            y_hat = x_test @ beta_budget

        errors[budget] = mean_squared_error(y_test, y_hat)

    # Remove keys with values under budget
    # errors = {k: v for k, v in errors.items() if v is not None}
    return list(errors.values()) if return_list else errors


class DataSelector:
    def __init__(self,
                 x_sell: np.ndarray = None,
                 y_sell: np.ndarray = None,
                 x_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 costs: Optional[np.ndarray] = None, ):
        """
        Initialize DataSelector with seller and validation data
        """
        self.x_sell = x_sell.astype(np.single) if x_sell is not None else None
        self.y_sell = y_sell.astype(np.single) if y_sell is not None else None
        self.x_val = x_val.astype(np.single) if x_val is not None else None
        self.y_val = y_val.astype(np.single) if y_val is not None else None
        self.costs = costs if costs is not None else None
        self.n_samples = len(x_sell) if x_sell is not None else None

    def get_error(self,
                  w_fw,
                  x_test,
                  y_test,
                  x_sell,
                  y_sell,
                  costs=None,
                  max_eval_range=150,
                  eval_step=25
                  ):

        eval_range = list(range(1, 30, 1)) + list(
            range(30, max_eval_range, eval_step)
        )
        err_kwargs = dict(
            x_test=x_test, y_test=y_test, x_s=x_sell, y_s=y_sell, eval_range=eval_range
        )

        if costs is not None:
            error_func = get_error_under_budget
            err_kwargs["costs"] = costs
        else:
            error_func = get_error_fixed
            err_kwargs["return_list"] = True

        err = error_func(w=w_fw, **err_kwargs)
        return err

    def set_sell(
            self,
            x_sell: Optional[np.ndarray] = None,
            y_sell: Optional[np.ndarray] = None,
            costs: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Sets the seller data.

        Parameters:
            x_sell (Optional[np.ndarray]): Features for sellers.
            y_sell (Optional[np.ndarray]): Labels for sellers.
            costs (Optional[Union[np.ndarray, list]]): Costs associated with sellers.

        Raises:
            TypeError: If inputs are not of expected types.
            ValueError: If provided arrays have mismatched lengths.
        """
        if x_sell is not None:
            if not isinstance(x_sell, np.ndarray):
                raise TypeError("x_sell must be a NumPy array.")
            self.x_sell = x_sell

        if y_sell is not None:
            if not isinstance(y_sell, np.ndarray):
                raise TypeError("y_sell must be a NumPy array.")
            self.y_sell = y_sell

        if costs is not None:
            if not isinstance(costs, (np.ndarray, list)):
                raise TypeError("costs must be a NumPy array or a list.")
            self.costs = costs

        # Update n_samples only if x_sell is provided
        if x_sell is not None:
            self.n_samples = len(x_sell)
            # Validate that y_sell and costs have compatible lengths
            if y_sell is not None and len(y_sell) != len(x_sell):
                raise ValueError("y_sell must have the same number of samples as x_sell.")
            if costs is not None and len(costs) != len(x_sell):
                raise ValueError("costs must have the same number of samples as x_sell.")
        elif hasattr(self, 'x_sell') and self.x_sell is not None:
            self.n_samples = len(self.x_sell)
        else:
            self.n_samples = 0

    def select_data(self,
                    x_buy: np.ndarray,
                    y_buy: np.ndarray,
                    strategy: SelectionStrategy,
                    **kwargs) -> Dict[str, np.ndarray]:
        """
        Select data using specified strategy and return weights
        """
        strategy_map = {
            SelectionStrategy.DAVED_SINGLE_STEP: self._daved_single_step_selection,
            SelectionStrategy.DAVED_MULTI_STEP: self._daved_multi_step_selection,
            SelectionStrategy.DATASHAPLEY: self._data_shapley_selection,
            SelectionStrategy.BETASHAPLEY: self._beta_shapley_selection,
            SelectionStrategy.BANZHAF: self._banzhaf_selection,
            SelectionStrategy.RANDOM: self._random_selection,
            SelectionStrategy.INFLUENCE: self._influence_selection
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        weights = strategy_map[strategy](x_buy, y_buy, **kwargs)
        return weights
        # return self._process_weights(weights, **kwargs)

    def select_all_strategies(self,
                              x_buy: np.ndarray,
                              y_buy: np.ndarray,
                              **kwargs) -> Dict[str, Dict]:
        """
        Run all selection strategies and return their weights and metrics
        """
        results = {}
        for strategy in SelectionStrategy:
            start_time = time.perf_counter()
            weights = self.select_data(x_buy, y_buy, strategy, **kwargs)
            end_time = time.perf_counter()

            results[strategy.value] = {
                'weights': weights,
                'runtime': end_time - start_time
            }
        return results

    def get_top_k(self,
                  weights: np.ndarray,
                  k: int,
                  return_indices: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k samples based on weights
        """
        top_indices = np.argsort(weights)[-k:]
        if return_indices:
            return top_indices
        return self.x_sell[top_indices], self.y_sell[top_indices]

    def _daved_single_step_selection(self,
                                     x_buy: np.ndarray,
                                     y_buy: np.ndarray,
                                     **kwargs) -> np.ndarray:
        """Single-step DAVED"""
        weight = daved_one_step(self.x_sell, x_buy)
        return weight

    def _daved_multi_step_selection(self,
                                    x_buy: np.ndarray,
                                    y_buy: np.ndarray,
                                    num_iters: int = 100,
                                    reg_lambda: float = 0.0,
                                    **kwargs) -> np.ndarray:
        """Multi-step Frank-Wolfe selection"""
        # Implementation based on your design_selection function
        res = daved_design_selection(self.x_sell,
                                     self.y_sell,
                                     x_buy,
                                     y_buy,
                                     num_select=10,
                                     num_iters=num_iters,
                                     alpha=None,
                                     recompute_interval=0,
                                     line_search=True,
                                     costs=self.costs,
                                     reg_lambda=reg_lambda, )
        # Add implementation here
        return res["weights"]

    def _data_shapley_selection(self,
                                x_buy: np.ndarray,
                                y_buy: np.ndarray,
                                **kwargs) -> np.ndarray:
        """Shapley value based selection"""
        """Data Shapley selection method"""
        weights, time = get_selection_general(
            self.x_sell,
            self.y_sell,
            x_buy,
            y_buy,
            x_buy,
            y_buy,
            selection_method="DataShapley",
            baseline_kwargs=
            {"mc_epochs": 100, "models_per_iteration": 10}
            ,
        )

        return weights
        # "DataBanzhaf": {"mc_epochs": 100, "models_per_iteration": 10},
        # "BetaShapley": {"mc_epochs": 100, "models_per_iteration": 10},
        # "DataOob": {"num_models": 500},
        # "InfluenceSubsample": {"num_models": 500},

    def _beta_shapley_selection(self,
                                x_buy: np.ndarray,
                                y_buy: np.ndarray,
                                **kwargs) -> np.ndarray:
        """Shapley value based selection"""
        """Data Shapley selection method"""
        weights, time = get_selection_general(
            self.x_sell,
            self.y_sell,
            x_buy,
            y_buy,
            x_buy,
            y_buy,
            selection_method="BetaShapley",
            baseline_kwargs=
            {"mc_epochs": 100, "models_per_iteration": 10}
            ,
        )

        return weights

    def _banzhaf_selection(self,
                           x_buy: np.ndarray,
                           y_buy: np.ndarray,
                           **kwargs) -> np.ndarray:
        """Banzhaf value based selection"""
        weights, time = get_selection_general(
            self.x_sell,
            self.y_sell,
            x_buy,
            y_buy,
            x_buy,
            y_buy,
            selection_method="DataBanzhaf",
            baseline_kwargs=
            {"mc_epochs": 100, "models_per_iteration": 10}
        )

        return weights

    def _dataOob_selection(self,
                           x_buy: np.ndarray,
                           y_buy: np.ndarray,
                           **kwargs) -> np.ndarray:
        """Influence-based selection"""
        weights, time = get_selection_general(
            self.x_sell,
            self.y_sell,
            x_buy,
            y_buy,
            x_buy,
            y_buy,
            selection_method="DataBanzhaf",
            baseline_kwargs={"num_models": 500},
        )

        return weights

    def _influence_selection(self,
                             x_buy: np.ndarray,
                             y_buy: np.ndarray,
                             num_models: int = 500,
                             **kwargs) -> np.ndarray:
        """Influence-based selection"""
        weights, time = get_selection_general(
            self.x_sell,
            self.y_sell,
            x_buy,
            y_buy,
            x_buy,
            y_buy,
            selection_method="InfluenceSubsample",
            baseline_kwargs=
            {"num_models": 500}
        )

        return weights

    def _random_selection(self,
                          x_buy: np.ndarray,
                          y_buy: np.ndarray,
                          **kwargs) -> np.ndarray:
        """Random selection — returns uniform random weights (consistent with other strategies)."""
        return np.random.uniform(size=self.n_samples)

    def _process_weights(self,
                         weights: np.ndarray,
                         normalize: bool = True,
                         **kwargs) -> np.ndarray:
        """Process and normalize weights if needed"""
        if normalize and weights.sum() != 0:
            return weights / weights.sum()
        return weights
