import os
import pickle
from typing import List, Dict, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class TestSetBuilder:
    def __init__(self, seller_data: pd.DataFrame, selection_func: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Initialize the TestSetBuilder.

        :param seller_data: DataFrame containing the seller's dataset.
        :param selection_func: Function that takes a query DataFrame and returns selected DataFrame.
        """
        self.seller_data = seller_data
        self.selection_func = selection_func
        self.test_set = []

    def generate_query(self, method: str = 'random_sampling', query_size: int = 10, perturb: bool = False,
                       perturb_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate a single query based on the specified method.

        :param method: Method to generate queries ('random_sampling' or 'cluster_based').
        :param query_size: Number of data points in the query.
        :param perturb: Whether to add perturbations to the query.
        :param perturb_params: Parameters for perturbation (e.g., noise level).
        :return: DataFrame representing the query.
        """
        if method == 'random_sampling':
            query = self.seller_data.sample(n=query_size, replace=False).reset_index(drop=True)
        elif method == 'cluster_based':
            # Placeholder for cluster-based sampling
            # Implement cluster-based sampling as needed
            # For simplicity, using random sampling here
            query = self.seller_data.sample(n=query_size, replace=False).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown query generation method: {method}")

        if perturb:
            if perturb_params is None:
                noise_level = 0.01
            else:
                noise_level = perturb_params.get('noise_level', 0.01)
            # Assuming numeric features; adjust as necessary for your data
            query = query + np.random.normal(0, noise_level, query.shape)
            # Ensure data types are maintained
            query = pd.DataFrame(query, columns=self.seller_data.columns)

        return query

    def build_test_set(self, num_queries: int = 100, query_size: int = 10, method: str = 'random_sampling',
                       perturb: bool = False, perturb_params: Dict[str, Any] = None):
        """
        Build the test set by generating queries and obtaining selected data points.

        :param num_queries: Number of queries to generate.
        :param query_size: Number of data points in each query.
        :param method: Query generation method ('random_sampling' or 'cluster_based').
        :param perturb: Whether to add perturbations to the queries.
        :param perturb_params: Parameters for perturbation.
        """
        for i in range(num_queries):
            # Generate query
            query = self.generate_query(method=method, query_size=query_size, perturb=perturb,
                                        perturb_params=perturb_params)

            # Apply selection
            selected = self.selection_func(query)

            # Append to test set
            self.test_set.append({
                'query_id': i,
                'query': query.reset_index(drop=True),
                'selected': selected.reset_index(drop=True)
            })

            if (i + 1) % 10 == 0:
                print(f"Generated and selected {i + 1}/{num_queries} queries.")

    def save_test_set(self, file_path: str):
        """
        Save the test set to a file using pickle.

        :param file_path: Path to save the test set.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.test_set, f)
        print(f"Test set saved to {file_path}.")

    def load_test_set(self, file_path: str):
        """
        Load the test set from a file.

        :param file_path: Path to load the test set from.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'rb') as f:
            self.test_set = pickle.load(f)
        print(f"Test set loaded from {file_path}.")

    def get_test_set(self) -> List[Dict[str, Any]]:
        """
        Get the current test set.

        :return: List of test set entries.
        """
        return self.test_set

    def clear_test_set(self):
        """
        Clear the current test set.
        """
        self.test_set = []
        print("Test set cleared.")


class ReconstructionExperiment:
    def __init__(self, test_set: List[Dict[str, Any]]):
        """
        Initialize the ReconstructionExperiment.

        :param test_set: List of dictionaries containing 'query' and 'selected' DataFrames.
        """
        self.test_set = test_set

    def reconstruct_query(self, selected: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct the query from the selected data points.
        Implement your reconstruction logic here.

        :param selected: DataFrame of selected data points.
        :return: DataFrame representing the reconstructed query.
        """
        # Placeholder reconstruction logic:
        # For example, use the centroid of selected points as the reconstructed query
        reconstructed_centroid = selected.mean().values.reshape(1, -1)
        reconstructed_query = pd.DataFrame(reconstructed_centroid, columns=selected.columns)
        return reconstructed_query

    def evaluate_reconstruction(self, original_query: pd.DataFrame, reconstructed_query: pd.DataFrame) -> Dict[
        str, float]:
        """
        Evaluate the reconstruction quality.

        :param original_query: DataFrame of the original query.
        :param reconstructed_query: DataFrame of the reconstructed query.
        :return: Dictionary of evaluation metrics.
        """
        # Example metrics: Cosine Similarity and L2 Distance
        from sklearn.metrics.pairwise import cosine_similarity

        # Compute centroids
        original_centroid = original_query.mean().values.reshape(1, -1)
        reconstructed_centroid = reconstructed_query.mean().values.reshape(1, -1)

        # Cosine Similarity
        cos_sim = cosine_similarity(original_centroid, reconstructed_centroid)[0][0]

        # L2 Distance
        l2_dist = np.linalg.norm(original_centroid - reconstructed_centroid)

        return {
            'cosine_similarity': cos_sim,
            'l2_distance': l2_dist
        }

    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run the reconstruction experiment on the entire test set.

        :return: List of evaluation results for each query.
        """
        results = []
        for entry in self.test_set:
            query_id = entry['query_id']
            original_query = entry['query']
            selected = entry['selected']
            reconstructed_query = self.reconstruct_query(selected)
            evaluation = self.evaluate_reconstruction(original_query, reconstructed_query)
            results.append({
                'query_id': query_id,
                'evaluation': evaluation
            })
        return results

    def save_results(self, results: List[Dict[str, Any]], file_path: str):
        """
        Save the experiment results to a file.

        :param results: List of evaluation results.
        :param file_path: Path to save the results.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Reconstruction results saved to {file_path}.")

    def load_results(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load the experiment results from a file.

        :param file_path: Path to load the results from.
        :return: List of evaluation results.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Reconstruction results loaded from {file_path}.")
        return results


class ReconstructionExperimentExtended(ReconstructionExperiment):
    def plot_metrics_distribution(self, results: List[Dict[str, Any]]):
        """
        Plot the distribution of evaluation metrics.

        :param results: List of reconstruction results.
        """
        metrics = pd.DataFrame([res['evaluation'] for res in results])
        plt.figure(figsize=(12, 5))

        # Cosine Similarity Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(metrics['cosine_similarity'], bins=20, kde=True, color='skyblue')
        plt.title('Cosine Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')

        # L2 Distance Distribution
        plt.subplot(1, 2, 2)
        sns.histplot(metrics['l2_distance'], bins=20, kde=True, color='salmon')
        plt.title('L2 Distance Distribution')
        plt.xlabel('L2 Distance')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.close()

    def detailed_evaluation(self, results: List[Dict[str, Any]]):
        """
        Print detailed evaluation metrics.

        :param results: List of reconstruction results.
        """
        metrics = pd.DataFrame([res['evaluation'] for res in results])
        print("=== Reconstruction Evaluation ===")
        print(f"Cosine Similarity - Mean: {metrics['cosine_similarity'].mean():.4f}")
        print(f"Cosine Similarity - Std Dev: {metrics['cosine_similarity'].std():.4f}")
        print(f"L2 Distance - Mean: {metrics['l2_distance'].mean():.4f}")
        print(f"L2 Distance - Std Dev: {metrics['l2_distance'].std():.4f}")
        print("=================================")


# Example usage of the entire pipeline
if __name__ == "__main__":
    # 1. Data Preparation
    # Load seller's dataset
    # Replace 'seller_dataset.csv' with your actual data file
    seller_data_path = 'seller_dataset.csv'
    if not os.path.exists(seller_data_path):
        raise FileNotFoundError(f"The seller data file {seller_data_path} does not exist.")

    seller_data = pd.read_csv(seller_data_path)


    # 2. Define the Selection Function
    def selection(query: pd.DataFrame) -> pd.DataFrame:
        """
        Selection function that takes a query DataFrame and returns selected data points.
        Replace this with your actual selection logic.

        :param query: DataFrame containing query data points.
        :return: DataFrame of selected data points.
        """
        # Placeholder implementation:
        # For example, select data points from seller_data that are most similar to the query
        # Here, we use a simple nearest neighbors approach based on Euclidean distance
        from sklearn.neighbors import NearestNeighbors

        # Combine all query points into a single centroid for simplicity
        query_centroid = query.mean().values.reshape(1, -1)

        # Fit NearestNeighbors on seller_data
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(seller_data.values)
        distances, indices = nbrs.kneighbors(query_centroid)

        # Get the selected data points
        selected = seller_data.iloc[indices[0]]
        return selected


    # 3. Initialize the TestSetBuilder
    test_set_builder = TestSetBuilder(seller_data=seller_data, selection_func=selection)

    # 4. Build the Test Set
    num_queries = 100  # Number of queries to generate
    query_size = 10  # Number of data points in each query
    generation_method = 'random_sampling'  # Currently only 'random_sampling' is implemented
    perturb = True
    perturb_params = {'noise_level': 0.01}

    print("Building the test set...")
    test_set_builder.build_test_set(
        num_queries=num_queries,
        query_size=query_size,
        method=generation_method,
        perturb=perturb,
        perturb_params=perturb_params
    )

    # 5. Save the Test Set
    save_path = 'test_set.pkl'
    test_set_builder.save_test_set(save_path)

    # 6. Load the Test Set (for reconstruction experiments)
    # This step is optional if you continue in the same session
    # Otherwise, you can load the test set in a separate script or later in the pipeline
    # Uncomment the following lines if needed
    # test_set_builder.load_test_set(save_path)
    # loaded_test_set = test_set_builder.get_test_set()

    # 7. Initialize the Reconstruction Experiment
    loaded_test_set = test_set_builder.get_test_set()
    reconstruction_experiment_ext = ReconstructionExperimentExtended(test_set=loaded_test_set)

    # 8. Run the Reconstruction Experiment
    print("Running the reconstruction experiment...")
    reconstruction_results = reconstruction_experiment_ext.run_experiment()

    # 9. Save the Reconstruction Results
    results_save_path = 'reconstruction_results.pkl'
    reconstruction_experiment_ext.save_results(reconstruction_results, results_save_path)

    # 10. Analyze the Results
    reconstruction_experiment_ext.detailed_evaluation(reconstruction_results)
    reconstruction_experiment_ext.plot_metrics_distribution(reconstruction_results)
