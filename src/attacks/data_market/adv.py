import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.attacks.data_market.general_attack.my_utils import image_modification, embed_image, \
    load_model_and_preprocessor


class Adv:
    def __init__(self, x_s, y_s, costs, indices, emb_model, device, img_paths):

        # Reference the original dataset
        self.original_x = x_s.copy()
        self.original_y = y_s.copy()

        # Indices for this subset
        self.indices = indices

        self.x = self.original_x[self.indices]
        self.y = self.original_y[self.indices]
        if costs is not None:
            self.original_costs = costs.copy()
            self.costs = self.original_costs[self.indices]
        self.img_path = img_paths

        self.emb_model, self.preprocess, self.emb_inference_func = load_model_and_preprocessor(emb_model, device)
        self.device = device
        # Initialize CLIP model and tokenizer
        self.emb_model.eval()

        # Initialize Fisher Information Matrix
        # self.fisher_information = self.build_fisher_information_matrix(self.embeddings)

    def update_subset(self, new_values, attribute="x"):
        """
        Update values in the subset and reflect them in the original dataset.

        Parameters:
        - new_values: New values to update the subset with.
        - attribute: Which attribute ('x', 'y', 'costs') to update.
        """
        if attribute == "x":
            self.x[:] = new_values
            self.original_x[self.indices] = new_values
        elif attribute == "y":
            self.y[:] = new_values
            self.original_y[self.indices] = new_values
        elif attribute == "costs":
            self.costs[:] = new_values
            self.original_costs[self.indices] = new_values
        else:
            raise ValueError("Invalid attribute to update")

    def reset(self):
        """
        Reset the subset to its original state based on the original dataset.
        """
        self.x = self.original_x[self.indices]
        self.y = self.original_y[self.indices]
        self.costs = self.original_costs[self.indices]

    def embed_image(self, img, normalize_embeddings=True):
        return embed_image(img, self.emb_model, self.preprocess, self.emb_inference_func, device=self.device,
                           normalize_embeddings=normalize_embeddings)

    def set_new_purchase(self):
        pass

    def attack(self, attack_type, attack_param, x_s, costs, img_path):
        """
        Perform specified attack type: cost manipulation, data manipulation, or both.

        Parameters:
        - attack_type (str): Type of attack to perform ("cost", "data", or "both").
        - attack_param (dict): Parameters for the attack (e.g., manipulation method, factors).
        - target_images (list, optional): List of target images to mimic for data manipulation.
        - target_costs (np.ndarray, optional): Array of target costs to mimic or undercut in cost manipulation.

        Returns:
        - dict: Dictionary containing manipulated results (e.g., manipulated costs, manipulated images).
        """
        res = {}
        match attack_type:
            case "cost_manipulation":
                print("Applying cost_manipulation Attack...")
                cost = self.cost_manipulation(attack_param, x_s, costs, img_path)
            case "data_manipulation":
                target_indices = attack_param["selected_indices"]
                modify_indices = attack_param["unselected_indices"]
                poison_rate = attack_param["poison_rate"]
                output_dir = attack_param["output_dir"]

                return self.data_manipulation(target_indices, modify_indices, poison_rate, attack_steps=100,
                                              attack_lr=0.1,
                                              attack_reg=0.1, epsilon=0.05,
                                              output_dir=output_dir)
            case "embedding_reconstruction_math":
                print("Applying embedding_reconstruction Attack...")
            case "embedding_reconstruction_opt":
                print("Applying embedding_reconstruction Attack...")
            case "embedding_space_estimation":
                print("Applying embedding_space_estimation Attack...")
            case "embedding_property_inference":
                print("Applying embedding_property_inference Attack...")

            # {"target_vector": target_vector,
            #  "index": random_selected_index,
            #  "target_img_path": img_path[random_selected_index],
            #  "original_img_path": img_path[idx]}
            # Initialize storage for manipulated image mappings
            # Perform mimic attack by mapping each original image to a target image

        return res

    def cost_manipulation(self, attack_param, x_s, costs, img_path):
        # Step 1: Apply Cost Manipulation
        target_indices = attack_param["selected_indices"]
        modify_indices = attack_param["unselected_indices"]
        num_samples = int(len(modify_indices) * self.poison_rate)
        # Perform random selection without replacement
        modify_indices = np.random.choice(modify_indices, size=num_samples, replace=False)

        preprocess = attack_param["img_preprocess"]
        global_selected_indices = attack_param["global_selected_indices"]
        device = attack_param["device"]
        model = attack_param["emb_model"]
        modified_images_path = attack_param["modified_images_path"]
        modified_images_path = f"{modified_images_path}/"
        manipulated_costs = None
        manipulated_info = None
        if len(target_indices) == 0:
            target_indices = global_selected_indices
        print("Applying Cost Manipulation Attack...")
        target_costs = costs[target_indices]

        # Retrieve base costs for manipulation
        manipulated_costs = self.costs.copy()

        # Different types of cost manipulations
        if attack_param["cost_manipulation_method"] == "scale":
            manipulated_costs *= attack_param.get("cost_factor", 1.0)

        elif attack_param["cost_manipulation_method"] == "offset":
            manipulated_costs += attack_param.get("cost_offset", 0.0)

        elif attack_param["cost_manipulation_method"] == "power":
            manipulated_costs = np.power(manipulated_costs, attack_param.get("cost_power", 1.0))

        elif attack_param["cost_manipulation_method"] == "match_target":
            # Set costs to match target costs exactly if provided
            if target_costs is not None:
                manipulated_costs = target_costs

        elif attack_param["cost_manipulation_method"] == "undercut_target":
            # Make costs slightly lower than target costs
            if target_costs is not None:
                manipulated_costs = np.minimum(manipulated_costs,
                                               target_costs - attack_param.get("undercut_margin", 0.01))

        elif attack_param["cost_manipulation_method"] == "random_reduce":
            # Randomly reduce costs within a specified range
            reduction_factor = np.random.uniform(
                attack_param.get("min_reduction", 0.8),
                attack_param.get("max_reduction", 1.0),
                size=manipulated_costs.shape
            )
            manipulated_costs *= reduction_factor

        print("Cost Manipulation Completed.")

    def assign_random_targets(self, selected_indices, unsampled_indices, img_path):
        """
        Assign a random target vector (from selected samples) to each unselected sample.

        Parameters:
        - x_s (np.array): The array of embeddings for all samples.
        - selected_indices (list or np.array): Indices of selected samples to choose from.
        - unsampled_indices (list or np.array): Indices of unsampled data points to assign targets.

        Returns:
        - target_vectors (dict): A dictionary where keys are unsampled indices and values are target vectors.
        """
        target_vectors = {}
        for idx in unsampled_indices:
            random_selected_index = np.random.choice(selected_indices)  # Choose a random selected sample
            target_vector = self.original_x[random_selected_index]
            # target_vector = target_vector / np.linalg.norm(target_vector)  # Normalize the target vector
            target_vectors[idx] = {"target_vector": target_vector,
                                   "target_index": random_selected_index,
                                   "target_img_path": img_path[random_selected_index],
                                   "original_img_path": img_path[idx]}
        return target_vectors

    def data_manipulation(self, target_indices, modify_indices, poison_rate, attack_steps=100, attack_lr=0.1,
                          attack_reg=0.1, epsilon=0.05,
                          output_dir="./"):
        num_samples = int(len(modify_indices) * poison_rate)
        # Perform random selection without replacement
        modify_indices = np.random.choice(modify_indices, size=num_samples, replace=False)

        # create the modify info for each of images

        img_manipulate_dic = self.assign_random_targets(target_indices, modify_indices, self.img_path)
        manipulated_info = image_modification(
            modify_info=img_manipulate_dic,
            model=self.emb_model,
            processor=self.preprocess,
            device=self.device,
            num_steps=attack_steps,
            learning_rate=attack_lr,
            lambda_reg=attack_reg,
            epsilon=epsilon,
            output_dir=output_dir,
        )
        return manipulated_info

    def embedding_reconstruction(self, attack_type, attack_param, x_s, costs, img_path):
        self.fisher_information = self.build_fisher_information_matrix(self.embeddings)

    def embedding_space_estimation(self, attack_type, attack_param, x_s, costs, img_path):
        pass

    def embedding_property_inference(self, attack_type, attack_param, x_s, costs, img_path):
        pass

    def compute_clip_embeddings(self, data_points, batch_size=32):
        """
        Compute CLIP embeddings for the given data points.

        Args:
            data_points (list of str): Data points to embed.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Array of CLIP embeddings.
        """
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(data_points), batch_size):
                batch = data_points[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                normalized = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(normalized.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings  # Shape: (num_data_points, embedding_dim)

    def build_fisher_information_matrix(self, embeddings):
        """
        Build the Fisher Information Matrix based on CLIP embeddings.

        Args:
            embeddings (np.ndarray): CLIP embeddings of seller data.

        Returns:
            np.ndarray: Fisher Information Matrix.
        """
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        fim = np.cov(centered, rowvar=False)
        return fim  # Shape: (embedding_dim, embedding_dim)

    def frank_wolfe_optimization(self, fim, num_iterations=100):
        """
        Perform Frank-Wolfe optimization to select data points.

        Args:
            fim (np.ndarray): Fisher Information Matrix.
            num_iterations (int): Number of iterations.

        Returns:
            list of int: Indices of selected data points.
        """
        num_data, dim = self.embeddings.shape
        selected = set()
        gradient = np.zeros(dim)

        for it in range(num_iterations):
            # Compute gradient (for demonstration, using random gradient)
            gradient = np.random.randn(dim)

            # Linear optimization oracle
            scores = self.embeddings @ gradient
            selected_index = np.argmax(scores)
            selected.add(selected_index)

            if len(selected) >= self.budget:
                break

        return list(selected)

    def v_optimization(self, fim, selected_indices):
        """
        Perform v-optimization to estimate parameters based on selected data.

        Args:
            fim (np.ndarray): Fisher Information Matrix.
            selected_indices (list of int): Indices of selected data points.

        Returns:
            np.ndarray: Optimized v vector.
        """
        selected_embeddings = self.embeddings[selected_indices]
        mean_selected = np.mean(selected_embeddings, axis=0)

        def objective(v):
            return -v @ mean_selected  # Maximizing the dot product

        def constraint(v):
            return np.sum(v) - 1  # Sum of v should be 1

        cons = {'type': 'eq', 'fun': constraint}
        bounds = [(0, 1) for _ in range(len(mean_selected))]

        initial_v = np.ones(len(mean_selected)) / len(mean_selected)

        result = minimize(objective, initial_v, method='SLSQP', bounds=bounds, constraints=cons)

        if result.success:
            return result.x
        else:
            raise ValueError("v-optimization failed.")

    def infer_buyer_preferences(self, selected_indices):
        """
        Infer buyer's preferences based on selected data points.

        Args:
            selected_indices (list of int): Indices of selected data points.

        Returns:
            dict: Inferred information about the buyer.
        """
        selected_embeddings = self.embeddings[selected_indices]
        mean_embedding = np.mean(selected_embeddings, axis=0)

        prototypes = self.get_prototypes()

        similarities = {proto_name: np.dot(mean_embedding, proto_vec) for proto_name, proto_vec in prototypes.items()}
        inferred_preferences = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        return {"similarities": inferred_preferences}

    def get_prototypes(self):
        """
        Define some prototype vectors for inference.

        Returns:
            dict: Prototype name to vector mapping.
        """
        # In a real scenario, these would be predefined or learned
        return {
            "Category_A": np.random.randn(self.embeddings.shape[1]),
            "Category_B": np.random.randn(self.embeddings.shape[1]),
            "Category_C": np.random.randn(self.embeddings.shape[1]),
        }

    def sample_unselected_data(self, selected_indices):
        """
        Sample a subset of unselected data points to use as negative samples.

        Args:
            selected_indices (list of int): Indices of selected data points.

        Returns:
            list of int: Indices of sampled unselected data points.
        """
        all_indices = set(range(len(self.embeddings)))
        unselected_indices = list(all_indices - set(selected_indices))
        if len(unselected_indices) < self.unselected_sample_size:
            sampled_unselected = unselected_indices
        else:
            sampled_unselected = random.sample(unselected_indices, self.unselected_sample_size)
        return sampled_unselected

    def train_selection_classifier(self, selected_indices, sampled_unselected_indices):
        """
        Train a binary classifier to distinguish between selected and unselected data points.

        Args:
            selected_indices (list of int): Indices of selected data points.
            sampled_unselected_indices (list of int): Indices of sampled unselected data points.

        Returns:
            sklearn classifier: Trained classifier.
            dict: Evaluation metrics.
        """
        # Prepare labels
        labels = np.zeros(len(self.embeddings))
        labels[selected_indices] = 1
        labels[sampled_unselected_indices] = 0

        # Combine selected and unselected embeddings
        combined_indices = selected_indices + sampled_unselected_indices
        X = self.embeddings[combined_indices]
        y = labels[combined_indices]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train logistic regression classifier
        clf = LogisticRegression(random_state=0, max_iter=1000)
        clf.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }

        return clf, metrics

    def analyze_feature_importance(self, classifier):
        """
        Analyze feature importance from the trained classifier.

        Args:
            classifier (sklearn classifier): Trained classifier.

        Returns:
            np.ndarray: Feature importance scores.
        """
        if hasattr(classifier, 'coef_'):
            # For linear models like Logistic Regression
            importance = np.abs(classifier.coef_[0])
        elif hasattr(classifier, 'feature_importances_'):
            # For tree-based models
            importance = classifier.feature_importances_
        else:
            raise ValueError("Classifier does not have coef_ or feature_importances_ attribute.")

        # Normalize importance
        importance /= importance.sum()
        return importance

    def build_weighted_fisher_information_matrix(self, selected_indices, sampled_unselected_indices):
        """
        Build a weighted Fisher Information Matrix, giving higher weights to selected samples.

        Args:
            selected_indices (list of int): Indices of selected data points.
            sampled_unselected_indices (list of int): Indices of sampled unselected data points.

        Returns:
            np.ndarray: Weighted Fisher Information Matrix.
        """
        # Assign weights
        weights = np.ones(len(self.embeddings)) * self.w_unselected
        weights[selected_indices] = self.w_selected

        # Select data points to include in FIM
        combined_indices = selected_indices + sampled_unselected_indices
        selected_weights = weights[combined_indices]
        X = self.embeddings[combined_indices]

        # Normalize weights
        selected_weights = selected_weights / selected_weights.sum()

        # Compute weighted mean
        mean = np.average(X, axis=0, weights=selected_weights)

        # Compute weighted covariance matrix
        X_centered = X - mean
        weighted_cov = np.cov(X_centered.T, aweights=selected_weights)

        return weighted_cov  # Shape: (embedding_dim, embedding_dim)

    def explore_fisher_information_matrix(self, weighted_fim):
        """
        Explore the properties of the weighted Fisher Information Matrix.

        Args:
            weighted_fim (np.ndarray): Weighted Fisher Information Matrix.

        Returns:
            dict: Eigenvalues and eigenvectors of the FIM.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_fim)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }

    def visualize_eigenvalues(self, eigenvalues):
        """
        Plot the eigenvalues of the Fisher Information Matrix.

        Args:
            eigenvalues (np.ndarray): Eigenvalues of the FIM.

        Returns:
            None: Displays a plot.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=np.arange(1, len(eigenvalues) + 1), y=eigenvalues, palette='viridis')
        plt.title("Eigenvalues of the Weighted Fisher Information Matrix")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue Magnitude")
        plt.close()

    def visualize_principal_components(self, eigenvectors, X, num_components=2):
        """
        Visualize the data in the space of the top principal components.

        Args:
            eigenvectors (np.ndarray): Eigenvectors of the FIM.
            X (np.ndarray): Data points to visualize.
            num_components (int): Number of principal components to plot.

        Returns:
            None: Displays a scatter plot.
        """
        pcs = eigenvectors[:, :num_components]
        X_pca = X @ pcs

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
        plt.title(f"Data Projected onto Top {num_components} Principal Components")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.close()

    def fim_exploration_classifier_training(self):
        """
        Execute the attack and explore the Fisher Information Matrix.

        Returns:
            dict: Results including selected indices, classifier metrics, feature importance, and FIM analysis.
        """
        selected_indices = self.frank_wolfe_optimization(self.fisher_information)

        sampled_unselected = self.sample_unselected_data(selected_indices)

        classifier, metrics = self.train_selection_classifier(selected_indices, sampled_unselected)

        feature_importance = self.analyze_feature_importance(classifier)

        weighted_fim = self.build_weighted_fisher_information_matrix(selected_indices, sampled_unselected)

        fim_analysis = self.explore_fisher_information_matrix(weighted_fim)

        self.visualize_eigenvalues(fim_analysis["eigenvalues"])

        combined_indices = selected_indices + sampled_unselected
        X_visualize = self.embeddings[combined_indices]
        self.visualize_principal_components(fim_analysis["eigenvectors"], X_visualize, num_components=2)

        return {
            "selected_indices": selected_indices,
            "sampled_unselected_indices": sampled_unselected,
            "classifier_metrics": metrics,
            "feature_importance": feature_importance,
            "weighted_fisher_information_matrix": weighted_fim,
            "fim_eigenvalues": fim_analysis["eigenvalues"],
            "fim_eigenvectors": fim_analysis["eigenvectors"]
        }


if __name__ == "__main__":
    # Example seller data (e.g., text descriptions)
    seller_data = [
        "A photo of a cat sitting on a sofa.",
        "An image of a bustling city street at night.",
        "A diagram of a complex machine.",
        "A portrait of a smiling person.",
        "A scenic view of mountains during sunset.",
        "A close-up of a colorful butterfly on a flower.",
        "An aerial shot of a dense forest.",
        "A picture of a vintage car parked in front of a house.",
        "A snapshot of a busy marketplace with various stalls.",
        "An artwork depicting abstract shapes and vibrant colors.",
        # Add more data points as needed
    ]

    budget = 3  # Number of data points the buyer can select

    # Initialize the attacker with higher weight for selected samples
    attacker = Adv(
        seller_data=seller_data,
        budget=budget,
        device='cpu',
        unselected_sample_size=5,
        w_selected=3.0,  # Higher weight for selected samples
        w_unselected=1.0  # Lower weight for unselected samples
    )

    # Execute the attack and explore the weighted Fisher Information Matrix
    attack_results = attacker.fim_exploration_classifier_training()

    print("Selected Indices by Buyer:", attack_results["selected_indices"])
    print("Sampled Unselected Indices:", attack_results["sampled_unselected_indices"])
    print("Classifier Evaluation Metrics:", attack_results["classifier_metrics"])
    print("Feature Importance (Normalized):", attack_results["feature_importance"])
    print("Weighted Fisher Information Matrix:\n", attack_results["weighted_fisher_information_matrix"])
    print("FIM Eigenvalues:", attack_results["fim_eigenvalues"])
    # Further analysis can be done as needed
