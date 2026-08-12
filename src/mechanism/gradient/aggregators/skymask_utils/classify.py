import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # <-- Import StandardScaler


def GMM2(mask_list):
    if not mask_list or len(mask_list) < 2:
        print("Warning: GMM requires at least 2 samples. Returning default labels.")
        return [0] * len(mask_list)

    try:
        mask_array = np.array(mask_list).astype('float64')  # <-- Optional: Use float64 for precision

        if mask_array.ndim == 1:
            print("Warning: PCA/GMM input is 1D. Adjusting.")
            return [0] * len(mask_list)

        if mask_array.shape[1] < 2:
            print("Warning: Feature dimension < 2. Skipping PCA.")
            newX = mask_array
            n_components_gmm = min(2, mask_array.shape[0])
        else:
            pca = PCA(n_components=2)
            newX = pca.fit_transform(mask_array)
            n_components_gmm = 2

        scaler = StandardScaler()
        newX = scaler.fit_transform(newX)
        # ------------------------------------------

        n_components_gmm = min(n_components_gmm, newX.shape[0])
        if n_components_gmm < 1:
            print("Warning: Not enough samples for GMM after PCA. Returning defaults.")
            return [0] * len(mask_list)

        gmm = mixture.GaussianMixture(
            n_components=n_components_gmm,
            random_state=0,
            reg_covar=1e-6  # <-- This is the key fix
        ).fit(newX)
        # ----------------------------

        y_pred = gmm.predict(newX)

        if n_components_gmm == 1:
            print("GMM ran with 1 component. Assuming all inliers.")
            return [1] * len(mask_list)

        benign = np.sum(y_pred == 1)
        mali = np.sum(y_pred == 0)

        # Your logic to flip labels so 1 is the majority (benign) cluster
        if mali > benign:
            print("Flipping GMM labels: Assuming larger cluster is benign (1).")
            y_pred = 1 - y_pred  # Flip 0s to 1s and 1s to 0s

        return y_pred.tolist()

    except Exception as e:
        print(f"Error during GMM/PCA: {e}")
        # Print shape of the *array* that failed, not the list
        if 'mask_array' in locals():
            print(f"Input mask_array shape: {mask_array.shape}")
        return [0] * len(mask_list)
