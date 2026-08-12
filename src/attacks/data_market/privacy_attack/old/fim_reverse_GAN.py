import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def build_generator(latent_dim, n_features):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_features, activation='linear'))
    return model


def build_discriminator(n_features):
    model = Sequential()
    model.add(Dense(128, input_dim=n_features))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_gan(X_selected, X_unselected, latent_dim=100, epochs=10000, batch_size=64, save_interval=1000):
    """
    Trains a Conditional GAN to generate x_test.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - latent_dim: Dimension of the latent space
    - epochs: Number of training iterations
    - batch_size: Size of each training batch
    - save_interval: Interval to save and display generated samples

    Returns:
    - generator: Trained generator model
    - discriminator: Trained discriminator model
    """
    n_features = X_selected.shape[1]
    # Labels: 1 for selected, 0 for unselected
    y_selected = np.ones((X_selected.shape[0], 1))
    y_unselected = np.zeros((X_unselected.shape[0], 1))

    # Combine data
    X = np.vstack((X_selected, X_unselected))
    y = np.vstack((y_selected, y_unselected))

    # Build and compile the discriminator
    discriminator = build_discriminator(n_features)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    # Build the generator
    generator = build_generator(latent_dim, n_features)

    # Build the GAN by combining generator and discriminator
    z = Input(shape=(latent_dim,))
    generated_x = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_x)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Training Loop
    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of real samples
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_samples = X[idx]
        real_labels = y[idx]

        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))  # Generator tries to fool discriminator

        # Train the generator via the combined model
        g_loss = combined.train_on_batch(noise, valid_y)

        # Print the progress
        if epoch % 1000 == 0 and verbose:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            sample_images(generator, latent_dim, n_features=10, epoch=epoch)

    return generator, discriminator


def sample_images(generator, latent_dim, n_features, epoch):
    """
    Generates and plots sample images from the generator.
    """
    noise = np.random.normal(0, 1, (5, latent_dim))
    gen_samples = generator.predict(noise)
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.stem(gen_samples[i], use_line_collection=True)
        plt.title(f'Epoch {epoch}')
        plt.tight_layout()
    plt.close()


def reconstruct_x_test_gan(generator, x_test, latent_dim=100):
    """
    Attempts to reconstruct x_test using the trained GAN.

    Parameters:
    - generator: Trained generator model
    - x_test: True x_test vector
    - latent_dim: Dimension of the latent space

    Returns:
    - closest_sample: Generated sample closest to x_test
    - mse: Mean Squared Error
    - cosine_sim: Cosine Similarity
    """
    num_samples = 1000
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    gen_samples = generator.predict(noise)

    # Compute distances to x_test
    mse = np.mean((gen_samples - x_test) ** 2, axis=1)
    closest_idx = np.argmin(mse)
    closest_sample = gen_samples[closest_idx]

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity([x_test], [closest_sample])[0, 0]

    # Plot the closest sample
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_test))
    plt.stem(x_test, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.stem(closest_sample, linefmt='r--', markerfmt='rx', basefmt=' ')
    plt.title('GAN-Based Reconstruction: True vs. Closest Generated Sample')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend(['True x_test', 'Closest Generated Sample'])
    plt.grid(True)
    plt.close()

    print(f"GAN-Based Reconstruction MSE: {mse[closest_idx]:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")

    return closest_sample, mse[closest_idx], cosine_sim


# Example Usage
if __name__ == "__main__":
    # Note: Training GANs can be computationally intensive and may require GPU acceleration for larger datasets.
    # For demonstration, we'll use smaller numbers.
    n_features = 10
    n_selected = 100
    n_unselected = 200
    latent_dim = 50
    epochs = 5000
    batch_size = 64
    gamma = 0.1
    n_components = 5
    lambda_reg = 1e-5
    top_k = 2
    n_clusters = 3
    num_shadow_models = 5
    verbose = True

    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Train GAN
    generator, discriminator = train_gan(X_selected, X_unselected, latent_dim=latent_dim, epochs=epochs,
                                         batch_size=batch_size, save_interval=1000)

    # Reconstruct x_test using GAN
    closest_sample, mse_gan, cosine_sim_gan = reconstruct_x_test_gan(generator, x_test, latent_dim=latent_dim)

    # Summary
    print("\n--- GAN-Based Inference Summary ---")
    print(f"Reconstructed x_test: {closest_sample}")
    print(f"MSE: {mse_gan:.6f}")
    print(f"Cosine Similarity: {cosine_sim_gan:.6f}")
