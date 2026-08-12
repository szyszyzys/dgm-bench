import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import numpy as np

# Device configuration: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_image_from_url(url, transform):
    """
    Load an image from a URL and apply the given transformations.
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return transform(image)


def preprocess_image(image):
    """
    Preprocess the image for CLIP.
    """
    return processor(images=image, return_tensors="pt").to(device)


def extract_features(image_tensor):
    """
    Extract CLIP image features.
    """
    with torch.no_grad():
        image_features = model.get_image_features(**image_tensor)
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features


def show_images(original, modified, title1='Original Image', title2='Modified Image'):
    """
    Display the original and modified images side by side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(original)
    axs[0].set_title(title1)
    axs[0].axis('off')

    axs[1].imshow(modified)
    axs[1].set_title(title2)
    axs[1].axis('off')

    plt.close()


def optimize_image_to_target(
        original_image,
        target_embedding,
        model,
        processor,
        num_iterations=300,
        learning_rate=0.05,
        lambda_reg=1e-2,
        epsilon=0.1,
):
    """
    Optimize the image so that its CLIP embedding is close to the target embedding.

    Parameters:
    - original_image (PIL.Image): The image to be optimized.
    - target_embedding (torch.Tensor): The desired CLIP embedding (normalized).
    - model (CLIPModel): The pre-trained CLIP model.
    - processor (CLIPProcessor): The CLIP processor.
    - num_iterations (int): Number of optimization steps.
    - learning_rate (float): Learning rate for the optimizer.
    - lambda_reg (float): Regularization strength to limit image changes.
    - epsilon (float): Maximum allowed perturbation per pixel.

    Returns:
    - modified_image (PIL.Image): The optimized image.
    - history (dict): Dictionary containing loss history.
    """
    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    postprocess = transforms.ToPILImage()

    # Preprocess the image and make it a parameter
    image_tensor = preprocess(original_image).unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)

    # Initialize history
    history = {'total_loss': [], 'cosine_loss': [], 'reg_loss': []}

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass: compute CLIP embedding
        inputs = processor(images=postprocess(image_tensor.squeeze(0).detach().cpu()), return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)

        # Cosine similarity loss (we want to maximize similarity)
        cosine_sim = F.cosine_similarity(image_features, target_embedding)
        cosine_loss = 1 - cosine_sim  # Minimizing this will maximize similarity

        # Regularization loss to keep changes small
        reg_loss = torch.norm(image_tensor.grad) if image_tensor.grad is not None else torch.tensor(0.0).to(device)
        reg_loss = lambda_reg * torch.norm(image_tensor - preprocess(original_image).unsqueeze(0).to(device))

        # Total loss
        total_loss = cosine_loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Clamp the image tensor to ensure pixel values are valid and within perturbation limits
        with torch.no_grad():
            perturbation = torch.clamp(image_tensor - preprocess(original_image).unsqueeze(0).to(device), -epsilon,
                                       epsilon)
            image_tensor.copy_(torch.clamp(preprocess(original_image).unsqueeze(0).to(device) + perturbation, 0, 1))

        # Record history
        history['total_loss'].append(total_loss.item())
        history['cosine_loss'].append(cosine_loss.item())
        history['reg_loss'].append(reg_loss.item())

        # Print progress every 50 iterations
        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(
                f"Iteration {iteration + 1}/{num_iterations}, Total Loss: {total_loss.item():.4f}, Cosine Loss: {cosine_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")

    # Detach and convert to PIL Image
    modified_image = image_tensor.detach().cpu().squeeze(0)
    modified_image = postprocess(modified_image)

    return modified_image, history


def run_experiment(original_img, target_image):
    # Define URLs for the original and target images

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load images
    print("Loading images...")
    original_image = transforms(original_img)
    target_image = transforms(target_image)

    # Display original and target images
    show_images(original_image, target_image, "Original Image", "Target Image")

    # Extract features
    print("Extracting features...")
    original_features = extract_features(preprocess(original_image).unsqueeze(0), model, processor)
    target_features = extract_features(preprocess(target_image).unsqueeze(0), model, processor)

    # Normalize target features
    target_features = F.normalize(target_features, p=2, dim=-1)

    # Optimize the original image to match the target embedding
    print("Optimizing the original image to match the target embedding...")
    modified_image, history = optimize_image_to_target(
        original_image=original_image,
        target_embedding=target_features,
        model=model,
        processor=processor,
        num_iterations=300,
        learning_rate=0.05,
        lambda_reg=1e-2,
        epsilon=0.1,
    )

    # Extract features of the modified image
    print("Extracting features of the modified image...")
    modified_features = extract_features(preprocess(modified_image).unsqueeze(0), model, processor)

    # Compute cosine similarities
    original_similarity = F.cosine_similarity(original_features, target_features).item()
    modified_similarity = F.cosine_similarity(modified_features, target_features).item()

    print(f"Cosine Similarity - Original Image vs. Target: {original_similarity:.4f}")
    print(f"Cosine Similarity - Modified Image vs. Target: {modified_similarity:.4f}")

    # Display original, target, and modified images
    show_images(original_image, modified_image, "Original Image", "Modified Image")

    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['cosine_loss'], label='Cosine Loss')
    plt.plot(history['reg_loss'], label='Regularization Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Loss History')
    plt.legend()
    plt.close()

    # Calculate the amount of change (L2 distance)
    original_tensor = preprocess(original_image).unsqueeze(0).to(device)
    modified_tensor = preprocess(modified_image).unsqueeze(0).to(device)
    l2_distance = torch.norm(original_tensor - modified_tensor).item()
    print(f"L2 Distance Between Original and Modified Image: {l2_distance:.4f}")

    # Save the modified image
    modified_image.save("modified_image.jpg")
    print("Modified image saved as 'modified_image.jpg'.")


# Run the experiment
if __name__ == "__main__":
    run_experiment()
