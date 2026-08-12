def extract_features(image_tensor, model, processor):
    """
    Extract CLIP image features.
    """
    inputs = processor(images=image_tensor, return_tensors="pt").to(model.device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to NumPy
