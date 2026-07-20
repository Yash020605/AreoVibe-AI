import cv2
import numpy as np

def compute_true_heatmap(image_path_or_bytes):
    """
    Takes an RGB image and computes a True Vegetation Heatmap using VARI
    (Visible Atmospherically Resistant Index).
    Formula: (G - R) / (G + R - B + epsilon)
    """
    if isinstance(image_path_or_bytes, str):
        img = cv2.imread(image_path_or_bytes)
    else:
        # Assuming bytes from Streamlit
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image could not be loaded into OpenCV.")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract channels (as floats for division)
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)
    
    # Calculate VARI = (G - R) / (G + R - B)
    denominator = (G + R - B)
    # Avoid zero division
    denominator[denominator == 0] = 1e-5
    
    vari = (G - R) / denominator
    
    # VARI values typically range from -1 to 1. Normalize to [0, 255]
    # We clip it to remove extreme outliers caused by denominator being very small
    vari_clipped = np.clip(vari, -1.0, 1.0)
    vari_norm = ((vari_clipped + 1.0) / 2.0) * 255.0
    vari_norm = np.clip(vari_norm, 0, 255).astype(np.uint8)
    
    # Apply COLORMAP_TURBO for a highly sophisticated, perceptual heatmap 
    heatmap = cv2.applyColorMap(vari_norm, cv2.COLORMAP_TURBO)
    
    # Convert heatmap back to RGB for Streamlit rendering
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return img_rgb, heatmap_rgb
