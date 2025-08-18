import numpy as np
import torch
import open3d as o3d
from sonata.model import load as load_model
from sonata.transform import default as default_transform
from sonata.data import load as load_data

def run_sonata_inference(point_cloud_path):
    """
    Runs the Sonata inference pipeline on a given point cloud file.

    Args:
        point_cloud_path (str): Path to the input point cloud file.

    Returns:
        np.ndarray: The extracted features.
    """
    # 1. Load the model
    try:
        model = load_model("sonata").cuda()
        model.eval()
    except Exception as e:
        print(f"Failed to load the model: {e}")
        print("Please ensure that the Pointcept library is installed correctly and that you have a working CUDA environment.")
        return None

    # 2. Get the default transformation pipeline
    transform = default_transform()

    # 3. Load the point cloud data
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    point = {
        "coord": np.asarray(pcd.points, dtype=np.float32),
        "color": np.asarray(pcd.colors, dtype=np.float32) * 255, # Pointcept expects color in [0, 255]
        "normal": np.asarray(pcd.normals, dtype=np.float32),
    }

    # 4. Apply the transformations
    # The transform pipeline will return a dictionary of tensors.
    point = transform(point)

    # 5. Move data to GPU and run inference
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].cuda()
    
    with torch.no_grad():
        features = model(point)

    # 6. Post-process features
    # The output of the model is a Point object with a 'feat' attribute.
    # We return the features as a numpy array.
    return features.feat.cpu().numpy()

if __name__ == '__main__':
    try:
        features = run_sonata_inference("sample_point_cloud.ply")
        if features is not None:
            print("Successfully ran Sonata inference.")
            print("Output feature shape:", features.shape)
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        print("Please ensure all dependencies are installed and the environment is set up correctly.")
        print("If you are still facing issues, please check the GEMINI.md file for troubleshooting.")
