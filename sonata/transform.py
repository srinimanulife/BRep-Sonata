from pointcept.datasets.transform import Compose as PointceptCompose
from pointcept.datasets.transform import TRANSFORMS

def default():
    """
    Returns the default inference augmentation pipeline using Pointcept transforms.
    """
    # The configuration for the transformations is defined as a list of dictionaries.
    # Each dictionary specifies the type of transformation and its parameters.
    transform_config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",  # Even for inference, 'train' mode is used to get a single point cloud
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    
    # We can instantiate the Compose class with the configuration.
    # The Compose class will build the transformations from the registry.
    return PointceptCompose(transform_config)
