import open3d as o3d
import numpy as np

def load(name):
    """
    Loads a sample point cloud.
    In the future, this could be extended to load other samples.
    """
    if name == "sample1":
        # The README mentions a "sample1". We will use our existing sample file.
        pcd_path = "sample_point_cloud.ply"
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        point = {
            "coord": np.asarray(pcd.points),
            "color": np.asarray(pcd.colors),
            "normal": np.asarray(pcd.normals),
        }
        return point
    else:
        raise ValueError(f"Unknown sample name: {name}")
