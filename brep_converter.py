import numpy as np
import open3d as o3d

def xyz_to_ply(input_path, output_path):
    """
    Converts a .xyz file to a .ply file.

    Args:
        input_path (str): Path to the input .xyz file.
        output_path (str): Path to the output .ply file.
    """
    try:
        pcd = o3d.io.read_point_cloud(input_path, format='xyz')
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to {output_path}: {e}")

def ply_to_xyz(input_path, output_path):
    """
    Converts a .ply file to a .xyz file.

    Args:
        input_path (str): Path to the input .ply file.
        output_path (str): Path to the output .xyz file.
    """
    try:
        pcd = o3d.io.read_point_cloud(input_path)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to {output_path}: {e}")
