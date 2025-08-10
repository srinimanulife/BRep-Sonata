
import open3d as o3d
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

class BRepVisualizer:
    """
    A class to visualize BRep models and point clouds with interactive controls.
    """
    def __init__(self, pcd_path, brep_path, transparency=0.5, output_path="render.png"):
        """
        Initializes the visualizer with paths to point cloud and BRep files.
        """
        print("Loading data...")
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        if not self.pcd.has_points():
            print(f"Error: Failed to load point cloud from {pcd_path}")
            sys.exit(1)

        self.brep = o3d.io.read_triangle_mesh(brep_path)
        if self.brep.is_empty():
            print(f"Error: Failed to load BRep mesh from {brep_path}")
            sys.exit(1)
        
        self.brep_original = o3d.io.read_triangle_mesh(brep_path) # Keep an unmodified copy for slicing

        self.transparency = transparency
        self.output_path = output_path
        self.vis_mode = 0  # 0: Both, 1: PCD only, 2: BRep only
        self.slicing_axis = None  # 'x', 'y', or 'z'
        self.slicing_pos = 0.0
        self.view_changed = True

        print("Coloring BRep mesh...")
        self._color_brep_by_height()
        self._update_brep_transparency()

    def _color_brep_by_height(self):
        """Assigns vertex colors to the BRep mesh based on Z-axis height."""
        vertices = np.asarray(self.brep.vertices)
        if vertices.shape[0] == 0:
            return
        z_vals = vertices[:, 2]
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        
        if z_max - z_min < 1e-6:
            normalized_z = np.zeros(len(z_vals))
        else:
            normalized_z = (z_vals - z_min) / (z_max - z_min)
            
        cmap = plt.get_cmap("viridis")
        colors = cmap(normalized_z)[:, :3]
        self.brep.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.brep_original.vertex_colors = self.brep.vertex_colors


    def _update_brep_transparency(self):
        """Updates the BRep material to reflect the current transparency."""
        material = self.brep.get_triangle_material()
        material.base_color = [1.0, 1.0, 1.0, 1.0 - self.transparency]
        material.shader = "defaultLitTransparency"
        self.brep.triangle_material = material


    def _toggle_visibility(self, vis):
        """Cycles through the visibility modes."""
        self.vis_mode = (self.vis_mode + 1) % 3
        modes = {0: "Both", 1: "Point Cloud Only", 2: "BRep Only"}
        print(f"Display Mode: {modes[self.vis_mode]}")
        self.view_changed = True
        return False

    def _adjust_transparency(self, vis, amount):
        """Adjusts the transparency of the BRep model."""
        self.transparency = np.clip(self.transparency + amount, 0.0, 1.0)
        print(f"BRep Transparency: {self.transparency:.2f}")
        self._update_brep_transparency()
        self.view_changed = True
        return False

    def _set_slicing_axis(self, vis, axis):
        """Sets the axis for slicing."""
        if self.slicing_axis == axis:
            self.slicing_axis = None
            print("Slicing disabled.")
            self.brep = o3d.geometry.TriangleMesh(self.brep_original) # Restore original
            self._update_brep_transparency()
        else:
            self.slicing_axis = axis
            # Initialize slicing position to the center of the chosen axis
            center = self.brep_original.get_center()
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            self.slicing_pos = center[axis_map[axis]]
            print(f"Slicing enabled on {axis.upper()}-axis. Use Left/Right arrows to move plane.")
        self._apply_slicing()
        self.view_changed = True
        return False

    def _adjust_slicing(self, vis, amount):
        """Adjusts the position of the slicing plane."""
        if self.slicing_axis is None:
            return False
        
        bounds = self.brep_original.get_axis_aligned_bounding_box()
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        step = (bounds.max_bound[axis_map[self.slicing_axis]] - bounds.min_bound[axis_map[self.slicing_axis]]) * amount
        self.slicing_pos += step
        print(f"Slicing Plane Position: {self.slicing_pos:.2f}")
        self._apply_slicing()
        self.view_changed = True
        return False

    def _apply_slicing(self):
        """Crops the geometry based on the current slicing settings."""
        if self.slicing_axis is None:
            return

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_index = axis_map[self.slicing_axis]
        
        # Create a temporary copy to crop
        cropped_brep = o3d.geometry.TriangleMesh(self.brep_original)
        
        # Crop logic: create a bounding box and crop
        min_bound = cropped_brep.get_min_bound()
        max_bound = cropped_brep.get_max_bound()
        min_bound[axis_index] = self.slicing_pos
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        self.brep = cropped_brep.crop(bbox)
        self._update_brep_transparency()


    def _save_render(self, vis):
        """Saves a high-resolution render of the current view."""
        print(f"Saving render to {self.output_path}...")
        vis.capture_screen_image(self.output_path, do_render=True)
        print("Render saved.")
        return False

    def _update_geometries(self, vis):
        """Updates the geometries in the visualizer based on the current state."""
        if not self.view_changed:
            return

        vis.clear_geometries()
        if self.vis_mode in [0, 1]: # Show PCD
            vis.add_geometry(self.pcd, reset_bounding_box=False)
        if self.vis_mode in [0, 2]: # Show BRep
            vis.add_geometry(self.brep, reset_bounding_box=False)
        
        self.view_changed = False


    def run_visualization(self):
        """Starts the Open3D visualization window."""
        print("\n--- Interactive Controls ---")
        print("  T: Toggle visibility (Both / PCD / BRep)")
        print("  Up/Down Arrows: Adjust BRep transparency")
        print("  X/Y/Z: Toggle slicing on an axis")
        print("  Left/Right Arrows: Move slicing plane")
        print("  S: Save a high-resolution render")
        print("  Q: Quit")
        print("--------------------------\n")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # Register key callbacks
        vis.register_key_callback(ord("T"), self._toggle_visibility)
        vis.register_key_callback(ord("S"), self._save_render)
        vis.register_key_callback(ord("Q"), lambda v: v.destroy_window())
        vis.register_key_callback(265, lambda v: self._adjust_transparency(v, 0.05))  # Up Arrow
        vis.register_key_callback(264, lambda v: self._adjust_transparency(v, -0.05)) # Down Arrow
        vis.register_key_callback(ord("X"), lambda v: self._set_slicing_axis(v, 'x'))
        vis.register_key_callback(ord("Y"), lambda v: self._set_slicing_axis(v, 'y'))
        vis.register_key_callback(ord("Z"), lambda v: self._set_slicing_axis(v, 'z'))
        vis.register_key_callback(263, lambda v: self._adjust_slicing(v, -0.02)) # Left Arrow
        vis.register_key_callback(262, lambda v: self._adjust_slicing(v, 0.02))  # Right Arrow

        # Initial geometry setup
        self._update_geometries(vis)
        
        # Main loop
        while True:
            self._update_geometries(vis)
            if not vis.poll_events():
                break
            vis.update_renderer()
        
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="BRep Parameter Visualization Tool")
    parser.add_argument("pcd_path", help="Path to the input point cloud file (.ply, .pcd)")
    parser.add_argument("brep_path", help="Path to the input BRep/mesh file (.obj, .stl)")
    parser.add_argument("--transparency", type=float, default=0.5, help="Initial transparency of the BRep model (0.0 to 1.0)")
    parser.add_argument("--output_path", type=str, default="render.png", help="Path to save high-resolution renders")
    args = parser.parse_args()

    visualizer = BRepVisualizer(
        pcd_path=args.pcd_path,
        brep_path=args.brep_path,
        transparency=args.transparency,
        output_path=args.output_path
    )
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
