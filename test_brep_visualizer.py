
import pytest
import os
import numpy as np
import open3d as o3d
from unittest.mock import patch, MagicMock

# This is a bit of a hack to be able to import the script file.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from brep_visualizer import BRepVisualizer, main as brep_main

@pytest.fixture
def sample_files(tmp_path):
    """Create dummy point cloud and brep files for testing."""
    pcd_path = tmp_path / "sample.ply"
    brep_path = tmp_path / "sample.obj"

    # Create a simple point cloud file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    # Create a simple mesh file
    mesh = o3d.geometry.TriangleMesh.create_box()
    o3d.io.write_triangle_mesh(str(brep_path), mesh)

    return str(pcd_path), str(brep_path)

def test_visualizer_initialization(sample_files):
    """Test if the BRepVisualizer class initializes correctly."""
    pcd_path, brep_path = sample_files
    visualizer = BRepVisualizer(pcd_path, brep_path, transparency=0.7)
    
    assert visualizer.pcd is not None
    assert visualizer.brep is not None
    assert not visualizer.pcd.is_empty()
    assert not visualizer.brep.is_empty()
    assert visualizer.transparency == 0.7
    assert visualizer.vis_mode == 0

def test_color_brep_by_height(sample_files):
    """Test the height-based coloring logic."""
    pcd_path, brep_path = sample_files
    visualizer = BRepVisualizer(pcd_path, brep_path)
    
    # The mesh should have vertex colors after initialization
    assert visualizer.brep.has_vertex_colors()
    
    # All colors should be within the valid range [0, 1]
    colors = np.asarray(visualizer.brep.vertex_colors)
    assert np.all(colors >= 0) and np.all(colors <= 1)

@patch('argparse.ArgumentParser.parse_args')
def test_main_argument_parsing(mock_parse_args, sample_files):
    """Test if the main function parses arguments correctly."""
    pcd_path, brep_path = sample_files
    
    # Mock the command-line arguments
    mock_parse_args.return_value = MagicMock(
        pcd_path=pcd_path,
        brep_path=brep_path,
        transparency=0.9,
        output_path="test_render.png"
    )

    # Mock the visualizer and its run method to prevent the UI from starting
    with patch('brep_visualizer.BRepVisualizer') as mock_visualizer_class:
        mock_viz_instance = MagicMock()
        mock_visualizer_class.return_value = mock_viz_instance

        brep_main()

        # Check if BRepVisualizer was called with the correct arguments
        mock_visualizer_class.assert_called_once_with(
            pcd_path=pcd_path,
            brep_path=brep_path,
            transparency=0.9,
            output_path="test_render.png"
        )
        # Check if the visualization was started
        mock_viz_instance.run_visualization.assert_called_once()

def test_transparency_adjustment(sample_files):
    """Test the transparency adjustment logic."""
    pcd_path, brep_path = sample_files
    visualizer = BRepVisualizer(pcd_path, brep_path, transparency=0.5)
    
    # Mock the visualizer window
    mock_vis = MagicMock()

    # Increase transparency
    visualizer._adjust_transparency(mock_vis, 0.2)
    assert np.isclose(visualizer.transparency, 0.7)

    # Decrease transparency
    visualizer._adjust_transparency(mock_vis, -0.4)
    assert np.isclose(visualizer.transparency, 0.3)

    # Test clamping at 1.0
    visualizer._adjust_transparency(mock_vis, 1.0)
    assert np.isclose(visualizer.transparency, 1.0)

    # Test clamping at 0.0
    visualizer._adjust_transparency(mock_vis, -2.0)
    assert np.isclose(visualizer.transparency, 0.0)

def test_visibility_toggle(sample_files):
    """Test the visibility toggle logic."""
    pcd_path, brep_path = sample_files
    visualizer = BRepVisualizer(pcd_path, brep_path)
    mock_vis = MagicMock()

    assert visualizer.vis_mode == 0  # Initial: Both
    visualizer._toggle_visibility(mock_vis)
    assert visualizer.vis_mode == 1  # PCD only
    visualizer._toggle_visibility(mock_vis)
    assert visualizer.vis_mode == 2  # BRep only
    visualizer._toggle_visibility(mock_vis)
    assert visualizer.vis_mode == 0  # Back to Both

def test_slicing_logic(sample_files):
    """Test the slicing functionality."""
    pcd_path, brep_path = sample_files
    visualizer = BRepVisualizer(pcd_path, brep_path)
    mock_vis = MagicMock()

    # Initially, slicing is disabled
    assert visualizer.slicing_axis is None

    # Enable slicing on X-axis
    visualizer._set_slicing_axis(mock_vis, 'x')
    assert visualizer.slicing_axis == 'x'
    original_brep_vertices = np.asarray(visualizer.brep_original.vertices)
    sliced_brep_vertices = np.asarray(visualizer.brep.vertices)
    # After slicing, the number of vertices should be less than or equal to the original
    assert sliced_brep_vertices.shape[0] <= original_brep_vertices.shape[0]

    # Adjust slicing position
    initial_pos = visualizer.slicing_pos
    visualizer._adjust_slicing(mock_vis, 0.1) # Move plane
    assert visualizer.slicing_pos != initial_pos

    # Disable slicing
    visualizer._set_slicing_axis(mock_vis, 'x') # Toggle off
    assert visualizer.slicing_axis is None
    # After disabling, the mesh should be restored
    restored_brep_vertices = np.asarray(visualizer.brep.vertices)
    assert np.array_equal(restored_brep_vertices, original_brep_vertices)
