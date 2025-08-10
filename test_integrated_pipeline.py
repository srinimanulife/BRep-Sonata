
import pytest
import os
import numpy as np
import open3d as o3d
from unittest.mock import patch, MagicMock
from pathlib import Path

# This is a bit of a hack to be able to import the script file.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from integrated_pipeline import SonataSegmenter, BRepExtractor, IntegratedPipeline, DEFAULT_CONFIG

@pytest.fixture
def default_config():
    """Return a copy of the default config for modification."""
    config = DEFAULT_CONFIG.copy()
    # Use a temporary directory for checkpoints
    config['output']['checkpoint_dir'] = 'test_checkpoints'
    return config

@pytest.fixture
def mock_point_cloud():
    """Create a dummy Open3D PointCloud object."""
    pcd = MagicMock()
    points = np.random.rand(1000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.has_points.return_value = True
    pcd.get_center.return_value = np.mean(points, axis=0)
    # Mock select_by_index to return a smaller point cloud
    pcd.select_by_index.return_value = pcd
    return pcd

# --- SonataSegmenter (Simulated) Tests ---

def test_simulated_segmenter_output(default_config, mock_point_cloud):
    """Test the output of the simulated segmenter in integrated_pipeline."""
    segmenter = SonataSegmenter(default_config['segmentation'])
    labels = segmenter.segment(mock_point_cloud)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(mock_point_cloud.points)
    # Check if it produced the expected geometric labels
    assert set(np.unique(labels)).issubset({-1, 1, 2, 3})

# --- BRepExtractor Tests ---

@patch('open3d.geometry.PointCloud.segment_plane')
def test_brep_extractor_logic(mock_segment_plane, default_config, mock_point_cloud):
    """Test the BRep extractor's logic to fit primitives."""
    # Make plane fitting fail to test fallback logic
    mock_segment_plane.side_effect = RuntimeError("Failed to fit plane")
    
    extractor = BRepExtractor(default_config)
    
    # The mock cloud will "fail" plane fitting and should be identified as a sphere
    segments = {'1': mock_point_cloud}
    results = extractor.extract(segments)
    
    assert '1' in results
    assert results['1']['type'] == 'sphere' # Should fall back to sphere
    assert 'center' in results['1']
    assert 'radius' in results['1']

# --- IntegratedPipeline Tests ---

@pytest.fixture
def pipeline_test_files(tmp_path):
    """Create dummy files for pipeline testing."""
    pcd_path = tmp_path / "pipeline_test.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    
    # Also create the directory for checkpoints
    (tmp_path / 'test_checkpoints').mkdir()
    
    return [pcd_path]

@patch('integrated_pipeline.SonataSegmenter')
@patch('integrated_pipeline.BRepExtractor')
def test_pipeline_initialization(mock_brep, mock_segmenter, default_config, tmp_path):
    """Test that the pipeline initializes and creates the checkpoint directory."""
    default_config['output']['checkpoint_dir'] = str(tmp_path / 'test_checkpoints')
    pipeline = IntegratedPipeline(default_config)
    assert pipeline.config is not None
    assert (tmp_path / 'test_checkpoints').exists()

@patch('integrated_pipeline.o3d.io.write_point_cloud')
@patch('integrated_pipeline.json.dump')
def test_pipeline_single_file_processing(mock_json, mock_write_pcd, pipeline_test_files, default_config, tmp_path):
    """Test the end-to-end processing for a single valid file."""
    default_config['output']['checkpoint_dir'] = str(tmp_path / 'test_checkpoints')
    pipeline = IntegratedPipeline(default_config)
    result = pipeline._process_single(pipeline_test_files[0])

    assert result['status'] == 'Success'
    assert 'timings' in result
    # Check that checkpointing was attempted
    mock_write_pcd.assert_called_once()
    mock_json.assert_called_once()

def test_pipeline_handles_io_error(default_config, tmp_path):
    """Test that the pipeline gracefully handles a non-existent file."""
    default_config['output']['checkpoint_dir'] = str(tmp_path / 'test_checkpoints')
    pipeline = IntegratedPipeline(default_config)
    non_existent_file = Path("non_existent_file.ply")
    result = pipeline._process_single(non_existent_file)
    
    assert result['status'] == 'Failed'
    assert 'error' in result

@patch('argparse.ArgumentParser.parse_args')
@patch('integrated_pipeline.IntegratedPipeline.run')
def test_main_function_arg_parsing(mock_run, mock_parse_args, pipeline_test_files):
    """Test the main function's argument parsing."""
    mock_parse_args.return_value = MagicMock(
        input_files=pipeline_test_files,
        output_dir=None, # Test default
        no_parallel=False
    )
    
    # Mock matplotlib since it's imported in main
    with patch.dict('sys.modules', {'matplotlib.pyplot': MagicMock()}):
        from integrated_pipeline import main as integrated_main
        integrated_main()

    mock_run.assert_called_once_with(pipeline_test_files, parallel=True)

@patch('concurrent.futures.ProcessPoolExecutor.submit')
def test_parallel_execution_submission(mock_submit, pipeline_test_files, default_config, tmp_path):
    """Test that the pipeline submits jobs to the executor in parallel mode."""
    default_config['output']['checkpoint_dir'] = str(tmp_path / 'test_checkpoints')
    pipeline = IntegratedPipeline(default_config)
    
    # Mock the result of the future
    mock_future = MagicMock()
    mock_future.result.return_value = {"status": "Success"}
    mock_submit.return_value = mock_future

    pipeline.run(pipeline_test_files, parallel=True)
    
    # Check that submit was called for each file
    assert mock_submit.call_count == len(pipeline_test_files)
    mock_submit.assert_called_with(pipeline._process_single, pipeline_test_files[0])
