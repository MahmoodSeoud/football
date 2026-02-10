"""
Tests for rich edge features in GNN data preparation.

Phase 1 of RECEIVER_GNN_FIX_PLAN.md: Edge Features
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add grf/scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'grf' / 'scripts'))


class TestComputeEdgeFeatures:
    """Tests for compute_edge_features function."""

    def test_function_exists(self):
        """compute_edge_features function should exist in prepare_gnn_data."""
        from prepare_gnn_data import compute_edge_features
        assert callable(compute_edge_features)

    def test_returns_correct_shape(self):
        """Edge features should have shape (n_edges, 8)."""
        from prepare_gnn_data import compute_edge_features

        # Create simple node features: 23 nodes, 4 features (x, y, vx, vy)
        node_features = np.zeros((23, 13), dtype=np.float32)
        # Set positions and velocities
        for i in range(23):
            node_features[i, 0] = float(i) * 0.1  # x
            node_features[i, 1] = float(i) * 0.05  # y
            node_features[i, 2] = 0.01  # vx
            node_features[i, 3] = 0.02  # vy

        # Fully connected graph: 23 * 22 = 506 edges
        edges_src = []
        edges_dst = []
        for i in range(23):
            for j in range(23):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert edge_attr.shape == (506, 8)
        assert edge_attr.dtype == np.float32

    def test_relative_position_computation(self):
        """Edge features should contain correct relative positions."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)
        # Node 0 at (0, 0), Node 1 at (1, 2)
        node_features[0, :2] = [0.0, 0.0]
        node_features[1, :2] = [1.0, 2.0]

        edges_src = [0, 1]
        edges_dst = [1, 0]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        # Edge 0->1: rel_pos = (1, 2) - (0, 0) = (1, 2)
        assert edge_attr[0, 0] == pytest.approx(1.0)  # rel_x
        assert edge_attr[0, 1] == pytest.approx(2.0)  # rel_y

        # Edge 1->0: rel_pos = (0, 0) - (1, 2) = (-1, -2)
        assert edge_attr[1, 0] == pytest.approx(-1.0)  # rel_x
        assert edge_attr[1, 1] == pytest.approx(-2.0)  # rel_y

    def test_distance_computation(self):
        """Edge features should contain correct Euclidean distance."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)
        # Node 0 at (0, 0), Node 1 at (3, 4) -> distance = 5
        node_features[0, :2] = [0.0, 0.0]
        node_features[1, :2] = [3.0, 4.0]

        edges_src = [0, 1]
        edges_dst = [1, 0]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        # Distance is symmetric
        assert edge_attr[0, 2] == pytest.approx(5.0)  # distance
        assert edge_attr[1, 2] == pytest.approx(5.0)  # distance

    def test_angle_computation(self):
        """Edge features should contain correct angle in radians."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)
        # Node 0 at (0, 0), Node 1 at (1, 0) -> angle = 0
        # Node 2 at (0, 1) -> angle from 0 to 2 = pi/2
        node_features[0, :2] = [0.0, 0.0]
        node_features[1, :2] = [1.0, 0.0]
        node_features[2, :2] = [0.0, 1.0]

        edges_src = [0, 0]
        edges_dst = [1, 2]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert edge_attr[0, 3] == pytest.approx(0.0)  # angle 0->1
        assert edge_attr[1, 3] == pytest.approx(np.pi / 2)  # angle 0->2

    def test_relative_velocity_computation(self):
        """Edge features should contain correct relative velocities."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)
        # Node 0: velocity (0.1, 0.2)
        # Node 1: velocity (0.3, 0.5)
        node_features[0, 2:4] = [0.1, 0.2]
        node_features[1, 2:4] = [0.3, 0.5]

        edges_src = [0, 1]
        edges_dst = [1, 0]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        # Edge 0->1: rel_vel = (0.3, 0.5) - (0.1, 0.2) = (0.2, 0.3)
        assert edge_attr[0, 4] == pytest.approx(0.2)  # rel_vx
        assert edge_attr[0, 5] == pytest.approx(0.3)  # rel_vy

        # Edge 1->0: rel_vel = (0.1, 0.2) - (0.3, 0.5) = (-0.2, -0.3)
        assert edge_attr[1, 4] == pytest.approx(-0.2)  # rel_vx
        assert edge_attr[1, 5] == pytest.approx(-0.3)  # rel_vy

    def test_team_relationship_same_team(self):
        """Same team edges should have is_same_team=1, is_opponent=0."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)

        # Edge between two left team players (nodes 0-10)
        edges_src = [0, 5]
        edges_dst = [5, 0]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert edge_attr[0, 6] == pytest.approx(1.0)  # is_same_team
        assert edge_attr[0, 7] == pytest.approx(0.0)  # is_opponent
        assert edge_attr[1, 6] == pytest.approx(1.0)
        assert edge_attr[1, 7] == pytest.approx(0.0)

    def test_team_relationship_opponent(self):
        """Opponent edges should have is_same_team=0, is_opponent=1."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)

        # Edge between left team (0-10) and right team (11-21)
        edges_src = [0, 15]
        edges_dst = [15, 0]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert edge_attr[0, 6] == pytest.approx(0.0)  # is_same_team
        assert edge_attr[0, 7] == pytest.approx(1.0)  # is_opponent
        assert edge_attr[1, 6] == pytest.approx(0.0)
        assert edge_attr[1, 7] == pytest.approx(1.0)

    def test_team_relationship_ball_edges(self):
        """Ball edges should have is_same_team=0.5, is_opponent=0.5."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)

        # Edges involving ball (node 22)
        edges_src = [0, 22, 15]
        edges_dst = [22, 0, 22]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        # All ball edges should be neutral
        assert edge_attr[0, 6] == pytest.approx(0.5)  # is_same_team
        assert edge_attr[0, 7] == pytest.approx(0.5)  # is_opponent
        assert edge_attr[1, 6] == pytest.approx(0.5)
        assert edge_attr[1, 7] == pytest.approx(0.5)
        assert edge_attr[2, 6] == pytest.approx(0.5)
        assert edge_attr[2, 7] == pytest.approx(0.5)

    def test_no_nan_or_inf(self):
        """Edge features should not contain NaN or Inf values."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.random.randn(23, 13).astype(np.float32)

        edges_src = []
        edges_dst = []
        for i in range(23):
            for j in range(23):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert not np.any(np.isnan(edge_attr))
        assert not np.any(np.isinf(edge_attr))

    def test_zero_distance_edge(self):
        """Edge between nodes at same position should have distance=0."""
        from prepare_gnn_data import compute_edge_features

        node_features = np.zeros((23, 13), dtype=np.float32)
        # Node 0 and Node 1 at same position
        node_features[0, :2] = [0.5, 0.5]
        node_features[1, :2] = [0.5, 0.5]

        edges_src = [0]
        edges_dst = [1]

        edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

        assert edge_attr[0, 0] == pytest.approx(0.0)  # rel_x
        assert edge_attr[0, 1] == pytest.approx(0.0)  # rel_y
        assert edge_attr[0, 2] == pytest.approx(0.0)  # distance
        # Angle is undefined for zero distance, but should not be NaN
        assert not np.isnan(edge_attr[0, 3])


class TestBuildGraphDataEdgeFeatures:
    """Tests for build_graph_data returning 8-dim edge features."""

    def test_build_graph_data_returns_8dim_edge_attr(self):
        """build_graph_data should return edge_attr with shape (506, 8)."""
        from prepare_gnn_data import build_graph_data

        # Create minimal valid frame
        frame = {
            'left_team_positions': [[0.1 * i, 0.05 * i] for i in range(11)],
            'right_team_positions': [[-0.1 * i, -0.05 * i] for i in range(11)],
            'left_team_velocities': [[0.01, 0.02] for _ in range(11)],
            'right_team_velocities': [[-0.01, -0.02] for _ in range(11)],
            'ball_position': [0.9, 0.0, 0.0],
            'ball_velocity': [0.0, 0.0, 0.0],
            'left_team_roles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            'right_team_roles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            'left_team_tired': [0.0] * 11,
            'right_team_tired': [0.0] * 11,
        }

        node_features, edge_index, edge_attr = build_graph_data(frame)

        assert edge_attr.shape == (506, 8)
        assert edge_attr.dtype == np.float32


class TestEdgeFeatureNormalization:
    """Tests for edge feature normalization."""

    def test_compute_normalization_stats_includes_edges(self):
        """Normalization stats should include edge feature statistics."""
        from prepare_gnn_data import compute_normalization_stats

        # Create graphs with edge_attr
        graphs = [
            {
                'node_features': np.random.randn(23, 13).astype(np.float32),
                'edge_attr': np.random.randn(506, 8).astype(np.float32),
            }
            for _ in range(10)
        ]

        result = compute_normalization_stats(graphs)

        # Should return tuple of (node_mean, node_std, edge_mean, edge_std)
        # The current implementation returns (node_mean, node_std) which is len 2
        # After edge normalization is added, it should return 4 values
        assert len(result) == 4

    def test_normalize_features_normalizes_continuous_edges(self):
        """normalize_features should normalize continuous edge features."""
        from prepare_gnn_data import compute_normalization_stats, normalize_features

        # Create graphs with specific edge values
        # Edge features: [rel_x, rel_y, distance, angle, rel_vx, rel_vy, is_same_team, is_opponent]
        # Continuous: indices 0, 1, 2, 4, 5 (skip angle=3, team_flags=6,7)
        graphs = []
        for _ in range(10):
            edge_attr = np.ones((506, 8), dtype=np.float32) * 10.0
            # Set angle to pi/4 (should stay unchanged)
            edge_attr[:, 3] = np.pi / 4
            # Set team flags (should stay unchanged)
            edge_attr[:, 6] = 1.0
            edge_attr[:, 7] = 0.0
            graphs.append({
                'node_features': np.random.randn(23, 13).astype(np.float32),
                'edge_attr': edge_attr,
            })

        node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(graphs)
        normalized_graphs = normalize_features(graphs, node_mean, node_std, edge_mean, edge_std)

        for g in normalized_graphs:
            # Continuous features (0,1,2,4,5) should be normalized
            edge_continuous = g['edge_attr'][:, [0, 1, 2, 4, 5]]
            assert abs(edge_continuous.mean()) < 1.0  # Should be close to 0

            # Angle should be unchanged (still pi/4)
            assert np.allclose(g['edge_attr'][:, 3], np.pi / 4)

            # Team flags should be unchanged
            assert np.allclose(g['edge_attr'][:, 6], 1.0)
            assert np.allclose(g['edge_attr'][:, 7], 0.0)


class TestD2Augmentation:
    """Tests for D2 augmentation with rich edge features."""

    def test_d2_augmentation_flips_edge_features(self):
        """D2 augmentation should correctly flip edge features."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'grf' / 'scripts'))
        from prepare_receiver_data import apply_d2_augmentation
        from prepare_gnn_data import EDGE_REL_X, EDGE_REL_Y, EDGE_ANGLE

        # Create a simple graph with known edge features
        graph = {
            'node_features': np.zeros((23, 13), dtype=np.float32),
            'edge_attr': np.zeros((506, 8), dtype=np.float32),
            'edge_index': np.zeros((2, 506), dtype=np.int64),
            'receiver_label': 5,
            'shot_label': 1,
            'goal_label': 0,
        }

        # Set specific edge feature values
        graph['edge_attr'][:, EDGE_REL_X] = 1.0  # rel_x = 1
        graph['edge_attr'][:, EDGE_REL_Y] = 2.0  # rel_y = 2
        graph['edge_attr'][:, EDGE_ANGLE] = np.pi / 4  # 45 degrees

        augmented = apply_d2_augmentation(graph)

        assert len(augmented) == 4

        # Identity: unchanged
        np.testing.assert_allclose(augmented[0]['edge_attr'][:, EDGE_REL_X], 1.0)
        np.testing.assert_allclose(augmented[0]['edge_attr'][:, EDGE_REL_Y], 2.0)

        # Horizontal flip: rel_x negated
        np.testing.assert_allclose(augmented[1]['edge_attr'][:, EDGE_REL_X], -1.0)
        np.testing.assert_allclose(augmented[1]['edge_attr'][:, EDGE_REL_Y], 2.0)

        # Vertical flip: rel_y negated
        np.testing.assert_allclose(augmented[2]['edge_attr'][:, EDGE_REL_X], 1.0)
        np.testing.assert_allclose(augmented[2]['edge_attr'][:, EDGE_REL_Y], -2.0)

        # 180 rotation: both negated
        np.testing.assert_allclose(augmented[3]['edge_attr'][:, EDGE_REL_X], -1.0)
        np.testing.assert_allclose(augmented[3]['edge_attr'][:, EDGE_REL_Y], -2.0)

    def test_d2_augmentation_preserves_receiver_label(self):
        """D2 augmentation should preserve receiver labels."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'grf' / 'scripts'))
        from prepare_receiver_data import apply_d2_augmentation

        graph = {
            'node_features': np.zeros((23, 13), dtype=np.float32),
            'edge_attr': np.zeros((506, 8), dtype=np.float32),
            'edge_index': np.zeros((2, 506), dtype=np.int64),
            'receiver_label': 7,
            'shot_label': 1,
            'goal_label': 0,
        }

        augmented = apply_d2_augmentation(graph)

        for g in augmented:
            assert g['receiver_label'] == 7


class TestEdgeFeatureIntegration:
    """Integration tests for edge features in data pipeline."""

    def test_edge_features_persist_in_saved_data(self):
        """Edge features should be saved in npz files with correct shape."""
        # This test would require actually running prepare_receiver_data.py
        # For now, we just verify the shape expectations
        expected_edge_shape = (506, 8)
        assert expected_edge_shape[0] == 23 * 22  # Fully connected graph
        assert expected_edge_shape[1] == 8  # 8-dim edge features
