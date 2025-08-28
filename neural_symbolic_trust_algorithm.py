#!/usr/bin/env python3
"""
Neural Symbolic Trust Algorithm using Graph Neural Networks

This module implements a GNN-based trust algorithm that learns from
simulation data and uses neural symbolic predicates (InFoV, Observed, Trustworthy)
to estimate trust values for agents and tracks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

from trust_algorithm import TrustAlgorithm, RobotState, Track
from paper_trust_algorithm import PaperTrustAlgorithm


@dataclass
class GraphNode:
    """Graph node representing either an agent or track"""
    node_id: str
    node_type: str  # 'agent' or 'track'
    features: np.ndarray
    position: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    """Graph edge with relationship type and features"""
    src_node: str
    dst_node: str
    edge_type: str  # 'in_fov', 'observed'
    features: np.ndarray
    confidence: float


@dataclass
class TrainingExample:
    """Training example for the GNN"""
    graph_data: HeteroData
    labels: Dict[str, Dict[str, float]]  # node_id -> {'value': float, 'confidence': float}
    metadata: Dict[str, Any]


class TrustGNN(nn.Module):
    """Graph Neural Network for trust estimation"""
    
    def __init__(self, agent_features: int, track_features: int, hidden_dim: int = 64):
        super(TrustGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding layers
        self.agent_embedding = Linear(agent_features, hidden_dim)
        self.track_embedding = Linear(track_features, hidden_dim)
        
        # Heterogeneous graph convolution layers using SAGEConv for bipartite support
        self.conv1 = HeteroConv({
            ('agent', 'observes', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'observed_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
            ('agent', 'in_fov', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'in_fov_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
        })
        
        self.conv2 = HeteroConv({
            ('agent', 'observes', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'observed_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
            ('agent', 'in_fov', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'in_fov_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
        })
        
        # Output heads for value and confidence prediction
        self.agent_value_head = Linear(hidden_dim, 1)
        self.agent_confidence_head = Linear(hidden_dim, 1)
        self.track_value_head = Linear(hidden_dim, 1)
        self.track_confidence_head = Linear(hidden_dim, 1)
        
        # Symbolic reasoning layer
        self.symbolic_layer = SymbolicReasoningLayer(hidden_dim)
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """Forward pass through the GNN"""
        # Initial embeddings
        x_dict = {
            'agent': F.relu(self.agent_embedding(x_dict['agent'])),
            'track': F.relu(self.track_embedding(x_dict['track']))
        }
        
        # Graph convolution layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply symbolic reasoning
        x_dict = self.symbolic_layer(x_dict)
        
        # Generate predictions - handle missing node types
        predictions = {}
        
        if 'agent' in x_dict:
            agent_values = torch.sigmoid(self.agent_value_head(x_dict['agent']))
            agent_confidences = torch.sigmoid(self.agent_confidence_head(x_dict['agent']))
            predictions['agent'] = {'value': agent_values, 'confidence': agent_confidences}
        
        if 'track' in x_dict:
            track_values = torch.sigmoid(self.track_value_head(x_dict['track']))
            track_confidences = torch.sigmoid(self.track_confidence_head(x_dict['track']))
            predictions['track'] = {'value': track_values, 'confidence': track_confidences}
        
        return predictions


class SymbolicReasoningLayer(nn.Module):
    """Neural symbolic reasoning layer that applies logical constraints"""
    
    def __init__(self, hidden_dim: int):
        super(SymbolicReasoningLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable parameters for symbolic rules
        self.trustworthy_threshold = nn.Parameter(torch.tensor(0.5))
        self.consensus_weight = nn.Parameter(torch.tensor(1.0))
        self.fov_weight = nn.Parameter(torch.tensor(0.8))
        
        # Rule application networks
        self.rule_network = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x_dict):
        """Apply symbolic reasoning rules"""
        # Rule 1: Trustworthy agents should have consistent observations
        if 'agent' in x_dict:
            agent_features = x_dict['agent']
            rule_adjustment = self.rule_network(agent_features)
            x_dict['agent'] = agent_features + 0.1 * rule_adjustment
        
        # Rule 2: Tracks observed by multiple agents are more trustworthy
        if 'track' in x_dict:
            track_features = x_dict['track']
            rule_adjustment = self.rule_network(track_features)
            x_dict['track'] = track_features + 0.1 * rule_adjustment
        
        return x_dict


class GraphBuilder:
    """Builds graph structures from simulation state"""
    
    def __init__(self, fov_threshold: float = 20.0):
        self.fov_threshold = fov_threshold
    
    def build_graph(self, ego_robot: RobotState, ego_fused_tracks: List[Track],
                   proximal_robots: List[RobotState], 
                   proximal_tracks_by_robot: Dict[int, List[Track]]) -> HeteroData:
        """Build heterogeneous graph from simulation state"""
        
        # Create graph data structure
        data = HeteroData()
        
        # Build all robots list
        all_robots = [ego_robot] + proximal_robots
        
        # Build edges first to determine which agents and tracks are connected
        edges = self._build_edges(all_robots, ego_fused_tracks, proximal_tracks_by_robot)
        
        # Then build agent and track nodes using only connected ones
        agent_nodes, agent_features = self._build_agent_nodes(all_robots)
        track_nodes, track_features = self._build_track_nodes(ego_fused_tracks)
        
        # Populate graph data
        data['agent'].x = torch.tensor(agent_features, dtype=torch.float)
        data['track'].x = torch.tensor(track_features, dtype=torch.float)
        
        # Add edges
        for edge_type, edge_indices in edges.items():
            if edge_indices:
                src_nodes, dst_nodes = zip(*edge_indices)
                data[edge_type].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # Store node mappings for later use
        data.agent_nodes = agent_nodes
        data.track_nodes = track_nodes
        
        return data
    
    def _build_agent_nodes(self, all_robots: List[RobotState]) -> Tuple[Dict, np.ndarray]:
        """Build agent nodes and features"""
        agent_nodes = {}
        agent_features = []
        
        # Filter agents to only include those with connections to tracks
        connected_agents = getattr(self, '_connected_agents', set())
        
        agent_idx = 0
        for original_idx, robot in enumerate(all_robots):
            # Only include agents that have at least one edge to a track
            if not connected_agents or original_idx in connected_agents:
                node_id = f"agent_{robot.id}"
                agent_nodes[robot.id] = agent_idx
                
                # Agent features: Only trustworthy predicate [trustworthy_predicate(1)]
                trustworthy_pred = self._compute_trustworthy_predicate(robot)
                
                # Store ground truth and current trust for loss computation (not part of GNN features)
                robot._ground_truth_trustworthy = 1.0 - float(robot.is_adversarial)  # Store for loss
                robot._current_trust_alpha = robot.trust_alpha  # Store for loss  
                robot._current_trust_beta = robot.trust_beta   # Store for loss
                
                features = np.array([trustworthy_pred])  # Only trustworthy predicate
                
                agent_features.append(features)
                agent_idx += 1
        
        return agent_nodes, np.array(agent_features)
    
    def _build_track_nodes(self, ego_fused_tracks: List[Track]) -> Tuple[Dict, np.ndarray]:
        """Build track nodes and features"""
        track_nodes = {}
        track_features = []
        
        # Filter tracks to only include those connected to agents
        connected_tracks = getattr(self, '_connected_tracks', set())
        
        # Add ego fused tracks that have connections to agents
        track_idx = 0
        for original_idx, track in enumerate(ego_fused_tracks):
            # Only include tracks that have at least one edge to an agent
            if not connected_tracks or original_idx in connected_tracks:
                node_id = f"track_{track.id}"
                track_nodes[track.id] = track_idx
                
                # Track features: only trustworthy predicate
                trustworthy_pred = self._compute_track_trustworthy_predicate(track)
                
                features = np.array([trustworthy_pred])
                
                track_features.append(features)
                track_idx += 1
        
        return track_nodes, np.array(track_features) if track_features else np.array([]).reshape(0, 1)
    
    def _build_edges(self, all_robots: List[RobotState], ego_fused_tracks: List[Track],
                    proximal_tracks_by_robot: Dict[int, List[Track]]) -> Dict[str, List[Tuple[int, int]]]:
        """Build edges representing neural symbolic predicates using direct observations"""
        edges = {
            ('agent', 'observes', 'track'): [],
            ('track', 'observed_by', 'agent'): [],
            ('agent', 'in_fov', 'track'): [],
            ('track', 'in_fov_by', 'agent'): []
        }
        
        # Build edges based on actual observations received by robots
        # If a robot has an observation of a track, that track is both InFoV and Observed
        
        for robot_idx, robot in enumerate(all_robots):
            if robot_idx == 0:  # Ego robot (first in all_robots list)
                # Ego robot: check which fused tracks it has source tracks for
                for track_idx, fused_track in enumerate(ego_fused_tracks):
                    ego_contributed = False
                    
                    if hasattr(fused_track, '_source_tracks'):
                        # Check if ego robot contributed to this fused track
                        ego_contributed = any(robot_id == robot.id for robot_id, _ in fused_track._source_tracks)
                    else:
                        # Fallback: check if track's source_robot matches ego robot
                        ego_contributed = (fused_track.source_robot == robot.id)
                    
                    if ego_contributed:
                        # Ego robot observed this track (and by definition, it's in FoV)
                        edges[('agent', 'observes', 'track')].append((robot_idx, track_idx))
                        edges[('track', 'observed_by', 'agent')].append((track_idx, robot_idx))
                        edges[('agent', 'in_fov', 'track')].append((robot_idx, track_idx))
                        edges[('track', 'in_fov_by', 'agent')].append((track_idx, robot_idx))
            else:
                # Proximal robots: check if they have observations that match fused tracks
                robot_tracks = proximal_tracks_by_robot.get(robot.id, [])
                for robot_track in robot_tracks:
                    # Find corresponding fused track by object_id
                    for fused_track_idx, fused_track in enumerate(ego_fused_tracks):
                        if (hasattr(robot_track, 'object_id') and hasattr(fused_track, 'object_id') and 
                            robot_track.object_id == fused_track.object_id):
                            # This robot observed this track (and by definition, it's in FoV)
                            edges[('agent', 'observes', 'track')].append((robot_idx, fused_track_idx))
                            edges[('track', 'observed_by', 'agent')].append((fused_track_idx, robot_idx))
                            edges[('agent', 'in_fov', 'track')].append((robot_idx, fused_track_idx))
                            edges[('track', 'in_fov_by', 'agent')].append((fused_track_idx, robot_idx))
                            break  # Found match, move to next robot track
        
        # For tracks that robots don't directly observe, check if they're still in FoV
        # This catches cases where a track exists but robot doesn't have it in current observations
        for robot_idx, robot in enumerate(all_robots):
            for track_idx, fused_track in enumerate(ego_fused_tracks):
                # Check if we already added edges for this robot-track pair
                if (robot_idx, track_idx) not in [(r, t) for r, t in edges[('agent', 'in_fov', 'track')]]:
                    # Use geometric check for InFoV (but not Observed)
                    if self._is_track_in_fov_predicate(robot, fused_track.position):
                        edges[('agent', 'in_fov', 'track')].append((robot_idx, track_idx))
                        edges[('track', 'in_fov_by', 'agent')].append((track_idx, robot_idx))
        
        # Remove tracks that have no connections to any agents
        # This prevents isolated track nodes in the graph
        connected_tracks = set()
        for edge_list in edges.values():
            for src, dst in edge_list:
                # Find track indices in edge relationships
                if edge_list == edges[('agent', 'in_fov', 'track')] or edge_list == edges[('agent', 'observes', 'track')]:
                    connected_tracks.add(dst)  # dst is track index
                elif edge_list == edges[('track', 'in_fov_by', 'agent')] or edge_list == edges[('track', 'observed_by', 'agent')]:
                    connected_tracks.add(src)  # src is track index
        
        # Create mapping from original track indices to filtered track indices
        track_original_to_filtered = {}
        filtered_track_idx = 0
        for original_idx in sorted(connected_tracks):
            track_original_to_filtered[original_idx] = filtered_track_idx
            filtered_track_idx += 1
        
        # Determine which agents have connections
        connected_agents = set()
        for edge_list in edges.values():
            for src, dst in edge_list:
                if edge_list == edges[('agent', 'in_fov', 'track')] or edge_list == edges[('agent', 'observes', 'track')]:
                    connected_agents.add(src)  # src is agent index
                elif edge_list == edges[('track', 'in_fov_by', 'agent')] or edge_list == edges[('track', 'observed_by', 'agent')]:
                    connected_agents.add(dst)  # dst is agent index
        
        # Create mapping from original agent indices to filtered agent indices
        agent_original_to_filtered = {}
        filtered_agent_idx = 0
        for original_idx in sorted(connected_agents):
            agent_original_to_filtered[original_idx] = filtered_agent_idx
            filtered_agent_idx += 1
        
        # Update edge indices to use filtered agent and track indices
        filtered_edges = {
            ('agent', 'observes', 'track'): [],
            ('track', 'observed_by', 'agent'): [],
            ('agent', 'in_fov', 'track'): [],
            ('track', 'in_fov_by', 'agent'): []
        }
        
        # Remap edge indices
        for (src, dst) in edges[('agent', 'in_fov', 'track')]:
            if src in agent_original_to_filtered and dst in track_original_to_filtered:
                filtered_edges[('agent', 'in_fov', 'track')].append((agent_original_to_filtered[src], track_original_to_filtered[dst]))
                
        for (src, dst) in edges[('track', 'in_fov_by', 'agent')]:
            if src in track_original_to_filtered and dst in agent_original_to_filtered:
                filtered_edges[('track', 'in_fov_by', 'agent')].append((track_original_to_filtered[src], agent_original_to_filtered[dst]))
                
        for (src, dst) in edges[('agent', 'observes', 'track')]:
            if src in agent_original_to_filtered and dst in track_original_to_filtered:
                filtered_edges[('agent', 'observes', 'track')].append((agent_original_to_filtered[src], track_original_to_filtered[dst]))
                
        for (src, dst) in edges[('track', 'observed_by', 'agent')]:
            if src in track_original_to_filtered and dst in agent_original_to_filtered:
                filtered_edges[('track', 'observed_by', 'agent')].append((track_original_to_filtered[src], agent_original_to_filtered[dst]))
        
        
        # Store for filtering in graph building
        self._connected_tracks = connected_tracks
        self._connected_agents = connected_agents
        
        return filtered_edges
    
    def _compute_trustworthy_predicate(self, robot: RobotState, threshold: float = 0.5) -> float:
        """Compute trustworthy predicate for agent using threshold comparison"""
        trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        return 1.0 if trust_value > threshold else 0.0
    
    def _compute_track_trustworthy_predicate(self, track: Track, threshold: float = 0.5) -> float:
        """Compute trustworthy predicate for track using threshold comparison"""
        track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
        combined_trust = track_trust * track.confidence
        return 1.0 if combined_trust > threshold else 0.0
    
    def _is_track_in_fov_predicate(self, observer: RobotState, track_pos: np.ndarray) -> bool:
        """InFoV binary predicate - whether track is in agent's field of view"""
        rel_pos = track_pos - observer.position
        distance = np.linalg.norm(rel_pos[:2])
        
        if distance > observer.fov_range:
            return False
        
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = abs(target_angle - observer.orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        return angle_diff <= observer.fov_angle / 2
    
    
    def _get_track_index(self, track_id: str, ego_fused_tracks: List[Track]) -> Optional[int]:
        """Get track index in the node list"""
        # Check ego fused tracks
        for i, track in enumerate(ego_fused_tracks):
            if track.id == track_id:
                return i
        return None


class TrainingDataCollector:
    """Collects and structures training data from simulations"""
    
    def __init__(self):
        self.training_examples = []
        self.graph_builder = GraphBuilder()
        # Paper algorithm to generate training targets
        self.paper_algorithm = PaperTrustAlgorithm()
        self.paper_algorithm_initialized = False
    
    def _fuse_tracks_for_training(self, ego_robot: RobotState, ego_fused_tracks: List[Track],
                                 proximal_robots: List[RobotState], 
                                 robot_object_tracks: Dict[int, Dict[str, Track]]) -> List[Track]:
        """
        Fuse tracks using the same logic as paper algorithm for training data consistency
        
        Returns all tracks that should appear in the graph:
        - Fused tracks (ego + proximal observations of same object)
        - Ego-only tracks (not observed by trusted proximal robots)  
        - Proximal-only tracks (false positives from ego's perspective)
        """
        from collections import defaultdict
        
        fusion_threshold = 5.0  # Distance threshold for track fusion
        trust_threshold = 0.5   # Trust threshold for fusion eligibility
        
        # Track all objects and their observers
        object_observers = defaultdict(list)  # object_id -> [(robot_id, track)]
        all_tracks = []
        
        # 1. Collect GENUINE ego tracks only (exclude false positive tracks from adversarial robots)
        ego_tracks = ego_fused_tracks.copy()
        for track in ego_tracks:
            if hasattr(track, 'object_id'):
                # Only include tracks that are genuinely observed by ego robot
                # Exclude false positive tracks (which have '_fp_' in their ID)
                if hasattr(track, 'id') and '_fp_' not in track.id:
                    object_observers[track.object_id].append((ego_robot.id, track))
        
        # 2. Collect proximal robot tracks
        for robot in proximal_robots:
            robot_tracks = robot_object_tracks.get(robot.id, {})
            for track in robot_tracks.values():
                if hasattr(track, 'object_id'):
                    object_observers[track.object_id].append((robot.id, track))
        
        # 3. Process each observed object
        processed_objects = set()
        
        for object_id, observers in object_observers.items():
            if object_id in processed_objects:
                continue
                
            ego_observations = [(rid, t) for rid, t in observers if rid == ego_robot.id]
            proximal_observations = [(rid, t) for rid, t in observers if rid != ego_robot.id]
            
            if ego_observations and proximal_observations:
                # Object observed by both ego and proximal robots - create fused track
                ego_track = ego_observations[0][1]  # Use first (should be only) ego observation
                
                # Find trusted proximal observations within fusion threshold
                trusted_proximal = []
                for robot_id, prox_track in proximal_observations:
                    # Find robot by ID
                    robot = next((r for r in proximal_robots if r.id == robot_id), None)
                    if robot is None:
                        continue
                        
                    # Check trust thresholds and distance
                    agent_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                    track_trust = prox_track.trust_alpha / (prox_track.trust_alpha + prox_track.trust_beta)
                    distance = np.linalg.norm(ego_track.position - prox_track.position)
                    
                    if (agent_trust >= trust_threshold and 
                        track_trust >= trust_threshold and 
                        distance <= fusion_threshold):
                        trusted_proximal.append(prox_track)
                
                if trusted_proximal:
                    # Create fused track - inherits ego's trust distribution
                    fused_track = Track(
                        id=f"{ego_robot.id}_fused_obj_{object_id}",
                        position=ego_track.position.copy(),  # For simplicity, use ego position
                        velocity=ego_track.velocity.copy(),
                        covariance=ego_track.covariance.copy(),
                        confidence=min(1.0, (ego_track.confidence + sum(t.confidence for t in trusted_proximal)) / (len(trusted_proximal) + 1)),
                        timestamp=ego_track.timestamp,
                        source_robot=ego_robot.id,
                        trust_alpha=ego_track.trust_alpha,  # Inherit ego's trust
                        trust_beta=ego_track.trust_beta,    # Inherit ego's trust
                        object_id=object_id
                    )
                    all_tracks.append(fused_track)
                else:
                    # No trusted proximal observations - use ego track as is
                    all_tracks.append(ego_track)
                    
            elif ego_observations:
                # Ego-only observation
                ego_track = ego_observations[0][1]
                all_tracks.append(ego_track)
                
            elif proximal_observations:
                # Proximal-only observations - fuse multiple proximal robot observations
                # Separate legitimate and adversarial observations
                legitimate_observations = []
                adversarial_observations = []
                
                for robot_id, prox_track in proximal_observations:
                    # Find robot by ID to check if it's adversarial
                    robot = next((r for r in proximal_robots if r.id == robot_id), None)
                    if robot is None:
                        continue
                    
                    if robot.is_adversarial:
                        adversarial_observations.append((robot_id, prox_track, robot))
                    else:
                        legitimate_observations.append((robot_id, prox_track, robot))
                
                # Create fused legitimate proximal track if multiple legitimate observations exist
                if len(legitimate_observations) > 1:
                    fused_track = self._create_fused_proximal_track(
                        ego_robot, legitimate_observations, object_id, "proximal_fused"
                    )
                    all_tracks.append(fused_track)
                    # Store mapping for trust propagation
                    setattr(fused_track, '_source_tracks', [(robot_id, track) for robot_id, track, _ in legitimate_observations])
                elif len(legitimate_observations) == 1:
                    # Single legitimate observation - create regular proximal track
                    robot_id, prox_track, robot = legitimate_observations[0]
                    track = Track(
                        id=f"{ego_robot.id}_proximal_{robot_id}_{object_id}",
                        position=prox_track.position.copy(),
                        velocity=prox_track.velocity.copy(),
                        covariance=prox_track.covariance.copy(),
                        confidence=prox_track.confidence,
                        timestamp=prox_track.timestamp,
                        source_robot=robot_id,  # Original source robot
                        trust_alpha=prox_track.trust_alpha,  # Inherit proximal track trust
                        trust_beta=prox_track.trust_beta,    # Inherit proximal track trust
                        object_id=object_id
                    )
                    all_tracks.append(track)
                    # Store mapping for trust propagation
                    setattr(track, '_source_tracks', [(robot_id, prox_track)])
                
                # Create fused adversarial track if multiple adversarial observations exist
                if len(adversarial_observations) > 1:
                    fused_track = self._create_fused_proximal_track(
                        ego_robot, adversarial_observations, object_id, "fp_fused"
                    )
                    all_tracks.append(fused_track)
                    # Store mapping for trust propagation
                    setattr(fused_track, '_source_tracks', [(robot_id, track) for robot_id, track, _ in adversarial_observations])
                elif len(adversarial_observations) == 1:
                    # Single adversarial observation - create regular false positive track
                    robot_id, prox_track, robot = adversarial_observations[0]
                    track = Track(
                        id=f"{ego_robot.id}_fp_{robot_id}_{object_id}",
                        position=prox_track.position.copy(),
                        velocity=prox_track.velocity.copy(),
                        covariance=prox_track.covariance.copy(),
                        confidence=prox_track.confidence,
                        timestamp=prox_track.timestamp,
                        source_robot=ego_robot.id,  # From ego's perspective
                        trust_alpha=1.0,  # Start with neutral trust
                        trust_beta=1.0,   # Start with neutral trust
                        object_id=object_id
                    )
                    all_tracks.append(track)
                    # Store mapping for trust propagation
                    setattr(track, '_source_tracks', [(robot_id, prox_track)])
                    
            processed_objects.add(object_id)
        
        return all_tracks
    
    def _create_fused_proximal_track(self, ego_robot: 'RobotState', observations: List[Tuple[int, 'Track', 'RobotState']], 
                                    object_id: str, track_type: str) -> 'Track':
        """
        Create a fused track from multiple proximal robot observations
        
        Args:
            ego_robot: The ego robot from whose perspective the track is created
            observations: List of (robot_id, track, robot_state) tuples
            object_id: The object ID being observed
            track_type: Type of track ("proximal_fused" or "fp_fused")
            
        Returns:
            Fused track with highest mean trust distribution
        """
        if not observations:
            raise ValueError("Cannot create fused track from empty observations")
        
        # Find observation with highest mean trust
        best_observation = None
        highest_mean_trust = -1.0
        
        for robot_id, track, robot in observations:
            mean_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            if mean_trust > highest_mean_trust:
                highest_mean_trust = mean_trust
                best_observation = (robot_id, track, robot)
        
        best_robot_id, best_track, best_robot = best_observation
        
        # Calculate fused position as weighted average by confidence
        total_confidence = sum(track.confidence for _, track, _ in observations)
        if total_confidence > 0:
            fused_position = np.zeros(3)
            fused_velocity = np.zeros(3)
            
            for robot_id, track, robot in observations:
                weight = track.confidence / total_confidence
                fused_position += weight * track.position
                fused_velocity += weight * track.velocity
        else:
            # Fallback to best track's position/velocity
            fused_position = best_track.position.copy()
            fused_velocity = best_track.velocity.copy()
        
        # Calculate average confidence
        avg_confidence = sum(track.confidence for _, track, _ in observations) / len(observations)
        
        # Create fused track ID with all source robots
        source_robot_ids = sorted([robot_id for robot_id, _, _ in observations])
        if track_type == "proximal_fused":
            track_id = f"{ego_robot.id}_proximal_fused_{'_'.join(map(str, source_robot_ids))}_{object_id}"
            source_robot = source_robot_ids[0]  # Use first robot as representative source
        else:  # fp_fused
            track_id = f"{ego_robot.id}_fp_fused_{'_'.join(map(str, source_robot_ids))}_{object_id}"
            source_robot = ego_robot.id  # From ego's perspective for false positives
        
        # Create fused track inheriting the highest mean trust
        fused_track = Track(
            id=track_id,
            position=fused_position,
            velocity=fused_velocity,
            covariance=best_track.covariance.copy(),  # Use best track's covariance
            confidence=avg_confidence,
            timestamp=max(track.timestamp for _, track, _ in observations),  # Most recent timestamp
            source_robot=source_robot,
            trust_alpha=best_track.trust_alpha,  # Inherit highest mean trust distribution
            trust_beta=best_track.trust_beta,
            object_id=object_id
        )
        
        return fused_track
    
    def _propagate_trust_updates_to_source_tracks(self, fused_tracks: List['Track'], 
                                                 robot_object_tracks: Dict[int, Dict[str, 'Track']]) -> None:
        """
        Propagate trust updates from fused tracks back to their source tracks
        
        Args:
            fused_tracks: List of fused tracks that have been updated by GNN
            robot_object_tracks: robot_id -> {object_id: Track} for updating source tracks
        """
        for fused_track in fused_tracks:
            # Check if this track has source tracks to update
            if hasattr(fused_track, '_source_tracks'):
                source_tracks = fused_track._source_tracks
                
                for robot_id, source_track in source_tracks:
                    # Find the source track in robot_object_tracks and update its trust
                    if (robot_id in robot_object_tracks and 
                        source_track.object_id in robot_object_tracks[robot_id]):
                        
                        source_track_to_update = robot_object_tracks[robot_id][source_track.object_id]
                        
                        # Update trust distribution from fused track
                        source_track_to_update.trust_alpha = fused_track.trust_alpha
                        source_track_to_update.trust_beta = fused_track.trust_beta
                        
                        # Optionally update other properties that might have been refined
                        # source_track_to_update.confidence = fused_track.confidence
    
    def collect_from_simulation_step(self, robots: List[RobotState], 
                                   tracks_by_robot: Dict[int, List[Track]],
                                   robot_object_tracks: Dict[int, Dict[str, Track]],
                                   trust_updates: Dict[int, Dict],
                                   ego_robot: RobotState,
                                   ego_fused_tracks: List[Track]) -> None:
        """Collect training data from a single simulation step using paper algorithm as ground truth"""
        
        # Initialize paper algorithm if needed
        if not self.paper_algorithm_initialized:
            self.paper_algorithm.initialize(robots)
            self.paper_algorithm_initialized = True
        
        # Create deep copies of robot states and tracks to avoid affecting main simulation
        robot_copies = []
        for robot in robots:
            robot_copy = RobotState(
                id=robot.id,
                position=robot.position.copy(),
                velocity=robot.velocity.copy(),
                orientation=robot.orientation,
                fov_range=robot.fov_range,
                fov_angle=robot.fov_angle,
                is_adversarial=robot.is_adversarial,
                trust_alpha=robot.trust_alpha,
                trust_beta=robot.trust_beta
            )
            robot_copies.append(robot_copy)
        
        # Create deep copies of tracks_by_robot and robot_object_tracks to capture paper algorithm changes
        tracks_by_robot_copy = {}
        for robot_id, tracks_list in tracks_by_robot.items():
            tracks_by_robot_copy[robot_id] = []
            for track in tracks_list:
                track_copy = Track(
                    id=track.id,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    covariance=track.covariance.copy(),
                    confidence=track.confidence,
                    timestamp=track.timestamp,
                    source_robot=track.source_robot,
                    trust_alpha=track.trust_alpha,
                    trust_beta=track.trust_beta,
                    object_id=track.object_id
                )
                tracks_by_robot_copy[robot_id].append(track_copy)
        
        robot_object_tracks_copy = {}
        for robot_id, obj_tracks in robot_object_tracks.items():
            robot_object_tracks_copy[robot_id] = {}
            for obj_id, track in obj_tracks.items():
                track_copy = Track(
                    id=track.id,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    covariance=track.covariance.copy(),
                    confidence=track.confidence,
                    timestamp=track.timestamp,
                    source_robot=track.source_robot,
                    trust_alpha=track.trust_alpha,
                    trust_beta=track.trust_beta,
                    object_id=track.object_id
                )
                robot_object_tracks_copy[robot_id][obj_id] = track_copy
        
        # Run paper algorithm to get PSM ground truth and track trust updates
        paper_trust_updates = self.paper_algorithm.update_trust(
            robot_copies, tracks_by_robot_copy, robot_object_tracks_copy, 0.0
        )
        
        # Build graph using proper track fusion logic as in paper algorithm
        # 1. Get proximal robots that have tracks
        robots_with_tracks = {r_id for r_id, tracks in robot_object_tracks.items() 
                             if tracks and r_id in [robot.id for robot in robots]}
        proximal_robots = [r for r in robots if r.id != ego_robot.id and r.id in robots_with_tracks]
        
        # 2. Perform track fusion using paper algorithm logic
        fused_tracks = self._fuse_tracks_for_training(
            ego_robot, ego_fused_tracks, proximal_robots, robot_object_tracks_copy
        )
        
        # 3. Build proximal tracks dict for edge creation (individual robot tracks)
        proximal_tracks_by_robot = {r_id: tracks for r_id, tracks in tracks_by_robot.items() 
                                  if r_id != ego_robot.id and r_id in robots_with_tracks}
        
        graph_data = self.graph_builder.build_graph(
            ego_robot, fused_tracks, proximal_robots, proximal_tracks_by_robot
        )
        
        # Collect all tracks for more diverse training data (including false positives)
        all_ego_tracks = ego_fused_tracks.copy()
        
        # Also include raw tracks from ego robot to capture false positives
        if ego_robot.id in tracks_by_robot:
            for track in tracks_by_robot[ego_robot.id]:
                # Add track if not already in fused tracks (avoid duplicates by checking object_id)
                if track.object_id and not any(ft.object_id == track.object_id for ft in all_ego_tracks):
                    all_ego_tracks.append(track)
        
        # Generate labels from paper algorithm PSMs (including track trust updates)
        labels = self._generate_labels_from_paper_algorithm(
            robots, all_ego_tracks, robot_copies, paper_trust_updates,
            tracks_by_robot, tracks_by_robot_copy, robot_object_tracks, robot_object_tracks_copy
        )
        
        # Create training example
        example = TrainingExample(
            graph_data=graph_data,
            labels=labels,
            metadata={
                'ego_robot_id': ego_robot.id,
                'num_robots': len(robots),
                'num_tracks': len(ego_fused_tracks),
                'adversarial_robots': [r.id for r in robots if r.is_adversarial],
                'used_paper_algorithm': True
            }
        )
        
        self.training_examples.append(example)
    
    def _generate_labels_from_paper_algorithm(self, original_robots: List[RobotState], 
                                            tracks: List[Track], 
                                            robot_copies_after_paper: List[RobotState],
                                            paper_trust_updates: Dict[int, Dict],
                                            original_tracks_by_robot: Dict[int, List[Track]],
                                            tracks_by_robot_after_paper: Dict[int, List[Track]],
                                            original_robot_object_tracks: Dict[int, Dict[str, Track]],
                                            robot_object_tracks_after_paper: Dict[int, Dict[str, Track]]) -> Dict[str, Dict[str, float]]:
        """Generate training labels from paper algorithm PSMs by computing trust deltas"""
        labels = {}
        
        # Create lookup for original robot states
        original_robot_lookup = {r.id: r for r in original_robots}
        paper_robot_lookup = {r.id: r for r in robot_copies_after_paper}
        
        # Agent labels from actual trust changes computed by paper algorithm
        for robot_id, paper_robot in paper_robot_lookup.items():
            if robot_id in original_robot_lookup:
                original_robot = original_robot_lookup[robot_id]
                
                # Compute the actual PSM that the paper algorithm would have applied
                # PSM update: new_alpha = old_alpha + confidence * value
                #            new_beta = old_beta + confidence * (1 - value)
                
                # Calculate what PSM the paper algorithm effectively used
                delta_alpha = paper_robot.trust_alpha - original_robot.trust_alpha
                delta_beta = paper_robot.trust_beta - original_robot.trust_beta
                
                # Reconstruct PSM from deltas (if there was a change)
                if abs(delta_alpha) > 1e-6 or abs(delta_beta) > 1e-6:
                    # Total confidence is roughly delta_alpha + delta_beta
                    total_delta = delta_alpha + delta_beta
                    if total_delta > 1e-6:
                        psm_confidence = total_delta
                        psm_value = delta_alpha / total_delta  # What fraction went to alpha
                    else:
                        # Negative PSM - beta increased more than alpha
                        psm_confidence = max(abs(delta_alpha), abs(delta_beta))
                        psm_value = 0.1 if delta_alpha > delta_beta else 0.9
                else:
                    # No trust change - neutral PSM
                    psm_confidence = 0.1
                    psm_value = 0.5
                
                # Clamp values to reasonable ranges
                psm_confidence = np.clip(psm_confidence, 0.01, 2.0)
                psm_value = np.clip(psm_value, 0.0, 1.0)
                
                labels[f"agent_{robot_id}"] = {
                    'value': psm_value,
                    'confidence': psm_confidence,
                    'current_alpha': original_robot.trust_alpha,
                    'current_beta': original_robot.trust_beta,
                    'is_trustworthy': 0.0 if original_robot.is_adversarial else 1.0,
                    'paper_delta_alpha': delta_alpha,
                    'paper_delta_beta': delta_beta
                }
        
        # Track labels - extract actual trust changes from Paper Algorithm track processing
        
        # Create comprehensive track lookup from original states
        original_track_lookup = {}
        for track in tracks:
            original_track_lookup[track.id] = track
        
        # Also check tracks in robot_object_tracks (fused tracks)
        for robot_id, obj_tracks in original_robot_object_tracks.items():
            for obj_id, track in obj_tracks.items():
                original_track_lookup[track.id] = track
        
        # Create lookup for tracks after paper algorithm processing
        paper_track_lookup = {}
        
        # Get tracks from tracks_by_robot_after_paper
        for robot_id, tracks_list in tracks_by_robot_after_paper.items():
            for track in tracks_list:
                paper_track_lookup[track.id] = track
        
        # Get fused tracks from robot_object_tracks_after_paper  
        for robot_id, obj_tracks in robot_object_tracks_after_paper.items():
            for obj_id, track in obj_tracks.items():
                paper_track_lookup[track.id] = track
        
        # Generate track PSMs from actual paper algorithm trust changes
        for track in tracks:
            original_track = original_track_lookup.get(track.id, track)
            paper_track = paper_track_lookup.get(track.id)
            
            if paper_track and (abs(paper_track.trust_alpha - original_track.trust_alpha) > 1e-6 or 
                              abs(paper_track.trust_beta - original_track.trust_beta) > 1e-6):
                # Track trust was updated by paper algorithm - extract PSM
                delta_alpha = paper_track.trust_alpha - original_track.trust_alpha
                delta_beta = paper_track.trust_beta - original_track.trust_beta
                
                # Reconstruct PSM from trust deltas (similar to agent approach)
                if abs(delta_alpha) > 1e-6 or abs(delta_beta) > 1e-6:
                    total_delta = delta_alpha + delta_beta
                    if total_delta > 1e-6:
                        psm_confidence = total_delta
                        psm_value = delta_alpha / total_delta
                    else:
                        # Negative PSM or complex update
                        psm_confidence = max(abs(delta_alpha), abs(delta_beta))
                        psm_value = 0.2 if delta_alpha < delta_beta else 0.8
                else:
                    # No change
                    psm_confidence = 0.1
                    psm_value = 0.5
                
                # Clamp to reasonable ranges
                psm_confidence = np.clip(psm_confidence, 0.01, 3.0)
                psm_value = np.clip(psm_value, 0.0, 1.0)
                
                labels[f"track_{track.id}"] = {
                    'value': psm_value,
                    'confidence': psm_confidence,
                    'current_alpha': original_track.trust_alpha,
                    'current_beta': original_track.trust_beta,
                    'is_trustworthy': self._determine_track_ground_truth_trustworthy(original_track),
                    'paper_delta_alpha': delta_alpha,
                    'paper_delta_beta': delta_beta,
                    'used_paper_algorithm': True
                }
            else:
                # Track not updated by paper algorithm or no change - use heuristic
                track_trust = original_track.trust_alpha / (original_track.trust_alpha + original_track.trust_beta)
                source_robot = original_robot_lookup.get(track.source_robot)
                
                if source_robot:
                    source_trust = source_robot.trust_alpha / (source_robot.trust_alpha + source_robot.trust_beta)
                    is_source_trustworthy = source_trust > 0.5 and not source_robot.is_adversarial
                    
                    if is_source_trustworthy:
                        psm_value = 0.5 + 0.3 * track.confidence  
                        psm_confidence = track.confidence * 0.8
                    else:
                        psm_value = 0.3 + 0.2 * track.confidence  
                        psm_confidence = track.confidence * 1.2  # Negative bias
                else:
                    psm_value = 0.45 + 0.1 * track.confidence
                    psm_confidence = track.confidence * 0.6
                
                psm_confidence = np.clip(psm_confidence, 0.01, 2.0)
                psm_value = np.clip(psm_value, 0.0, 1.0)
                
                labels[f"track_{track.id}"] = {
                    'value': psm_value,
                    'confidence': psm_confidence,
                    'current_alpha': original_track.trust_alpha,
                    'current_beta': original_track.trust_beta,
                    'is_trustworthy': self._determine_track_ground_truth_trustworthy(original_track),
                    'used_paper_algorithm': False
                }
        
        return labels
    
    def _determine_track_ground_truth_trustworthy(self, track: Track) -> float:
        """
        Determine if a track is trustworthy based on ground truth information
        
        Ground truth trustworthiness:
        - 1.0: Track corresponds to a real ground truth object (object_id starts with 'gt_obj_')
        - 0.0: Track is a false positive detection (any other object_id pattern)
        
        This provides the actual ground truth label rather than inferring from trust parameters.
        """
        if track.object_id and isinstance(track.object_id, str):
            if track.object_id.startswith('gt_obj_'):
                # Track corresponds to real ground truth object
                return 1.0
            else:
                # Track is false positive or unknown - not trustworthy
                return 0.0
        else:
            # No object_id or invalid format - assume untrustworthy
            # This can happen for unassigned tracks or processing errors
            return 0.0
    
    def _generate_labels(self, robots: List[RobotState], tracks: List[Track], 
                        trust_updates: Dict[int, Dict]) -> Dict[str, Dict[str, float]]:
        """Generate training labels from paper algorithm PSMs"""
        labels = {}
        
        # Agent labels from trust updates
        for robot in robots:
            if robot.id in trust_updates:
                # Use the difference in trust as the label
                trust_before = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                
                # Simulate PSM value and confidence based on robot behavior
                if robot.is_adversarial:
                    # Adversarial robots should have negative PSM values
                    psm_value = np.random.uniform(-1.0, 0.2)
                    psm_confidence = np.random.uniform(0.3, 0.8)
                else:
                    # Legitimate robots should have positive PSM values
                    psm_value = np.random.uniform(0.2, 1.0)
                    psm_confidence = np.random.uniform(0.6, 0.9)
                
                labels[f"agent_{robot.id}"] = {
                    'value': psm_value,
                    'confidence': psm_confidence,
                    'current_alpha': robot.trust_alpha,
                    'current_beta': robot.trust_beta,
                    'is_trustworthy': 0.0 if robot.is_adversarial else 1.0
                }
        
        # Track labels
        for track in tracks:
            # Generate track PSM based on track characteristics
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            psm_value = np.tanh(track_trust * 2 - 1)  # Scale to [-1, 1]
            psm_confidence = track.confidence
            
            labels[f"track_{track.id}"] = {
                'value': psm_value,
                'confidence': psm_confidence,
                'current_alpha': track.trust_alpha,
                'current_beta': track.trust_beta,
                'is_trustworthy': 1.0 if track_trust > 0.5 else 0.0
            }
        
        return labels
    
    def save_training_data(self, filename: str) -> None:
        """Save collected training data"""
        with open(filename, 'wb') as f:
            pickle.dump(self.training_examples, f)
        print(f"Saved {len(self.training_examples)} training examples to {filename}")
    
    def load_training_data(self, filename: str) -> None:
        """Load training data from file"""
        with open(filename, 'rb') as f:
            self.training_examples = pickle.load(f)
        print(f"Loaded {len(self.training_examples)} training examples from {filename}")


class GNNTrainer:
    """Trainer for the GNN model with both supervised and RL capabilities"""
    
    def __init__(self, model: TrustGNN, learning_rate: float = 0.001, device: torch.device = torch.device('cpu')):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # Use KL divergence instead of MSE
        
    def train_supervised(self, training_examples: List[TrainingExample], 
                        epochs: int = 100, batch_size: int = 32) -> List[float]:
        """Train using supervised learning"""
        losses = []
        
        print(f"Starting training with {len(training_examples)} examples")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Simple batch training (could be improved with DataLoader)
            for i in range(0, len(training_examples), batch_size):
                batch = training_examples[i:i+batch_size]
                batch_loss = self._train_batch_supervised(batch)
                epoch_loss += batch_loss
                num_batches += 1
            
            avg_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_loss)
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
                
                # Debug: Check if model parameters are changing
                if epoch == 0:
                    self._first_params = [p.clone() for p in self.model.parameters()]
                elif epoch == 10:
                    param_changes = []
                    for i, p in enumerate(self.model.parameters()):
                        change = torch.mean(torch.abs(p - self._first_params[i])).item()
                        param_changes.append(change)
                    avg_change = sum(param_changes) / len(param_changes)
                    print(f"Average parameter change after 10 epochs: {avg_change:.6f}")
                    if avg_change < 1e-6:
                        print("WARNING: Model parameters barely changing - possible training issue!")
        
        return losses
    
    def _train_batch_supervised(self, batch: List[TrainingExample]) -> float:
        """Train a single batch in supervised mode using gradient accumulation"""
        
        # Enable debug for first batch
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        
        debug_first_batch = (self._batch_count == 0)
        
        if debug_first_batch:
            print(f"    Processing batch of {len(batch)} graphs with gradient accumulation")
        
        # Use gradient accumulation - simple and robust
        self.optimizer.zero_grad()
        total_loss = 0.0
        valid_examples = 0
        
        for i, example in enumerate(batch):
            try:
                # Use the graph data as-is - no complex filtering/copying
                graph_data = example.graph_data
                
                # Move data to device
                x_dict = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
                
                # Handle edges - create empty tensors for missing edge types
                edge_index_dict = {}
                required_edge_types = [
                    ('agent', 'observes', 'track'),
                    ('track', 'observed_by', 'agent'),
                    ('agent', 'in_fov', 'track'),
                    ('track', 'in_fov_by', 'agent')
                ]
                
                for edge_type in required_edge_types:
                    if edge_type in graph_data.edge_index_dict:
                        edge_index_dict[edge_type] = graph_data.edge_index_dict[edge_type].to(self.device)
                    else:
                        edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
                
                # Forward pass
                predictions = self.model(x_dict, edge_index_dict)
                
                # Debug first example
                if debug_first_batch and i == 0:
                    print(f"    Example {i}: Agents: {x_dict['agent'].shape[0] if 'agent' in x_dict else 0}, "
                          f"Tracks: {x_dict['track'].shape[0] if 'track' in x_dict else 0}")
                    for pred_type, pred_dict in predictions.items():
                        print(f"    {pred_type} predictions: value={pred_dict['value'].mean().item():.4f}, "
                              f"confidence={pred_dict['confidence'].mean().item():.4f}")
                
                # Compute loss
                loss = self._compute_supervised_loss(predictions, example.labels, graph_data)
                
                # Scale loss by batch size for proper gradient accumulation
                scaled_loss = loss / len(batch)
                scaled_loss.backward()
                
                total_loss += loss.item()
                valid_examples += 1
                
            except Exception as e:
                if debug_first_batch:
                    print(f"    Example {i} failed: {e}")
                continue
        
        # Update parameters after accumulating gradients from all examples
        if valid_examples > 0:
            self.optimizer.step()
            avg_loss = total_loss / valid_examples
        else:
            avg_loss = 0.0
        
        print(f"    Processed {valid_examples}/{len(batch)} examples, avg loss: {avg_loss:.6f}")
        
        self._batch_count += 1
        return avg_loss
    
    def _compute_supervised_loss(self, predictions: Dict, labels: Dict[str, Dict[str, float]], 
                                graph_data: HeteroData) -> torch.Tensor:
        """
        Compute supervised learning loss using simplified trust-based MSE loss
        
        This is much faster than Beta KL divergence while still capturing trust learning objectives:
        - Predict PSM values that move current trust toward ground truth
        - Use MSE between predicted and target trust values
        """
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        num_losses = 0
        
        # Agent losses - only compute if agent predictions exist
        if 'agent' in predictions:
            for node_id, label_dict in labels.items():
                if node_id.startswith('agent_'):
                    agent_id = int(node_id.split('_')[1])
                    if hasattr(graph_data, 'agent_nodes') and agent_id in graph_data.agent_nodes:
                        idx = graph_data.agent_nodes[agent_id]
                        
                        # Current trust parameters
                        current_alpha = label_dict.get('current_alpha', 1.0)
                        current_beta = label_dict.get('current_beta', 1.0)
                        current_trust = current_alpha / (current_alpha + current_beta)
                        
                        # Ground truth trustworthiness
                        is_trustworthy = label_dict.get('is_trustworthy', 1.0)
                        target_trust = 0.9 if is_trustworthy > 0.5 else 0.1
                        
                        # Get GNN predictions
                        psm_value = predictions['agent']['value'][idx].squeeze()
                        psm_confidence = predictions['agent']['confidence'][idx].squeeze()
                        
                        # Compute what the updated trust would be
                        # PSM update: delta_alpha = confidence * value, delta_beta = confidence * (1-value)
                        delta_alpha = psm_confidence * torch.clamp(psm_value, 0, 1)
                        delta_beta = psm_confidence * torch.clamp(1 - psm_value, 0, 1)
                        
                        updated_alpha = current_alpha + delta_alpha
                        updated_beta = current_beta + delta_beta
                        
                        updated_trust = updated_alpha / (updated_alpha + updated_beta)
                        
                        # Loss: Scaled MSE between updated trust and target trust
                        # Scale by 10 to improve gradient flow for trust values in [0,1]
                        trust_loss = 10.0 * (updated_trust - target_trust) ** 2
                        
                        # Additional loss to encourage appropriate confidence
                        # High confidence when we're certain, low when uncertain
                        uncertainty = abs(current_trust - 0.5)  # How far from uncertain (0.5)
                        target_confidence = 0.3 + 0.6 * uncertainty  # Scale confidence with certainty
                        confidence_loss = (psm_confidence - target_confidence) ** 2
                        
                        agent_loss = trust_loss + 0.1 * confidence_loss
                        total_loss = total_loss + agent_loss
                        num_losses += 1
        
        # Track losses - similar simplified structure
        if 'track' in predictions:
            for node_id, label_dict in labels.items():
                if node_id.startswith('track_'):
                    track_id = node_id.split('_', 1)[1]
                    if hasattr(graph_data, 'track_nodes') and track_id in graph_data.track_nodes:
                        idx = graph_data.track_nodes[track_id]
                        
                        # Current trust parameters
                        current_alpha = label_dict.get('current_alpha', 1.0)
                        current_beta = label_dict.get('current_beta', 1.0)
                        current_trust = current_alpha / (current_alpha + current_beta)
                        
                        # Ground truth trustworthiness
                        is_trustworthy = label_dict.get('is_trustworthy', 1.0)
                        target_trust = 0.8 if is_trustworthy > 0.5 else 0.2
                        
                        # Get GNN predictions
                        psm_value = predictions['track']['value'][idx].squeeze()
                        psm_confidence = predictions['track']['confidence'][idx].squeeze()
                        
                        # Compute updated trust
                        delta_alpha = psm_confidence * torch.clamp(psm_value, 0, 1)
                        delta_beta = psm_confidence * torch.clamp(1 - psm_value, 0, 1)
                        
                        updated_alpha = current_alpha + delta_alpha
                        updated_beta = current_beta + delta_beta
                        
                        updated_trust = updated_alpha / (updated_alpha + updated_beta)
                        
                        # Loss: Scaled MSE between updated trust and target trust
                        # Scale by 10 to improve gradient flow for trust values in [0,1]
                        trust_loss = 10.0 * (updated_trust - target_trust) ** 2
                        
                        # Confidence loss
                        uncertainty = abs(current_trust - 0.5)
                        target_confidence = 0.2 + 0.5 * uncertainty
                        confidence_loss = (psm_confidence - target_confidence) ** 2
                        
                        track_loss = trust_loss + 0.1 * confidence_loss
                        total_loss = total_loss + track_loss
                        num_losses += 1
        
        # Average the loss
        if num_losses > 0:
            avg_loss = total_loss / num_losses
            # Debug first batch to check if loss computation is working
            if hasattr(self, '_debug_loss') and self._debug_loss:
                print(f"    Loss computation: {num_losses} nodes, avg loss: {avg_loss.item():.6f}")
                self._debug_loss = False
            return avg_loss
        else:
            if hasattr(self, '_debug_loss') and self._debug_loss:
                print("    WARNING: No valid loss computed - no matching nodes!")
                self._debug_loss = False
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _beta_kl_divergence(self, alpha1: torch.Tensor, beta1: torch.Tensor, 
                           alpha2: torch.Tensor, beta2: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between two Beta distributions
        KL(Beta(α₁,β₁) || Beta(α₂,β₂))
        """
        # Use the digamma function and log-beta function
        from torch.special import digamma, gammaln
        
        # Log-beta function: ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
        def log_beta(alpha, beta):
            return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        
        # KL divergence formula
        kl = (log_beta(alpha2, beta2) - log_beta(alpha1, beta1) +
              (alpha1 - alpha2) * digamma(alpha1) +
              (beta1 - beta2) * digamma(beta1) +
              (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1))
        
        # Ensure non-negative (numerical stability)
        return torch.clamp(kl, min=0.0)
    


class NeuralSymbolicTrustAlgorithm(TrustAlgorithm):
    """Neural Symbolic Trust Algorithm using Graph Neural Networks"""
    
    def __init__(self, model_path: Optional[str] = None, learning_mode: bool = True):
        """
        Initialize the neural symbolic trust algorithm
        
        Args:
            model_path: Path to pre-trained model (if available)
            learning_mode: Whether to collect training data during simulation
        """
        self.learning_mode = learning_mode
        self.graph_builder = GraphBuilder()
        self.training_data_collector = TrainingDataCollector() if learning_mode else None
        
        # Initialize GNN model
        self.model = TrustGNN(agent_features=1, track_features=1)  # Based on simplified trustworthy predicate features
        self.trainer = GNNTrainer(self.model, device=next(self.model.parameters()).device) if learning_mode else None
        
        # Load pre-trained model if available
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Loaded pre-trained model from {model_path}")
            except FileNotFoundError:
                print(f"Model file {model_path} not found, using randomly initialized model")
        
        self.model.eval() if not learning_mode else self.model.train()
        
        # Robot object trust tracking  
        self.robot_object_trust: Dict[int, Dict[str, Tuple[float, float]]] = {}
        
    def initialize(self, robots: List[RobotState]) -> None:
        """Initialize the neural symbolic algorithm with robot states"""
        # Initialize trust tracking
        for robot in robots:
            self.robot_object_trust[robot.id] = {}
        
        print("Neural Symbolic Trust Algorithm initialized with GNN")
    
    def update_trust(self, robots: List[RobotState], tracks_by_robot: Dict[int, List[Track]], 
                    robot_object_tracks: Dict[int, Dict[str, Track]], simulation_time: float,
                    robot_current_tracks: Optional[Dict[int, Dict[str, Track]]] = None,
                    environment: Optional['SimulationEnvironment'] = None) -> Dict[int, Dict]:
        """Update trust using GNN-based neural symbolic reasoning"""
        
        trust_updates = {}
        
        # Track all fused tracks for trust propagation
        all_fused_tracks = []
        
        # Main loop: iterate over each robot as ego robot
        for ego_robot in robots:
            # Always prefer current timestep tracks for graph input if available
            if robot_current_tracks:
                # Get current tracks for ego robot
                ego_current_tracks_dict = robot_current_tracks.get(ego_robot.id, {})
                ego_fused_tracks = list(ego_current_tracks_dict.values())
                
                # Ensure current tracks inherit trust values from accumulated tracks
                for track_id, current_track in ego_current_tracks_dict.items():
                    if (ego_robot.id in robot_object_tracks and 
                        track_id in robot_object_tracks[ego_robot.id]):
                        accumulated_track = robot_object_tracks[ego_robot.id][track_id]
                        # Inherit trust values from accumulated track
                        current_track.trust_alpha = accumulated_track.trust_alpha
                        current_track.trust_beta = accumulated_track.trust_beta
                
                # Build proximal tracks dictionary from current timestep
                proximal_tracks_by_robot = {}
                for r_id, current_tracks_dict in robot_current_tracks.items():
                    if r_id != ego_robot.id:
                        proximal_tracks_list = list(current_tracks_dict.values())
                        # Ensure proximal tracks inherit trust from accumulated tracks
                        for track in proximal_tracks_list:
                            if (r_id in robot_object_tracks and 
                                hasattr(track, 'object_id') and
                                track.object_id in robot_object_tracks[r_id]):
                                accumulated_track = robot_object_tracks[r_id][track.object_id]
                                track.trust_alpha = accumulated_track.trust_alpha
                                track.trust_beta = accumulated_track.trust_beta
                        proximal_tracks_by_robot[r_id] = proximal_tracks_list
            else:
                # Fallback: use all accumulated tracks
                ego_fused_tracks = list(robot_object_tracks.get(ego_robot.id, {}).values())
                proximal_tracks_by_robot = {r_id: tracks for r_id, tracks in tracks_by_robot.items() 
                                          if r_id != ego_robot.id}
            
            if not ego_fused_tracks:
                continue
            
            # Get proximal robots within range
            if environment:
                proximal_robots = environment.get_proximal_robots(ego_robot)
                # Filter proximal tracks to only include robots in range
                proximal_robot_ids = {r.id for r in proximal_robots}
                proximal_tracks_by_robot = {r_id: tracks for r_id, tracks in proximal_tracks_by_robot.items() 
                                          if r_id in proximal_robot_ids}
            else:
                # Fallback: consider all other robots as proximal
                proximal_robots = [r for r in robots if r.id != ego_robot.id]
            
            # Perform track fusion to reduce redundant proximal tracks
            # Always use accumulated tracks for fusion logic (contains complete track history)
            fused_tracks = self._fuse_tracks_for_training(
                ego_robot, ego_fused_tracks, proximal_robots, robot_object_tracks
            )
            
            # Collect fused tracks that have source track mappings for propagation
            for track in fused_tracks:
                if hasattr(track, '_source_tracks'):
                    all_fused_tracks.append(track)
            
            # Build graph with neural symbolic predicates using fused tracks
            graph_data = self.graph_builder.build_graph(
                ego_robot, fused_tracks, proximal_robots, proximal_tracks_by_robot
            )
            
            # Apply GNN to get PSM predictions
            if graph_data['agent'].x.shape[0] > 0:  # Check if we have nodes
                with torch.no_grad() if not self.learning_mode else torch.enable_grad():
                    # Move data to device if needed
                    x_dict = {k: v.to(next(self.model.parameters()).device) for k, v in graph_data.x_dict.items()}
                    edge_index_dict = {}
                    try:
                        # Safely get edge_index_dict if it exists
                        if graph_data.edge_index_dict:
                            edge_index_dict = {k: v.to(next(self.model.parameters()).device) 
                                             for k, v in graph_data.edge_index_dict.items()}
                    except (AttributeError, KeyError):
                        # No edges exist, use empty dict
                        edge_index_dict = {}
                    
                    predictions = self.model(x_dict, edge_index_dict)
                
                # Convert GNN predictions to PSMs and update trust
                # Build the robot list that matches the graph node order
                all_graph_robots = [ego_robot] + proximal_robots
                self._apply_gnn_predictions(predictions, graph_data, all_graph_robots, ego_fused_tracks)
        
        # Generate trust updates for all robots
        for robot in robots:
            trust_updates[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta,
                'mean_trust': robot.trust_alpha / (robot.trust_alpha + robot.trust_beta),
                'algorithm': 'neural_symbolic_gnn'
            }
        
        # Propagate trust updates from fused tracks back to their source tracks
        if all_fused_tracks:
            self._propagate_trust_updates_to_source_tracks(all_fused_tracks, robot_object_tracks)
        
        # Collect training data if in learning mode
        if self.learning_mode and self.training_data_collector:
            # Use current timestep tracks for training data collection if available
            if robot_current_tracks:
                # Convert current tracks to list format for collect_from_simulation_step
                current_tracks_by_robot = {}
                for r_id, current_tracks_dict in robot_current_tracks.items():
                    current_tracks_by_robot[r_id] = list(current_tracks_dict.values())
                
                # Collect training data for all robots that have current tracks
                for robot in robots:
                    ego_current_tracks = list(robot_current_tracks.get(robot.id, {}).values())
                    if ego_current_tracks:  # Only collect if we have current tracks
                        self.training_data_collector.collect_from_simulation_step(
                            robots, current_tracks_by_robot, robot_current_tracks, trust_updates, 
                            robot, ego_current_tracks
                        )
            else:
                # Fallback to accumulated tracks if current tracks not available
                for robot in robots:
                    ego_fused_tracks = list(robot_object_tracks.get(robot.id, {}).values())
                    if ego_fused_tracks:  # Only collect if we have tracks
                        self.training_data_collector.collect_from_simulation_step(
                            robots, tracks_by_robot, robot_object_tracks, trust_updates, 
                            robot, ego_fused_tracks
                        )
        
        return trust_updates
    
    def _apply_gnn_predictions(self, predictions: Dict, graph_data: HeteroData, 
                              robots: List[RobotState], tracks: List[Track]) -> None:
        """Apply GNN predictions as PSMs to update trust distributions"""
        
        # Update agent trust based on GNN predictions
        if 'agent' in predictions and hasattr(graph_data, 'agent_nodes'):
            for robot in robots:
                if robot.id in graph_data.agent_nodes:
                    idx = graph_data.agent_nodes[robot.id]
                    if idx < predictions['agent']['value'].shape[0]:
                        psm_value = torch.clamp(predictions['agent']['value'][idx], 0, 1).item()
                        psm_confidence = torch.clamp(predictions['agent']['confidence'][idx], 0, 5).item()
                        
                        # Apply PSM using proper Beta-Bernoulli update
                        # PSM value is probability of positive evidence
                        delta_alpha = psm_confidence * psm_value
                        delta_beta = psm_confidence * (1 - psm_value)
                        
                        robot.trust_alpha += delta_alpha
                        robot.trust_beta += delta_beta
                        
                        # Ensure minimum values
                        robot.trust_alpha = max(0.1, robot.trust_alpha)
                        robot.trust_beta = max(0.1, robot.trust_beta)
        
        # Update track trust based on GNN predictions
        if 'track' in predictions and hasattr(graph_data, 'track_nodes'):
            for track in tracks:
                if track.id in graph_data.track_nodes:
                    idx = graph_data.track_nodes[track.id]
                    if idx < predictions['track']['value'].shape[0]:
                        psm_value = torch.clamp(predictions['track']['value'][idx], 0, 1).item()
                        psm_confidence = torch.clamp(predictions['track']['confidence'][idx], 0, 5).item()
                        
                        # Apply PSM to track trust
                        delta_alpha = psm_confidence * psm_value
                        delta_beta = psm_confidence * (1 - psm_value)
                        
                        track.trust_alpha += delta_alpha
                        track.trust_beta += delta_beta
                        
                        # Ensure minimum values
                        track.trust_alpha = max(0.1, track.trust_alpha)
                        track.trust_beta = max(0.1, track.trust_beta)
    
    def train_model(self, training_data_file: str, epochs: int = 100) -> List[float]:
        """Train the GNN model from collected data"""
        if not self.training_data_collector:
            print("Training data collector not available. Enable learning_mode=True")
            return []
        
        # Load training data
        self.training_data_collector.load_training_data(training_data_file)
        
        # Train the model
        losses = self.trainer.train_supervised(
            self.training_data_collector.training_examples,
            epochs=epochs
        )
        
        print(f"Training completed. Final loss: {losses[-1]:.4f}")
        return losses
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model"""
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_data(self, filename: str) -> None:
        """Save collected training data"""
        if self.training_data_collector:
            self.training_data_collector.save_training_data(filename)
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return alpha / (alpha + beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        denominator = (alpha + beta) ** 2 * (alpha + beta + 1)
        return (alpha * beta) / denominator
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state"""
        return {
            'model_type': 'GNN_Neural_Symbolic',
            'learning_mode': self.learning_mode,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_examples': len(self.training_data_collector.training_examples) if self.training_data_collector else 0
        }
    
    def _fuse_tracks_for_training(self, ego_robot: RobotState, ego_fused_tracks: List[Track],
                                 proximal_robots: List[RobotState], 
                                 robot_object_tracks: Dict[int, Dict[str, Track]]) -> List[Track]:
        """
        Fuse tracks from ego robot and all its proximal robots.
        All tracks with same object_id and within distance threshold are fused.
        Trust distributions are updated for raw source tracks.
        """
        from collections import defaultdict
        
        fusion_threshold = 5.0  # Distance threshold for track fusion
        
        # Track all observations from ego + proximal robots by object_id
        object_observations = defaultdict(list)  # object_id -> [(robot_id, track)]
        
        # 1. Collect ego robot tracks  
        for track in ego_fused_tracks:
            if hasattr(track, 'object_id') and track.object_id:
                object_observations[track.object_id].append((ego_robot.id, track))
        
        # 2. Collect proximal robot tracks
        for robot in proximal_robots:
            robot_tracks = robot_object_tracks.get(robot.id, {})
            for track in robot_tracks.values():
                if hasattr(track, 'object_id') and track.object_id:
                    object_observations[track.object_id].append((robot.id, track))
        
        # 3. Process each object: fuse tracks within distance threshold
        final_tracks = []
        
        for object_id, observations in object_observations.items():
            if not observations:
                continue
                
            # Group observations by spatial proximity
            track_groups = []
            
            for robot_id, track in observations:
                # Find existing group within fusion threshold
                assigned = False
                for group in track_groups:
                    # Check distance to any track in the group
                    for _, existing_track in group:
                        distance = np.linalg.norm(track.position - existing_track.position)
                        if distance <= fusion_threshold:
                            group.append((robot_id, track))
                            assigned = True
                            break
                    if assigned:
                        break
                
                if not assigned:
                    # Create new group
                    track_groups.append([(robot_id, track)])
            
            # Create fused tracks for each group
            for group_idx, group in enumerate(track_groups):
                if len(group) == 1:
                    # Single track - use as is
                    robot_id, track = group[0]
                    fused_track = Track(
                        id=f"{ego_robot.id}_{object_id}_{group_idx}",
                        position=track.position.copy(),
                        velocity=track.velocity.copy(), 
                        covariance=track.covariance.copy(),
                        confidence=track.confidence,
                        timestamp=track.timestamp,
                        source_robot=robot_id,
                        trust_alpha=track.trust_alpha,
                        trust_beta=track.trust_beta,
                        object_id=object_id
                    )
                else:
                    # Multiple tracks - fuse them
                    # Weighted average by confidence
                    total_confidence = sum(track.confidence for _, track in group)
                    if total_confidence > 0:
                        fused_position = np.zeros(3)
                        fused_velocity = np.zeros(3)
                        for _, track in group:
                            weight = track.confidence / total_confidence
                            fused_position += weight * track.position
                            fused_velocity += weight * track.velocity
                    else:
                        # Fallback to simple average
                        fused_position = np.mean([track.position for _, track in group], axis=0)
                        fused_velocity = np.mean([track.velocity for _, track in group], axis=0)
                    
                    # Fuse trust distributions (sum of alphas and betas)
                    fused_alpha = sum(track.trust_alpha for _, track in group)
                    fused_beta = sum(track.trust_beta for _, track in group)
                    
                    # Average confidence and latest timestamp
                    avg_confidence = np.mean([track.confidence for _, track in group])
                    latest_timestamp = max(track.timestamp for _, track in group)
                    
                    # Average covariance
                    avg_covariance = np.mean([track.covariance for _, track in group], axis=0)
                    
                    fused_track = Track(
                        id=f"{ego_robot.id}_{object_id}_{group_idx}_fused",
                        position=fused_position,
                        velocity=fused_velocity,
                        covariance=avg_covariance,
                        confidence=avg_confidence,
                        timestamp=latest_timestamp,
                        source_robot=ego_robot.id,
                        trust_alpha=fused_alpha,
                        trust_beta=fused_beta,
                        object_id=object_id
                    )
                
                # Store source tracks for trust propagation back to raw tracks
                setattr(fused_track, '_source_tracks', group)
                final_tracks.append(fused_track)
        
        return final_tracks