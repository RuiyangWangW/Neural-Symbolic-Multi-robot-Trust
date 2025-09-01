#!/usr/bin/env python3
"""
Visualize the graph structure that is given as input to the GNN
Similar to visualize_training_data.py but focused on the current RL state
"""

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from typing import Dict, Any
from datetime import datetime

def visualize_gnn_input(graph_data, episode=None, timestep=None, save_dir="gnn_input_visualizations", current_state=None):
    """
    Visualize the graph structure given as input to the GNN
    
    Args:
        graph_data: The heterogeneous graph data (PyTorch Geometric HeteroData)
        episode: Current episode number (optional)
        timestep: Current timestep (optional)  
        save_dir: Directory to save the visualization
    """
    
    print(f"🎯 GNN INPUT GRAPH VISUALIZATION")
    print(f"=" * 50)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create NetworkX graph for visualization
    G = create_networkx_graph(graph_data)
    
    # Create and save visualization
    save_gnn_input_visualization(G, graph_data, episode, timestep, save_dir, current_state)
    
    return G

def create_networkx_graph(graph_data):
    """Create NetworkX graph from HeteroData for visualization"""
    G = nx.MultiGraph()  # Use MultiGraph to handle multiple edge types
    
    # Add agent nodes
    if hasattr(graph_data, 'agent_nodes'):
        for agent_id, idx in graph_data.agent_nodes.items():
            G.add_node(f"agent_{agent_id}", type='agent', idx=idx, agent_id=agent_id)
    
    # Add track nodes  
    if hasattr(graph_data, 'track_nodes'):
        for track_id, idx in graph_data.track_nodes.items():
            G.add_node(f"track_{track_id}", type='track', idx=idx, track_id=track_id)
    
    # Add edges based on hetero_data
    hetero_data = graph_data
    if hasattr(hetero_data, 'edge_types'):
        edge_types = list(hetero_data.edge_types)
        
        for edge_type in edge_types:
            edge_index = hetero_data[edge_type].edge_index
            src_type, _, dst_type = edge_type
            
            # Get node mappings
            if src_type == 'agent' and hasattr(graph_data, 'agent_nodes'):
                src_nodes = {idx: f"agent_{agent_id}" for agent_id, idx in graph_data.agent_nodes.items()}
            elif src_type == 'track' and hasattr(graph_data, 'track_nodes'):
                src_nodes = {idx: f"track_{track_id}" for track_id, idx in graph_data.track_nodes.items()}
            else:
                src_nodes = {}
                
            if dst_type == 'agent' and hasattr(graph_data, 'agent_nodes'):
                dst_nodes = {idx: f"agent_{agent_id}" for agent_id, idx in graph_data.agent_nodes.items()}
            elif dst_type == 'track' and hasattr(graph_data, 'track_nodes'):
                dst_nodes = {idx: f"track_{track_id}" for track_id, idx in graph_data.track_nodes.items()}
            else:
                dst_nodes = {}
            
            # Add edges
            for i in range(edge_index.shape[1]):
                src_idx = edge_index[0, i].item()
                dst_idx = edge_index[1, i].item()
                if src_idx in src_nodes and dst_idx in dst_nodes:
                    G.add_edge(src_nodes[src_idx], dst_nodes[dst_idx], type=edge_type)
    
    return G

def create_gnn_layout(G):
    """Create layout optimized for GNN visualization"""
    pos = {}
    
    # Agent nodes in outer circle
    agent_nodes = [n for n in G.nodes() if n.startswith('agent_')]
    for i, node in enumerate(agent_nodes):
        angle = 2 * np.pi * i / max(len(agent_nodes), 1)
        pos[node] = (3.5 * np.cos(angle), 3.5 * np.sin(angle))
    
    # Track nodes in inner area
    track_nodes = [n for n in G.nodes() if n.startswith('track_')]
    
    # Categorize track nodes by type for better layout
    fused_tracks = []
    fp_tracks = []
    proximal_tracks = []
    ego_tracks = []
    current_tracks = []  # Renamed from rl_tracks
    
    for n in track_nodes:
        # Extract track_id from node name (remove "track_" prefix)
        track_id = n[6:] if n.startswith('track_') else n
        
        # Priority order: fused > fp > proximal > current > ego
        if 'fused' in track_id:
            fused_tracks.append(n)
        elif '_fp_' in track_id:
            fp_tracks.append(n)
        elif '_proximal_' in track_id:
            proximal_tracks.append(n)
        elif track_id.startswith('current_robot_'):
            current_tracks.append(n)
        elif track_id.startswith('gt_robot_'):
            current_tracks.append(n)  # Ground truth tracks are current tracks
        else:
            ego_tracks.append(n)
    
    # Position different track types in concentric circles
    # Fused tracks in center
    for i, node in enumerate(fused_tracks):
        if len(fused_tracks) == 1:
            pos[node] = (0, 0)
        else:
            angle = 2 * np.pi * i / len(fused_tracks)
            pos[node] = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
    
    # Current tracks in inner ring (most common in RL training)
    for i, node in enumerate(current_tracks):
        angle = 2 * np.pi * i / max(len(current_tracks), 1)
        pos[node] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
    
    # Ego tracks in middle ring
    for i, node in enumerate(ego_tracks):
        angle = 2 * np.pi * i / max(len(ego_tracks), 1)
        pos[node] = (2.0 * np.cos(angle), 2.0 * np.sin(angle))
    
    # Proximal tracks in outer track ring
    for i, node in enumerate(proximal_tracks):
        angle = 2 * np.pi * i / max(len(proximal_tracks), 1)
        pos[node] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))
    
    # False positive tracks in outermost track ring
    for i, node in enumerate(fp_tracks):
        angle = 2 * np.pi * i / max(len(fp_tracks), 1)
        pos[node] = (2.8 * np.cos(angle), 2.8 * np.sin(angle))
    
    return pos

def draw_gnn_edges(G, pos):
    """Draw edges with different styles based on relationship type"""
    from matplotlib.lines import Line2D
    
    # Separate agent-track and agent-agent edges
    agent_track_relationships = {}  # (agent, track) -> set of edge types
    proximal_edges = []  # Agent-agent proximal edges
    
    for u, v, key, data in G.edges(data=True, keys=True):
        edge_type = data.get('type')
        if edge_type:
            # Check if this is a robot-robot proximal edge
            if u.startswith('agent_') and v.startswith('agent_') and 'isProximal' in str(edge_type):
                proximal_edges.append((u, v))
            else:
                # Handle agent-track relationships
                if u.startswith('agent_') and v.startswith('track_'):
                    agent_node, track_node = u, v
                elif u.startswith('track_') and v.startswith('agent_'):
                    agent_node, track_node = v, u
                else:
                    continue  # Skip if not agent-track relationship
                    
                pair_key = (agent_node, track_node)
                if pair_key not in agent_track_relationships:
                    agent_track_relationships[pair_key] = set()
                agent_track_relationships[pair_key].add(edge_type)
    
    # Categorize agent-track relationships based on updated edge semantics
    in_fov_only = []           # InFoV only (robot can see but doesn't observe)
    in_fov_and_observed = []   # InFoV and observed (robot actively observes within FoV)
    
    for (agent, track), edge_types in agent_track_relationships.items():
        # Check for updated edge types
        has_in_fov_only = any('in_fov_only' in str(et) for et in edge_types)
        has_in_fov_and_observed = any('in_fov_and_observed' in str(et) for et in edge_types)
        
        if has_in_fov_and_observed:
            in_fov_and_observed.append((agent, track))
        elif has_in_fov_only:
            in_fov_only.append((agent, track))
    
    # Draw edges with different styles
    legend_elements = []
    
    # Robot-robot proximal edges - orange, thick dotted
    if proximal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=proximal_edges, edge_color='orange', 
                              alpha=0.7, width=2.5, style='dotted')
        legend_elements.append(Line2D([0], [0], color='orange', lw=2.5, linestyle=':', 
                                     label=f'Robot Proximal ({len(proximal_edges)})'))
    
    # InFoV only edges - blue, dashed
    if in_fov_only:
        nx.draw_networkx_edges(G, pos, edgelist=in_fov_only, edge_color='blue', 
                              alpha=0.6, width=1.5, style='dashed')
        legend_elements.append(Line2D([0], [0], color='blue', lw=1.5, linestyle='--', 
                                     label=f'InFoV Only ({len(in_fov_only)})'))
    
    
    # InFoV and Observed edges - red, thick solid (robot actively observes within FoV)
    if in_fov_and_observed:
        nx.draw_networkx_edges(G, pos, edgelist=in_fov_and_observed, edge_color='red', 
                              alpha=0.9, width=3.0, style='solid')
        legend_elements.append(Line2D([0], [0], color='red', lw=3.0, 
                                     label=f'InFoV & Observed ({len(in_fov_and_observed)})'))
    
    
    return legend_elements

def create_gnn_node_labels(graph_data, current_state=None):
    """Create node labels for GNN visualization including actual trust values"""
    node_labels = {}
    
    # Agent labels with actual trust values
    if hasattr(graph_data, 'agent_nodes'):
        for agent_id, agent_idx in graph_data.agent_nodes.items():
            # Try to get actual trust values from current_state
            trust_value = None
            if current_state and hasattr(current_state, '_current_robots'):
                for robot in current_state._current_robots:
                    if robot.id == agent_id and hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                        trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                        break
            
            # If we have actual trust value, show it; otherwise fall back to binary predicate
            if trust_value is not None:
                node_labels[f"agent_{agent_id}"] = f"A{agent_id}\n({trust_value:.3f})"
            else:
                # Fallback to binary predicate from features
                if 'agent' in graph_data.node_types and agent_idx < graph_data['agent'].x.shape[0]:
                    trustworthy_pred = graph_data['agent'].x[agent_idx, 0].item()
                    trust_status = "T" if trustworthy_pred > 0.5 else "U"
                    node_labels[f"agent_{agent_id}"] = f"A{agent_id}\n({trust_status})"
                else:
                    node_labels[f"agent_{agent_id}"] = f"A{agent_id}"
    
    # Track labels with trust values (simplified for readability)
    if hasattr(graph_data, 'track_nodes') and 'track' in graph_data.node_types:
        track_count = 0
        track_features = graph_data['track'].x
        for track_id, track_idx in graph_data.track_nodes.items():
            if track_count >= 15:  # Limit labels for readability
                break
            
            # Determine if this is a ground truth or false positive track based on object_id
            # We need to get the track object to check its object_id
            track_obj = None
            if current_state:
                # Look for the track in _fused_tracks and _individual_tracks
                all_available_tracks = []
                if hasattr(current_state, '_fused_tracks'):
                    all_available_tracks.extend(current_state._fused_tracks)
                if hasattr(current_state, '_individual_tracks'):
                    all_available_tracks.extend(current_state._individual_tracks)
                
                # Find the track by track_id
                for track in all_available_tracks:
                    if track.track_id == track_id:
                        track_obj = track
                        break
            
            if track_obj and hasattr(track_obj, 'object_id'):
                # Check if it's a ground truth or false positive based on object_id
                object_id_str = str(track_obj.object_id)
                
                if 'gt_obj_' in object_id_str:
                    # Ground truth track - extract object number
                    obj_num = object_id_str.replace('gt_obj_', '')
                    short_name = f"GT_{obj_num}"
                elif 'fp_obj_' in object_id_str or object_id_str.startswith('fp_'):
                    # False positive track - extract object number  
                    if 'fp_obj_' in object_id_str:
                        obj_num = object_id_str.replace('fp_obj_', '')
                    else:
                        obj_num = object_id_str.replace('fp_', '')
                    short_name = f"FP_{obj_num}"
                else:
                    # Default fallback - check track_id patterns for classification
                    if 'gt' in track_id.lower():
                        obj_num = track_id.split('_')[-1]
                        short_name = f"GT_{obj_num}"
                    else:  # Must be false positive
                        obj_num = track_id.split('_')[-1]
                        short_name = f"FP_{obj_num}"
            else:
                # Fallback: Use track_id patterns to determine type
                if 'gt' in track_id.lower() or 'ground' in track_id.lower():
                    obj_num = track_id.split('_')[-1]
                    short_name = f"GT_{obj_num}"
                else:  # Must be false positive
                    obj_num = track_id.split('_')[-1]
                    short_name = f"FP_{obj_num}"
            
            # Add actual trust value to label
            trust_value = None
            if current_state and hasattr(current_state, '_current_tracks'):
                for track in current_state._current_tracks:
                    if track.id == track_id and hasattr(track, 'trust_alpha') and hasattr(track, 'trust_beta'):
                        trust_value = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                        break
            
            # If we have actual trust value, show it; otherwise fall back to binary predicate
            if trust_value is not None:
                node_labels[f"track_{track_id}"] = f"{short_name}\n({trust_value:.3f})"
            else:
                # Fallback to binary predicate from features
                if track_idx < track_features.shape[0]:
                    trustworthy_pred = track_features[track_idx, 0].item()
                    trust_status = "T" if trustworthy_pred > 0.5 else "U"
                    node_labels[f"track_{track_id}"] = f"{short_name}\n({trust_status})"
                else:
                    node_labels[f"track_{track_id}"] = short_name
                
            track_count += 1
    
    return node_labels

def save_gnn_input_visualization(G, graph_data, episode, timestep, save_dir, current_state=None):
    """Create and save the GNN input graph visualization"""
    plt.figure(figsize=(14, 10))
    
    # Create layout
    pos = create_gnn_layout(G)
    
    # Separate nodes by type
    agent_nodes = [n for n in G.nodes() if n.startswith('agent_')]
    track_nodes = [n for n in G.nodes() if n.startswith('track_')]
    
    # Identify ego robot (typically agent 0 or first agent)
    ego_agent = None
    proximal_agents = []
    
    for node in agent_nodes:
        try:
            agent_id = int(node.split('_')[1])
            if agent_id == 0:
                ego_agent = node
            else:
                proximal_agents.append(node)
        except (ValueError, IndexError):
            proximal_agents.append(node)
    
    if ego_agent is None and agent_nodes:
        ego_agent = agent_nodes[0]
        proximal_agents = agent_nodes[1:]
    
    # Categorize track nodes by type and ownership
    fused_tracks = []
    fp_tracks = []  # False positive tracks (red)
    gt_tracks = []  # Ground truth tracks (green)
    proximal_tracks = []
    ego_tracks = []
    current_tracks = []
    
    # Identify ego robot ID (assume first agent or agent_0)
    ego_robot_id = None
    if agent_nodes:
        # Try to find agent 0 first
        for node in agent_nodes:
            if node == 'agent_0':
                ego_robot_id = '0'
                break
        # If no agent_0, use first agent
        if ego_robot_id is None:
            try:
                ego_robot_id = agent_nodes[0].split('_')[1]
            except (IndexError, ValueError):
                ego_robot_id = '0'  # Default fallback
    
    for n in track_nodes:
        # Extract track_id from node name (remove "track_" prefix)
        track_id = n[6:] if n.startswith('track_') else n
        
        if 'fused' in track_id:
            fused_tracks.append(n)
        elif '_fp_' in track_id or 'fp_' in track_id:
            fp_tracks.append(n)
        elif '_proximal_' in track_id:
            proximal_tracks.append(n)
        elif track_id.startswith('current_robot_') or track_id.startswith('gt_robot_'):
            # Check if this is a ground truth track or false positive
            if '_fp_' in track_id or 'fp_' in track_id:
                fp_tracks.append(n)
            elif 'gt_obj' in track_id or track_id.startswith('gt_robot_'):
                # This is a ground truth track
                gt_tracks.append(n)
            else:
                # Check if this is ego robot's track or proximal robot's track
                parts = track_id.split('_')
                robot_id = parts[2] if len(parts) > 2 else None
                
                if robot_id == ego_robot_id:
                    ego_tracks.append(n)
                else:
                    current_tracks.append(n)  # Proximal robot tracks
        else:
            current_tracks.append(n)  # Other tracks
    
    # Draw nodes with different colors and shapes
    # Ego robot - dark blue, large circle
    if ego_agent:
        nx.draw_networkx_nodes(G, pos, nodelist=[ego_agent], node_color='darkblue', 
                              node_size=1500, alpha=0.9, node_shape='o')
    
    # Proximal robots - light blue circles
    if proximal_agents:
        nx.draw_networkx_nodes(G, pos, nodelist=proximal_agents, node_color='lightblue', 
                              node_size=1200, alpha=0.8, node_shape='o')
    
    # Track nodes with different colors (squares for tracks)
    if fused_tracks:
        nx.draw_networkx_nodes(G, pos, nodelist=fused_tracks, node_color='gold',
                              node_size=900, alpha=0.8, node_shape='s')
    if gt_tracks:
        # Ground truth tracks - bright green
        nx.draw_networkx_nodes(G, pos, nodelist=gt_tracks, node_color='limegreen',
                              node_size=800, alpha=0.9, node_shape='s')
    if current_tracks:
        nx.draw_networkx_nodes(G, pos, nodelist=current_tracks, node_color='lightgreen',
                              node_size=800, alpha=0.8, node_shape='s')
    if ego_tracks:
        nx.draw_networkx_nodes(G, pos, nodelist=ego_tracks, node_color='lightcyan',
                              node_size=750, alpha=0.8, node_shape='s')
    if proximal_tracks:
        nx.draw_networkx_nodes(G, pos, nodelist=proximal_tracks, node_color='lightblue',
                              node_size=750, alpha=0.8, node_shape='s')
    if fp_tracks:
        # False positive tracks - bright red
        nx.draw_networkx_nodes(G, pos, nodelist=fp_tracks, node_color='red',
                              node_size=700, alpha=0.9, node_shape='s')
    
    # Draw edges
    legend_elements = draw_gnn_edges(G, pos)
    
    # Add labels
    node_labels = create_gnn_node_labels(graph_data, current_state)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    # Create legend
    from matplotlib.lines import Line2D
    node_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=12, 
               label='Ego Robot', linewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, 
               label='Proximal Robots', linewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=10, 
               label=f'Fused Tracks ({len(fused_tracks)})', linewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='limegreen', markersize=9, 
               label=f'Ground Truth Tracks ({len(gt_tracks)})', linewidth=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, 
               label=f'False Positive Tracks ({len(fp_tracks)})', linewidth=0)
    ]
    
    all_legend_elements = node_legend_elements + legend_elements
    
    # Title
    title = "GNN Input Graph Structure"
    if episode is not None and timestep is not None:
        title += f" - Episode {episode}, Step {timestep}"
    elif episode is not None:
        title += f" - Episode {episode}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(handles=all_legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.axis('equal')
    
    # Add summary text with actual trust analysis
    trustworthy_agents = 0
    untrustworthy_agents = 0
    trustworthy_tracks = 0
    untrustworthy_tracks = 0
    agent_trust_values = []
    track_trust_values = []
    
    # Analyze agent trust status using actual trust values when available
    if hasattr(graph_data, 'agent_nodes'):
        for agent_id, agent_idx in graph_data.agent_nodes.items():
            trust_value = None
            if current_state and hasattr(current_state, '_current_robots'):
                for robot in current_state._current_robots:
                    if robot.id == agent_id and hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                        trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                        agent_trust_values.append(trust_value)
                        break
            
            if trust_value is not None:
                if trust_value > 0.5:
                    trustworthy_agents += 1
                else:
                    untrustworthy_agents += 1
            else:
                # Fallback to binary predicate
                if 'agent' in graph_data.node_types and agent_idx < graph_data['agent'].x.shape[0]:
                    if graph_data['agent'].x[agent_idx, 0].item() > 0.5:
                        trustworthy_agents += 1
                    else:
                        untrustworthy_agents += 1
    
    # Analyze track trust status using actual trust values when available
    if hasattr(graph_data, 'track_nodes'):
        for track_id, track_idx in graph_data.track_nodes.items():
            trust_value = None
            if current_state and hasattr(current_state, '_current_tracks'):
                for track in current_state._current_tracks:
                    if track.id == track_id and hasattr(track, 'trust_alpha') and hasattr(track, 'trust_beta'):
                        trust_value = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                        track_trust_values.append(trust_value)
                        break
            
            if trust_value is not None:
                if trust_value > 0.5:
                    trustworthy_tracks += 1
                else:
                    untrustworthy_tracks += 1
            else:
                # Fallback to binary predicate
                if 'track' in graph_data.node_types and track_idx < graph_data['track'].x.shape[0]:
                    if graph_data['track'].x[track_idx, 0].item() > 0.5:
                        trustworthy_tracks += 1
                    else:
                        untrustworthy_tracks += 1
    
    # Calculate trust statistics
    agent_trust_stats = ""
    if agent_trust_values:
        avg_agent_trust = sum(agent_trust_values) / len(agent_trust_values)
        min_agent_trust = min(agent_trust_values)
        max_agent_trust = max(agent_trust_values)
        agent_trust_stats = f"\n  - Avg: {avg_agent_trust:.3f} (Range: {min_agent_trust:.3f}-{max_agent_trust:.3f})"
    
    track_trust_stats = ""
    if track_trust_values:
        avg_track_trust = sum(track_trust_values) / len(track_trust_values)
        min_track_trust = min(track_trust_values)
        max_track_trust = max(track_trust_values)
        track_trust_stats = f"\n  - Avg: {avg_track_trust:.3f} (Range: {min_track_trust:.3f}-{max_track_trust:.3f})"
    
    # Count proximal edges for summary
    proximal_edge_count = 0
    if hasattr(graph_data, 'edge_types'):
        for edge_type in graph_data.edge_types:
            if 'isProximal' in str(edge_type):
                if hasattr(graph_data[edge_type], 'edge_index'):
                    proximal_edge_count = graph_data[edge_type].edge_index.shape[1]
                break
    
    summary_text = f"""Graph Summary:
• Agent nodes: {len(agent_nodes)} (T:{trustworthy_agents}, U:{untrustworthy_agents}){agent_trust_stats}
• Track nodes: {len(track_nodes)} (T:{trustworthy_tracks}, U:{untrustworthy_tracks}){track_trust_stats}
• Edge types: {len(graph_data.edge_types) if hasattr(graph_data, 'edge_types') else 0}
• Robot-Robot proximal edges: {proximal_edge_count}
• Features:
  - Agents: {graph_data['agent'].x.shape if 'agent' in graph_data.node_types else 'N/A'}
  - Tracks: {graph_data['track'].x.shape if 'track' in graph_data.node_types else 'N/A'}
• Trust Display:
  - Numbers show actual trust values (0.000-1.000)
  - (T) = Trustworthy (>0.5), (U) = Untrustworthy (≤0.5)
• Edge Legend:
  - Orange dotted: Robot proximity connections
  - Blue dashed: Track in FoV but not observed
  - Red thick: Track observed within FoV"""
    
    plt.text(1.05, 0.3, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%H%M%S")
    if episode is not None and timestep is not None:
        filename = f"gnn_input_ep{episode:03d}_step{timestep:03d}_{timestamp}.png"
    elif episode is not None:
        filename = f"gnn_input_ep{episode:03d}_{timestamp}.png"
    else:
        filename = f"gnn_input_{timestamp}.png"
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 GNN input visualization saved to: {filepath}")
    
    plt.show()
    return filepath

def main():
    """Main function for standalone testing"""
    print("⚠️  This visualization tool is designed to be called with graph_data from RL training")
    print("Usage: visualize_gnn_input(graph_data, episode=0, timestep=0)")

if __name__ == "__main__":
    main()