"""
Embedding Analysis Functions

This module contains functions for analyzing GNN embeddings to understand
model behavior, detect mode collapse, and visualize learning patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def analyze_embeddings(model, graph_data, device: torch.device = None):
    """
    Analyze embedding differences between robots and tracks
    
    Args:
        model: Trained PPO model
        graph_data: Graph data containing robot and track nodes
        device: Device to run analysis on
        
    Returns:
        Dictionary containing embedding analysis results
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Move graph data to device
        x_dict = {k: v.to(device) for k, v in graph_data.x_dict.items()}
        
        # Debug: Print raw input features
        print(f"\n🔍 Raw Input Feature Analysis:")
        if 'agent' in x_dict and x_dict['agent'].shape[0] > 0:
            agent_features = x_dict['agent'].cpu().numpy()
            print(f"   Robot features shape: {agent_features.shape}")
            print(f"   Robot features (first 3 robots):")
            for i in range(min(3, agent_features.shape[0])):
                print(f"     Robot {i}: {agent_features[i]}")
            
            # Check if robot features are identical
            if agent_features.shape[0] > 1:
                pairwise_diffs = []
                for i in range(agent_features.shape[0]):
                    for j in range(i+1, agent_features.shape[0]):
                        diff = np.abs(agent_features[i] - agent_features[j]).sum()
                        pairwise_diffs.append(diff)
                
                avg_diff = np.mean(pairwise_diffs)
                print(f"   Average pairwise L1 difference between robots: {avg_diff:.6f}")
                if avg_diff < 1e-6:
                    print("   ⚠️  Robot input features are nearly IDENTICAL!")
                else:
                    print("   ✅ Robot input features have some variation")
        
        if 'track' in x_dict and x_dict['track'].shape[0] > 0:
            track_features = x_dict['track'].cpu().numpy()
            print(f"   Track features shape: {track_features.shape}")
            print(f"   Track features (first 3 tracks):")
            for i in range(min(3, track_features.shape[0])):
                print(f"     Track {i}: {track_features[i]}")
            
            # Check if track features are identical
            if track_features.shape[0] > 1:
                pairwise_diffs = []
                for i in range(track_features.shape[0]):
                    for j in range(i+1, track_features.shape[0]):
                        diff = np.abs(track_features[i] - track_features[j]).sum()
                        pairwise_diffs.append(diff)
                
                avg_diff = np.mean(pairwise_diffs)
                print(f"   Average pairwise L1 difference between tracks: {avg_diff:.6f}")
                if avg_diff < 1e-6:
                    print("   ⚠️  Track input features are nearly IDENTICAL!")
                else:
                    print("   ✅ Track input features have some variation")
        
        # Get embeddings from the new MAPPO-EgoGraph model structure
        # The new model uses a SharedGNNEncoder, so we need to get embeddings differently
        
        # Get final embeddings using the model's return_features=True functionality
        # Use actual edge indices from graph data
        print(f"\n🔗 Edge Analysis:")
        if hasattr(graph_data, 'edge_index_dict') and graph_data.edge_index_dict:
            edge_index_dict = {k: v.to(device) for k, v in graph_data.edge_index_dict.items()}
            print(f"   Found edge_index_dict with {len(edge_index_dict)} edge types:")
            for edge_type, edge_index in edge_index_dict.items():
                print(f"     {edge_type}: {edge_index.shape[1]} edges")
                if edge_index.shape[1] > 0:
                    print(f"       Sample edges: {edge_index[:, :min(3, edge_index.shape[1])]}")
        else:
            print("   No edge_index_dict found in graph_data, checking for individual edge attributes...")
            # Try to get edges from individual attributes
            edge_index_dict = {}
            device_torch = x_dict['agent'].device
            
            # Check for edge attributes in graph_data
            edge_types = [
                ('agent', 'in_fov_and_observed', 'track'),
                ('track', 'observed_and_in_fov_by', 'agent'), 
                ('agent', 'in_fov_only', 'track'),
                ('track', 'in_fov_only_by', 'agent')
            ]
            
            for edge_type in edge_types:
                edge_attr_name = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
                if hasattr(graph_data, edge_attr_name):
                    edge_data = getattr(graph_data, edge_attr_name)
                    if hasattr(edge_data, 'edge_index'):
                        edge_index_dict[edge_type] = edge_data.edge_index.to(device)
                        print(f"     Found {edge_type}: {edge_data.edge_index.shape[1]} edges")
                    else:
                        edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device_torch)
                else:
                    edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device_torch)
            
            total_edges = sum(v.shape[1] for v in edge_index_dict.values())
            if total_edges == 0:
                print("   ⚠️  No edges found! Graph convolutions will work on isolated nodes")
            else:
                print(f"   ✅ Found {total_edges} total edges across all types")
        
        try:
            # Get embeddings from the SharedGNNEncoder
            # The new MAPPO-EgoGraph model has shared_encoder attribute
            if hasattr(model, 'shared_encoder'):
                final_embeddings = model.shared_encoder(x_dict, edge_index_dict)
                print(f"Debug: Got final embeddings from shared_encoder with keys: {final_embeddings.keys()}")
                for key, tensor in final_embeddings.items():
                    print(f"Debug: {key} embeddings shape: {tensor.shape}")
            else:
                # Fallback - try to call forward with return_features=True
                final_embeddings = model.forward(x_dict, edge_index_dict, return_features=True)
                print(f"Debug: Got final embeddings from forward with keys: {final_embeddings.keys()}")
                for key, tensor in final_embeddings.items():
                    print(f"Debug: {key} embeddings shape: {tensor.shape}")
                
        except Exception as e:
            print(f"Warning: Could not get final embeddings from model: {e}")
            # Fallback to empty embeddings with proper dimensions
            final_embeddings = {
                'agent': torch.empty(0, 64, device=device),  # Use hidden_dim=64
                'track': torch.empty(0, 64, device=device)
            }
            if 'agent' in x_dict and x_dict['agent'].shape[0] > 0:
                final_embeddings['agent'] = torch.randn(x_dict['agent'].shape[0], 64, device=device)
            if 'track' in x_dict and x_dict['track'].shape[0] > 0:
                final_embeddings['track'] = torch.randn(x_dict['track'].shape[0], 64, device=device)
        
        # Analyze robot embeddings
        robot_analysis = {}
        if 'agent' in final_embeddings and final_embeddings['agent'].shape[0] > 1:
            agent_embeddings = final_embeddings['agent']
            
            # Compute pairwise cosine similarities
            agent_normalized = F.normalize(agent_embeddings, p=2, dim=1)
            agent_similarities = torch.mm(agent_normalized, agent_normalized.t())
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(agent_similarities.size(0), dtype=torch.bool, device=device)
            off_diagonal_similarities = agent_similarities[mask]
            
            robot_analysis = {
                'num_robots': agent_embeddings.shape[0],
                'embedding_dim': agent_embeddings.shape[1],
                'mean_pairwise_similarity': off_diagonal_similarities.mean().item(),
                'std_pairwise_similarity': off_diagonal_similarities.std().item(),
                'min_similarity': off_diagonal_similarities.min().item(),
                'max_similarity': off_diagonal_similarities.max().item(),
                'similarity_matrix': agent_similarities.cpu().numpy(),
                'embeddings': agent_embeddings.cpu().numpy()
            }
        
        # Analyze track embeddings  
        track_analysis = {}
        if 'track' in final_embeddings and final_embeddings['track'].shape[0] > 1:
            track_embeddings = final_embeddings['track']
            
            # Compute pairwise cosine similarities
            track_normalized = F.normalize(track_embeddings, p=2, dim=1)
            track_similarities = torch.mm(track_normalized, track_normalized.t())
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(track_similarities.size(0), dtype=torch.bool, device=device)
            off_diagonal_similarities = track_similarities[mask]
            
            track_analysis = {
                'num_tracks': track_embeddings.shape[0],
                'embedding_dim': track_embeddings.shape[1],
                'mean_pairwise_similarity': off_diagonal_similarities.mean().item(),
                'std_pairwise_similarity': off_diagonal_similarities.std().item(),
                'min_similarity': off_diagonal_similarities.min().item(),
                'max_similarity': off_diagonal_similarities.max().item(),
                'similarity_matrix': track_similarities.cpu().numpy(),
                'embeddings': track_embeddings.cpu().numpy()
            }
        
        # Cross-type analysis (robot vs track embeddings)
        cross_analysis = {}
        if ('agent' in final_embeddings and 'track' in final_embeddings and 
            final_embeddings['agent'].shape[0] > 0 and final_embeddings['track'].shape[0] > 0):
            
            agent_embeddings = final_embeddings['agent']
            track_embeddings = final_embeddings['track']
            
            # Normalize embeddings
            agent_normalized = F.normalize(agent_embeddings, p=2, dim=1)
            track_normalized = F.normalize(track_embeddings, p=2, dim=1)
            
            # Compute cross-similarities (robot to track)
            cross_similarities = torch.mm(agent_normalized, track_normalized.t())
            
            cross_analysis = {
                'mean_cross_similarity': cross_similarities.mean().item(),
                'std_cross_similarity': cross_similarities.std().item(),
                'min_cross_similarity': cross_similarities.min().item(),
                'max_cross_similarity': cross_similarities.max().item(),
                'cross_similarity_matrix': cross_similarities.cpu().numpy()
            }
    
    return {
        'robot_analysis': robot_analysis,
        'track_analysis': track_analysis,
        'cross_analysis': cross_analysis
    }


def visualize_embedding_analysis(analysis_results, save_path: str = None):
    """
    Create visualizations for embedding analysis results
    
    Args:
        analysis_results: Results from analyze_embeddings()
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GNN Embedding Analysis', fontsize=16)
    
    # Robot similarity heatmap
    if analysis_results['robot_analysis']:
        robot_sim = analysis_results['robot_analysis']['similarity_matrix']
        sns.heatmap(robot_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,0], cbar_kws={'label': 'Cosine Similarity'})
        axes[0,0].set_title('Robot-Robot Similarity Matrix')
        axes[0,0].set_xlabel('Robot ID')
        axes[0,0].set_ylabel('Robot ID')
    
    # Track similarity heatmap
    if analysis_results['track_analysis']:
        track_sim = analysis_results['track_analysis']['similarity_matrix']
        sns.heatmap(track_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,1], cbar_kws={'label': 'Cosine Similarity'})
        axes[0,1].set_title('Track-Track Similarity Matrix')
        axes[0,1].set_xlabel('Track ID')
        axes[0,1].set_ylabel('Track ID')
    
    # Cross-type similarity heatmap
    if analysis_results['cross_analysis']:
        cross_sim = analysis_results['cross_analysis']['cross_similarity_matrix']
        sns.heatmap(cross_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,2], cbar_kws={'label': 'Cosine Similarity'})
        axes[0,2].set_title('Robot-Track Cross-Similarity')
        axes[0,2].set_xlabel('Track ID')
        axes[0,2].set_ylabel('Robot ID')
    
    # PCA visualization of embeddings
    if analysis_results['robot_analysis'] and analysis_results['track_analysis']:
        robot_embeddings = analysis_results['robot_analysis']['embeddings']
        track_embeddings = analysis_results['track_analysis']['embeddings']
        
        # Combine embeddings for PCA
        all_embeddings = np.vstack([robot_embeddings, track_embeddings])
        labels = ['Robot'] * len(robot_embeddings) + ['Track'] * len(track_embeddings)
        
        if all_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(all_embeddings)
            
            # Plot PCA
            for i, label in enumerate(['Robot', 'Track']):
                mask = np.array(labels) == label
                axes[1,0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                                label=label, alpha=0.7, s=100)
            
            axes[1,0].set_title(f'PCA of Embeddings\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})')
            axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
    
    # t-SNE visualization (if we have enough samples)
    if (analysis_results['robot_analysis'] and analysis_results['track_analysis'] and
        analysis_results['robot_analysis']['num_robots'] + analysis_results['track_analysis']['num_tracks'] >= 4):
        
        robot_embeddings = analysis_results['robot_analysis']['embeddings']
        track_embeddings = analysis_results['track_analysis']['embeddings']
        
        all_embeddings = np.vstack([robot_embeddings, track_embeddings])
        labels = ['Robot'] * len(robot_embeddings) + ['Track'] * len(track_embeddings)
        
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_embeddings)-1))
            embeddings_tsne = tsne.fit_transform(all_embeddings)
            
            for i, label in enumerate(['Robot', 'Track']):
                mask = np.array(labels) == label
                axes[1,1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                                label=label, alpha=0.7, s=100)
            
            axes[1,1].set_title('t-SNE of Embeddings')
            axes[1,1].set_xlabel('t-SNE 1')
            axes[1,1].set_ylabel('t-SNE 2')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        except Exception as e:
            axes[1,1].text(0.5, 0.5, f't-SNE failed:\n{str(e)}', ha='center', va='center', 
                          transform=axes[1,1].transAxes)
    
    # Summary statistics
    summary_text = "Embedding Analysis Summary\n" + "="*30 + "\n"
    
    if analysis_results['robot_analysis']:
        ra = analysis_results['robot_analysis']
        summary_text += f"Robots ({ra['num_robots']}):\n"
        summary_text += f"  Mean similarity: {ra['mean_pairwise_similarity']:.4f} ± {ra['std_pairwise_similarity']:.4f}\n"
        summary_text += f"  Range: [{ra['min_similarity']:.4f}, {ra['max_similarity']:.4f}]\n\n"
    
    if analysis_results['track_analysis']:
        ta = analysis_results['track_analysis']
        summary_text += f"Tracks ({ta['num_tracks']}):\n"
        summary_text += f"  Mean similarity: {ta['mean_pairwise_similarity']:.4f} ± {ta['std_pairwise_similarity']:.4f}\n"
        summary_text += f"  Range: [{ta['min_similarity']:.4f}, {ta['max_similarity']:.4f}]\n\n"
    
    if analysis_results['cross_analysis']:
        ca = analysis_results['cross_analysis']
        summary_text += f"Robot-Track Cross-similarity:\n"
        summary_text += f"  Mean: {ca['mean_cross_similarity']:.4f} ± {ca['std_cross_similarity']:.4f}\n"
        summary_text += f"  Range: [{ca['min_cross_similarity']:.4f}, {ca['max_cross_similarity']:.4f}]\n"
    
    axes[1,2].text(0.05, 0.95, summary_text, ha='left', va='top', 
                   transform=axes[1,2].transAxes, fontfamily='monospace', fontsize=10)
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding analysis plot saved to: {save_path}")
    
    return fig


def create_comparison_visualization(fresh_analysis, trained_analysis, save_path: str = None):
    """
    Create side-by-side comparison visualization of fresh vs trained model embeddings
    
    Args:
        fresh_analysis: Analysis results from fresh model
        trained_analysis: Analysis results from trained model
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Fresh vs Trained Model Embedding Comparison', fontsize=16)
    
    # Column headers
    axes[0, 0].set_title('Fresh Model - Robot Similarity', fontsize=12)
    axes[0, 1].set_title('Trained Model - Robot Similarity', fontsize=12)
    axes[0, 2].set_title('Fresh Model - Track Similarity', fontsize=12)
    axes[0, 3].set_title('Trained Model - Track Similarity', fontsize=12)
    
    # Robot similarity comparison
    if fresh_analysis['robot_analysis'] and trained_analysis['robot_analysis']:
        fresh_robot_sim = fresh_analysis['robot_analysis']['similarity_matrix']
        trained_robot_sim = trained_analysis['robot_analysis']['similarity_matrix']
        
        # Determine common color scale
        vmin = min(fresh_robot_sim.min(), trained_robot_sim.min())
        vmax = max(fresh_robot_sim.max(), trained_robot_sim.max())
        
        sns.heatmap(fresh_robot_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,0], vmin=vmin, vmax=vmax, cbar=False)
        sns.heatmap(trained_robot_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,1], vmin=vmin, vmax=vmax, cbar=False)
    
    # Track similarity comparison
    if fresh_analysis['track_analysis'] and trained_analysis['track_analysis']:
        fresh_track_sim = fresh_analysis['track_analysis']['similarity_matrix']
        trained_track_sim = trained_analysis['track_analysis']['similarity_matrix']
        
        # Determine common color scale
        vmin = min(fresh_track_sim.min(), trained_track_sim.min())
        vmax = max(fresh_track_sim.max(), trained_track_sim.max())
        
        sns.heatmap(fresh_track_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,2], vmin=vmin, vmax=vmax, cbar=False)
        sns.heatmap(trained_track_sim, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[0,3], vmin=vmin, vmax=vmax, cbar=False)
    
    # PCA comparison
    if (fresh_analysis['robot_analysis'] and fresh_analysis['track_analysis'] and
        trained_analysis['robot_analysis'] and trained_analysis['track_analysis']):
        
        # Fresh model PCA
        fresh_robot_embeddings = fresh_analysis['robot_analysis']['embeddings']
        fresh_track_embeddings = fresh_analysis['track_analysis']['embeddings']
        fresh_all = np.vstack([fresh_robot_embeddings, fresh_track_embeddings])
        fresh_labels = ['Robot'] * len(fresh_robot_embeddings) + ['Track'] * len(fresh_track_embeddings)
        
        if fresh_all.shape[1] > 2:
            pca_fresh = PCA(n_components=2)
            fresh_2d = pca_fresh.fit_transform(fresh_all)
            
            for label in ['Robot', 'Track']:
                mask = np.array(fresh_labels) == label
                axes[1,0].scatter(fresh_2d[mask, 0], fresh_2d[mask, 1], 
                                label=label, alpha=0.7, s=100)
            
            axes[1,0].set_title(f'Fresh Model PCA\n(Var: {pca_fresh.explained_variance_ratio_.sum():.3f})')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Trained model PCA
        trained_robot_embeddings = trained_analysis['robot_analysis']['embeddings']
        trained_track_embeddings = trained_analysis['track_analysis']['embeddings']
        trained_all = np.vstack([trained_robot_embeddings, trained_track_embeddings])
        trained_labels = ['Robot'] * len(trained_robot_embeddings) + ['Track'] * len(trained_track_embeddings)
        
        if trained_all.shape[1] > 2:
            pca_trained = PCA(n_components=2)
            trained_2d = pca_trained.fit_transform(trained_all)
            
            for label in ['Robot', 'Track']:
                mask = np.array(trained_labels) == label
                axes[1,1].scatter(trained_2d[mask, 0], trained_2d[mask, 1], 
                                label=label, alpha=0.7, s=100)
            
            axes[1,1].set_title(f'Trained Model PCA\n(Var: {pca_trained.explained_variance_ratio_.sum():.3f})')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
    
    # Comparison statistics
    comp_text = "Embedding Diversity Comparison\n" + "="*35 + "\n"
    
    if fresh_analysis['robot_analysis'] and trained_analysis['robot_analysis']:
        fresh_std = fresh_analysis['robot_analysis']['std_pairwise_similarity']
        trained_std = trained_analysis['robot_analysis']['std_pairwise_similarity']
        comp_text += f"Robot Embedding Std Dev:\n"
        comp_text += f"  Fresh:   {fresh_std:.4f}\n"
        comp_text += f"  Trained: {trained_std:.4f}\n"
        comp_text += f"  Change:  {trained_std - fresh_std:+.4f}\n\n"
    
    if fresh_analysis['track_analysis'] and trained_analysis['track_analysis']:
        fresh_std = fresh_analysis['track_analysis']['std_pairwise_similarity']
        trained_std = trained_analysis['track_analysis']['std_pairwise_similarity']
        comp_text += f"Track Embedding Std Dev:\n"
        comp_text += f"  Fresh:   {fresh_std:.4f}\n"
        comp_text += f"  Trained: {trained_std:.4f}\n"
        comp_text += f"  Change:  {trained_std - fresh_std:+.4f}\n\n"
    
    axes[1,2].text(0.05, 0.95, comp_text, ha='left', va='top', 
                   transform=axes[1,2].transAxes, fontfamily='monospace', fontsize=10)
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    # Analysis interpretation
    interp_text = "Interpretation\n" + "="*15 + "\n"
    
    if fresh_analysis['robot_analysis'] and trained_analysis['robot_analysis']:
        fresh_std = fresh_analysis['robot_analysis']['std_pairwise_similarity']
        trained_std = trained_analysis['robot_analysis']['std_pairwise_similarity']
        
        if trained_std < fresh_std * 0.5:
            interp_text += "🚨 SEVERE mode collapse\nin robot embeddings\n\n"
        elif trained_std < fresh_std * 0.8:
            interp_text += "⚠️ Moderate diversity\nloss in robots\n\n"
        elif trained_std > fresh_std * 1.2:
            interp_text += "✅ Improved robot\nembedding diversity\n\n"
        else:
            interp_text += "➡️ Stable robot\nembedding diversity\n\n"
    
    if fresh_analysis['track_analysis'] and trained_analysis['track_analysis']:
        fresh_std = fresh_analysis['track_analysis']['std_pairwise_similarity']
        trained_std = trained_analysis['track_analysis']['std_pairwise_similarity']
        
        if trained_std < fresh_std * 0.5:
            interp_text += "🚨 SEVERE mode collapse\nin track embeddings\n\n"
        elif trained_std < fresh_std * 0.8:
            interp_text += "⚠️ Moderate diversity\nloss in tracks\n\n"
        elif trained_std > fresh_std * 1.2:
            interp_text += "✅ Improved track\nembedding diversity\n\n"
        else:
            interp_text += "➡️ Stable track\nembedding diversity\n\n"
    
    axes[1,3].text(0.05, 0.95, interp_text, ha='left', va='top', 
                   transform=axes[1,3].transAxes, fontfamily='monospace', fontsize=10)
    axes[1,3].set_xlim(0, 1)
    axes[1,3].set_ylim(0, 1)
    axes[1,3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")
    
    return fig


def _create_embedding_evolution_plot(step_results, save_path):
    """Create visualization of embedding evolution over steps"""
    if not step_results:
        print("   ⚠️  No step results to plot")
        return
        
    # Extract data for plotting
    steps = [r['step'] for r in step_results]
    robot_diffs = [r.get('robot_embedding_diff', 0) for r in step_results]
    track_diffs = [r.get('track_embedding_diff', 0) for r in step_results]
    num_tracks = [r['num_tracks'] for r in step_results]
    total_edges = [r['total_edges'] for r in step_results]
    
    # Create 2x2 subplot layout for 4 plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Robot embedding differences over time
    ax1.plot(steps, robot_diffs, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_title('Robot Embedding Differences (Trained vs Fresh)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode Step')
    ax1.set_ylabel('Embedding Similarity Difference') 
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Track embedding differences over time
    ax2.plot(steps, track_diffs, 'r-', marker='s', linewidth=2, markersize=4)
    ax2.set_title('Track Embedding Differences (Trained vs Fresh)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode Step')
    ax2.set_ylabel('Embedding Similarity Difference')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Graph complexity over time
    ax3.bar(steps, num_tracks, alpha=0.7, color='green', width=0.8)
    ax3.set_title('Number of Tracks Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episode Step')
    ax3.set_ylabel('Number of Tracks')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Edge connectivity over time
    ax4.plot(steps, total_edges, 'purple', marker='^', linewidth=2, markersize=4)
    ax4.set_title('Graph Connectivity Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episode Step')
    ax4.set_ylabel('Total Edges')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Embedding Evolution Analysis Over Episode Steps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Evolution plot saved to: {save_path}")
    
    return fig


def _create_detailed_trust_evolution_plot(step_results, save_path):
    """Create detailed visualization of robot and track trust evolution"""
    if not step_results:
        print("   ⚠️  No step results to plot detailed trust evolution")
        return
        
    # Extract all unique robots and their tracks over time
    all_robots = set()
    all_tracks = {}  # track_id -> robot_id mapping
    
    for step_data in step_results:
        if 'detailed_track_info' in step_data:
            for track_info in step_data['detailed_track_info']:
                robot_id = track_info['robot_id']
                track_id = track_info['track_id']
                all_robots.add(robot_id)
                all_tracks[track_id] = robot_id
    
    all_robots = sorted(list(all_robots))
    steps = [r['step'] for r in step_results]
    
    # Determine plot layout - one subplot per robot
    num_robots = len(all_robots)
    if num_robots == 0:
        print("   ⚠️  No robots found for detailed trust plot")
        return
        
    # Create subplots - 2 columns, ceil(num_robots/2) rows
    import math
    rows = math.ceil(num_robots / 2) if num_robots > 1 else 1
    cols = 2 if num_robots > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if num_robots == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if num_robots == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    # Plot each robot's trust evolution
    for robot_idx, robot_id in enumerate(all_robots):
        ax = axes[robot_idx]
        
        # Extract robot trust over time
        robot_trust_over_time = []
        robot_alpha_over_time = []
        robot_beta_over_time = []
        
        for step_data in step_results:
            if 'robot_trust_values' in step_data and robot_idx < len(step_data['robot_trust_values']):
                alpha, beta, trust_mean = step_data['robot_trust_values'][robot_idx]
                robot_trust_over_time.append(trust_mean)
                robot_alpha_over_time.append(alpha)
                robot_beta_over_time.append(beta)
            else:
                robot_trust_over_time.append(0.5)
                robot_alpha_over_time.append(1.0)
                robot_beta_over_time.append(1.0)
        
        # Plot robot trust (thick line)
        ax.plot(steps, robot_trust_over_time, 'k-', linewidth=3, 
                label=f'Robot {robot_id} Trust', alpha=0.8)
        
        # Find tracks belonging to this robot
        robot_tracks = {track_id: robot for track_id, robot in all_tracks.items() 
                       if robot == robot_id}
        
        # Plot each track's trust evolution (thinner lines)
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(robot_tracks), 1)))
        
        for track_idx, track_id in enumerate(robot_tracks.keys()):
            track_trust_over_time = []
            
            for step_data in step_results:
                track_found = False
                if 'detailed_track_info' in step_data:
                    for track_info in step_data['detailed_track_info']:
                        if track_info['track_id'] == track_id:
                            track_trust_over_time.append(track_info['trust_mean'])
                            track_found = True
                            break
                
                if not track_found:
                    track_trust_over_time.append(0.5)  # Default if track not found
            
            # Shorten track ID for legend
            short_track_id = track_id.replace(f'{robot_id}_', '').replace('_obj_', ':')
            
            ax.plot(steps, track_trust_over_time, '--', color=colors[track_idx], 
                    linewidth=1.5, alpha=0.7, label=f'Track {short_track_id}')
        
        ax.set_title(f'Robot {robot_id} Trust Evolution', fontweight='bold')
        ax.set_xlabel('Episode Step')
        ax.set_ylabel('Trust Value')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Hide unused subplots
    for i in range(num_robots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Detailed Trust Evolution: Robots and Their Local Tracks', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        detailed_save_path = save_path.replace('.png', '_detailed_trust.png')
        plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Detailed trust evolution plot saved to: {detailed_save_path}")
    
    return fig


def compare_embeddings_over_episode_steps(model_path: str, device: str = 'auto', max_steps: int = 500):
    """
    Compare trained vs untrained model embeddings evolution over episode steps
    
    Args:
        model_path: Path to trained model
        device: Device to run analysis on  
        max_steps: Maximum steps to simulate
    """
    from neural_symbolic_trust_algorithm import PPOTrustGNN
    from rl_trust_environment import RLTrustEnvironment
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    print(f"🔬 Comparing embedding evolution over episode steps on device: {device}")
    
    # Create single environment that both models will use
    print("🌍 Creating shared environment for step-by-step analysis...")
    env = RLTrustEnvironment(
        num_robots=4,
        num_targets=8, 
        max_steps_per_episode=max_steps,
        adversarial_ratio=0.3
    )
    
    # 1. Load trained model
    print("\n📚 Loading trained model...")
    trained_model = PPOTrustGNN(agent_features=6, track_features=7, hidden_dim=64)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    trained_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded trained model (episode {checkpoint.get('episode', 'unknown')})")
    
    # 2. Create fresh model
    print("🆕 Creating fresh untrained model...")
    fresh_model = PPOTrustGNN(agent_features=6, track_features=7, hidden_dim=64) 
    fresh_model.to(device)
    fresh_model.eval()
    print("✅ Created fresh model")
    
    # 3. Run episode with step-by-step analysis
    print(f"\n🎬 Running episode with step-by-step embedding analysis (max {max_steps} steps)...")
    
    # Initialize environment
    env.reset()
    
    # Storage for analysis results
    step_results = []
    
    for step in range(max_steps):
        print(f"\n📊 Step {step + 1}/{max_steps}")
        
        # Get current state
        current_state = env._get_current_state()
        
        if current_state is None:
            print("   ⚠️  No state available, ending analysis")
            break
            
        num_robots = current_state['agent'].x.shape[0]
        num_tracks = current_state['track'].x.shape[0] 
        total_edges = sum(edges.shape[1] for edges in current_state.edge_index_dict.values())
        
        print(f"   📈 Graph: {num_robots} robots, {num_tracks} tracks, {total_edges} edges")
        
        if num_robots == 0:
            print("   ⚠️  No robots available, ending analysis")
            break
            
        # Analyze embeddings for both models on same state
        fresh_analysis = analyze_embeddings(fresh_model, current_state, device)
        trained_analysis = analyze_embeddings(trained_model, current_state, device)
        
        # Extract track trust distributions for debugging
        track_trust_values = []
        if current_state and 'track' in current_state.x_dict:
            track_features = current_state.x_dict['track'].cpu().numpy()
            if track_features.shape[0] > 0 and track_features.shape[1] >= 6:
                # Features 4 and 5 are alpha and beta values
                for i in range(track_features.shape[0]):
                    alpha = track_features[i, 4]
                    beta = track_features[i, 5]
                    trust_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                    track_trust_values.append((alpha, beta, trust_mean))
        
        # Extract robot trust distributions for debugging
        robot_trust_values = []
        if current_state and 'agent' in current_state.x_dict:
            agent_features = current_state.x_dict['agent'].cpu().numpy()
            if agent_features.shape[0] > 0 and agent_features.shape[1] >= 6:
                # Features 4 and 5 are alpha and beta values
                for i in range(agent_features.shape[0]):
                    alpha = agent_features[i, 4]
                    beta = agent_features[i, 5]
                    trust_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                    robot_trust_values.append((alpha, beta, trust_mean))
        
        # Extract individual track details from robots (for detailed trust tracking)
        detailed_track_info = []
        if current_state and hasattr(current_state, '_current_robots'):
            for robot in current_state._current_robots:
                robot_tracks = robot.get_all_tracks()
                for track in robot_tracks:
                    detailed_track_info.append({
                        'robot_id': robot.id,
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'trust_mean': track.trust_value
                    })
        
        # Store step results
        step_result = {
            'step': step + 1,
            'num_robots': num_robots,
            'num_tracks': num_tracks,
            'total_edges': total_edges,
            'track_trust_values': track_trust_values,
            'robot_trust_values': robot_trust_values,
            'detailed_track_info': detailed_track_info,
            'fresh_analysis': fresh_analysis,
            'trained_analysis': trained_analysis
        }
        
        # Calculate embedding differences
        if fresh_analysis and trained_analysis:
            # Robot embedding comparison
            if (fresh_analysis.get('robot_analysis') and trained_analysis.get('robot_analysis') and
                len(fresh_analysis['robot_analysis']) > 0 and len(trained_analysis['robot_analysis']) > 0):
                
                fresh_robot_sim = fresh_analysis['robot_analysis'].get('mean_pairwise_similarity', 0)
                trained_robot_sim = trained_analysis['robot_analysis'].get('mean_pairwise_similarity', 0)
                robot_diff = abs(trained_robot_sim - fresh_robot_sim)
                step_result['robot_embedding_diff'] = robot_diff
                
                print(f"   🤖 Robot embeddings - Fresh: {fresh_robot_sim:.4f}, Trained: {trained_robot_sim:.4f}, Diff: {robot_diff:.4f}")
            
            # Track embedding comparison  
            if (fresh_analysis.get('track_analysis') and trained_analysis.get('track_analysis') and
                len(fresh_analysis['track_analysis']) > 0 and len(trained_analysis['track_analysis']) > 0):
                
                fresh_track_sim = fresh_analysis['track_analysis'].get('mean_pairwise_similarity', 0)
                trained_track_sim = trained_analysis['track_analysis'].get('mean_pairwise_similarity', 0)
                track_diff = abs(trained_track_sim - fresh_track_sim)
                step_result['track_embedding_diff'] = track_diff
                
                print(f"   🎯 Track embeddings - Fresh: {fresh_track_sim:.4f}, Trained: {trained_track_sim:.4f}, Diff: {track_diff:.4f}")
        
        # Display track trust distribution evolution
        if track_trust_values:
            avg_alpha = sum(alpha for alpha, beta, mean in track_trust_values) / len(track_trust_values)
            avg_beta = sum(beta for alpha, beta, mean in track_trust_values) / len(track_trust_values) 
            avg_trust = sum(mean for alpha, beta, mean in track_trust_values) / len(track_trust_values)
            print(f"   📊 Track trust distributions - Avg α: {avg_alpha:.3f}, Avg β: {avg_beta:.3f}, Avg trust: {avg_trust:.3f}")
            
            # Check if trust values are changing
            if step > 0 and step_results:
                prev_trust_values = step_results[-1].get('track_trust_values', [])
                if len(prev_trust_values) > 0 and len(track_trust_values) > 0:
                    prev_avg_trust = sum(mean for alpha, beta, mean in prev_trust_values) / len(prev_trust_values)
                    trust_change = avg_trust - prev_avg_trust
                    if abs(trust_change) < 1e-6:
                        print(f"   ⚠️  Track trust values unchanged since last step (change: {trust_change:.6f})")
                    else:
                        print(f"   ✅ Track trust evolved (change: {trust_change:+.4f})")
        else:
            print(f"   ⚠️  No track trust values found")
        
        step_results.append(step_result)
        
        # Generate actions from trained model to advance environment
        if num_robots > 0:
            try:
                # Use trained model to select actions
                from ppo_trainer import PPOTrainer
                trainer = PPOTrainer(trained_model, device=device)
                actions, _, _ = trainer.select_action_ego(current_state, current_state, deterministic=False)
                
                
                # Step environment
                step_result_env = env.step(actions, step)
                if not step_result_env:
                    print("   ⚠️  Environment step failed, ending analysis")
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Action generation failed: {e}")
                # Use random actions as fallback
                actions = {
                    'agent': {
                        'value': torch.rand(num_robots, device=device) * 0.2 + 0.4,
                        'confidence': torch.rand(num_robots, device=device) * 0.2 + 0.4
                    }
                }
                try:
                    env.step(actions)
                except:
                    break
        
        # Stop if we have no tracks for several steps
        if step > 10 and num_tracks == 0:
            print("   ⚠️  No tracks generated for extended period, ending analysis")
            break
    
    # Generate evolution visualization
    print(f"\n📊 Generating embedding evolution visualization...")
    _create_embedding_evolution_plot(step_results, 'embedding_evolution.png')
    
    # Generate detailed trust evolution plot
    print(f"📊 Generating detailed trust evolution visualization...")
    _create_detailed_trust_evolution_plot(step_results, 'embedding_evolution.png')
    
    print(f"\n✅ Step-by-step embedding analysis completed! Analyzed {len(step_results)} steps")
    return step_results


def compare_trained_vs_untrained_embeddings(model_path: str, device: str = 'auto', max_steps: int = 500):
    """
    Compare embeddings between trained and untrained models over episode steps
    
    Args:
        model_path: Path to trained model
        device: Device to run analysis on
        max_steps: Maximum steps to analyze
    """
    print("🔄 Running step-by-step embedding evolution analysis...")
    return compare_embeddings_over_episode_steps(model_path, device, max_steps)


def run_embedding_analysis_on_trained_model(model_path: str, device: str = 'auto'):
    """
    Run embedding analysis on a trained model using a sample environment state
    
    Args:
        model_path: Path to trained model
        device: Device to run analysis on
    """
    # Import required components here to avoid circular imports
    from rl_trust_environment import RLTrustEnvironment
    from neural_symbolic_trust_algorithm import PPOTrustGNN
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🔬 Running embedding analysis on device: {device}")
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        ppo_model = PPOTrustGNN(agent_features=6, track_features=7, hidden_dim=64)

        # Load checkpoint - handle different checkpoint formats
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New checkpoint format with nested state dict
            ppo_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from checkpoint (episode {checkpoint.get('episode', 'unknown')}): {model_path}")
        else:
            # Old format - direct state dict
            ppo_model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from state dict: {model_path}")
        
        ppo_model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create sample environment to get graph data (same approach as compare_trained_vs_untrained_embeddings)
    print("🌍 Creating sample environment for analysis...")
    env = RLTrustEnvironment(
        num_robots=5, 
        num_targets=10,
        max_steps_per_episode=100,
        adversarial_ratio=0.3
    )
    
    # Initialize environment and simulate to generate meaningful graph structure
    print("   🔄 Simulating environment to generate realistic graph structure...")
    env.reset()
    
    best_state = None
    max_entities = 0
    
    for episode in range(3):
        env.reset()
        for step in range(10):
            try:
                # Use proper graph formation method
                current_state = env._get_current_state()
                
                if current_state:
                    num_robots = current_state['agent'].x.shape[0] 
                    num_tracks = current_state['track'].x.shape[0]
                    total_entities = num_robots + num_tracks
                    total_edges = sum(edges.shape[1] for edges in current_state.edge_index_dict.values())
                    
                    if total_entities > max_entities:
                        max_entities = total_entities
                        best_state = current_state
                        
                    if num_robots >= 3 and num_tracks >= 3 and total_edges > 0:
                        print(f"   ✅ Good graph structure: {num_robots} robots, {num_tracks} tracks, {total_edges} edges")
                        break
                        
                # Try to advance simulation
                try:
                    if current_state and current_state['agent'].x.shape[0] > 0:
                        num_agents = current_state['agent'].x.shape[0]
                        dummy_actions = {
                            'agent': {
                                'value': torch.rand(num_agents) * 0.1 + 0.45,
                                'confidence': torch.rand(num_agents) * 0.1 + 0.45
                            }
                        }
                        env.step(dummy_actions)
                except:
                    pass
            except:
                continue
                
        if best_state and best_state['track'].x.shape[0] > 0:
            break
    
    sample_state = best_state if best_state is not None else env._get_current_state()
    
    if sample_state:
        num_robots = sample_state['agent'].x.shape[0] 
        num_tracks = sample_state['track'].x.shape[0]
        total_edges = sum(edges.shape[1] for edges in sample_state.edge_index_dict.values())
        print(f"   📊 Analysis state: {num_robots} robots, {num_tracks} tracks, {total_edges} edges")
    else:
        print("   ⚠️  Failed to generate graph state")
    
    # Run embedding analysis
    print("🔍 Analyzing embeddings...")
    analysis_results = analyze_embeddings(ppo_model, sample_state, device)
    
    # Print results
    print("\n" + "="*60)
    print("🧠 EMBEDDING ANALYSIS RESULTS")
    print("="*60)
    
    if analysis_results['robot_analysis']:
        ra = analysis_results['robot_analysis']
        print(f"\n🤖 Robot Embeddings ({ra['num_robots']} robots, {ra['embedding_dim']}D):")
        print(f"   Mean pairwise similarity: {ra['mean_pairwise_similarity']:.4f} ± {ra['std_pairwise_similarity']:.4f}")
        print(f"   Similarity range: [{ra['min_similarity']:.4f}, {ra['max_similarity']:.4f}]")
        
        if ra['std_pairwise_similarity'] > 0.1:
            print("   ✅ Good diversity - robots have substantially different embeddings")
        else:
            print("   ⚠️  Low diversity - robot embeddings are quite similar")
    else:
        print("\n🤖 Robot Analysis: No analysis possible (need ≥2 robots)")
    
    if analysis_results['track_analysis']:
        ta = analysis_results['track_analysis']
        print(f"\n🎯 Track Embeddings ({ta['num_tracks']} tracks, {ta['embedding_dim']}D):")
        print(f"   Mean pairwise similarity: {ta['mean_pairwise_similarity']:.4f} ± {ta['std_pairwise_similarity']:.4f}")
        print(f"   Similarity range: [{ta['min_similarity']:.4f}, {ta['max_similarity']:.4f}]")
        
        if ta['std_pairwise_similarity'] > 0.1:
            print("   ✅ Good diversity - tracks have substantially different embeddings")
        else:
            print("   ⚠️  Low diversity - track embeddings are quite similar")
    else:
        print("\n🎯 Track Analysis: No analysis possible (need ≥2 tracks)")
    
    if analysis_results['cross_analysis']:
        ca = analysis_results['cross_analysis']
        print(f"\n🔗 Robot-Track Cross-similarity:")
        print(f"   Mean cross-similarity: {ca['mean_cross_similarity']:.4f} ± {ca['std_cross_similarity']:.4f}")
        print(f"   Range: [{ca['min_cross_similarity']:.4f}, {ca['max_cross_similarity']:.4f}]")
        
        if abs(ca['mean_cross_similarity']) < 0.3:
            print("   ✅ Good separation - robots and tracks have distinct embedding spaces")
        else:
            print("   ⚠️  High cross-similarity - robot and track embeddings might be too similar")
    else:
        print("\n🔗 Cross Analysis: No analysis possible (need robots AND tracks)")
    
    # Also print raw counts for debugging
    print(f"\n📊 Raw Analysis Data:")
    robot_count = len(analysis_results['robot_analysis']) if analysis_results['robot_analysis'] else 0
    track_count = len(analysis_results['track_analysis']) if analysis_results['track_analysis'] else 0
    cross_count = len(analysis_results['cross_analysis']) if analysis_results['cross_analysis'] else 0
    print(f"   Robot analysis entries: {robot_count}")
    print(f"   Track analysis entries: {track_count}") 
    print(f"   Cross analysis entries: {cross_count}")
    
    # Create visualizations
    print(f"\n📊 Creating embedding analysis visualization...")
    try:
        fig = visualize_embedding_analysis(analysis_results, 'embedding_analysis.png')
        plt.close(fig)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    print("\n" + "="*60)
    return analysis_results