"""
Neural-Symbolic Multi-robot Trust - GNN Reinforcement Learning Training

This is the main training script for the trust-based Graph Neural Network using
Proximal Policy Optimization (PPO). The components have been refactored into
separate modules for better organization:

- ppo_trainer.py: Contains the PPOTrainer class and PPOExperience namedtuple
- rl_trust_environment.py: Contains the RLTrustEnvironment class
- embedding_analysis.py: Contains all embedding analysis and visualization functions
"""

import torch
import numpy as np
from neural_symbolic_trust_algorithm import PPOTrustGNN
from visualize_gnn_input_graph import visualize_gnn_input

# Import from the new separate modules
from ppo_trainer import PPOTrainer, PPOExperience
from rl_trust_environment import RLTrustEnvironment
from embedding_analysis import compare_trained_vs_untrained_embeddings, run_embedding_analysis_on_trained_model

def train_gnn_with_ppo(episodes=1000, max_steps_per_episode=500, device='cpu', save_path='ppo_trust_gnn.pth', 
                       enable_visualization=True, visualize_frequency=50, visualize_steps=[100, 150, 250, 350],
                       # Environment parameters
                       num_robots=5, num_targets=20, adversarial_ratio=0.5,
                       world_size=(60, 60), false_positive_rate=0.5, false_negative_rate=0.0,
                       movement_speed=1.0, proximal_range=100.0, fov_range=50.0, fov_angle=np.pi/3):
    """
    Main PPO training loop
    
    Args:
        episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode (default: 500)
        device: Device to use ('cpu', 'cuda', or 'auto')
        save_path: Path to save the trained model
        enable_visualization: Enable GNN input visualization
        visualize_frequency: Frequency of visualization (every N episodes)
        visualize_steps: List of steps to visualize within episodes
        
        # Environment parameters
        num_robots: Number of robots in the simulation (default: 5)
        num_targets: Number of target objects (default: 20)
        adversarial_ratio: Ratio of adversarial robots (default: 0.5)
        world_size: Size of the simulation world (default: (60, 60))
        false_positive_rate: Rate of false positive detections (default: 0.5)
        false_negative_rate: Rate of false negative detections (default: 0.0)
        movement_speed: Speed of robot movement (default: 1.0)
        proximal_range: Range for proximal robot interactions (default: 100.0)
        fov_range: Field of view range for robots (default: 50.0)
        fov_angle: Field of view angle in radians (default: π/3, 60 degrees)
    """
    
    print("🚀 GNN REINFORCEMENT LEARNING TRAINING (PPO)")
    print("=" * 60)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # print(f"Device: {device}")
    
    # Initialize environment with all configurable parameters
    # print("🌍 Initializing RL environment...")
    env = RLTrustEnvironment(
        num_robots=num_robots,
        num_targets=num_targets, 
        adversarial_ratio=adversarial_ratio,
        max_steps_per_episode=max_steps_per_episode,
        world_size=world_size,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        movement_speed=movement_speed,
        proximal_range=proximal_range,
        fov_range=fov_range,
        fov_angle=fov_angle
    )
    
    # Initialize PPO model with simplified neural-symbolic features
    # print("🤖 Initializing PPO model...")
    ppo_model = PPOTrustGNN(agent_features=4, track_features=4, hidden_dim=64)  # Updated feature counts
    trainer = PPOTrainer(ppo_model, device=device)
    
    # IMPORTANT: Pass trainer to environment for multi-ego action selection
    env._trainer_instance = trainer
    
    # print(f"Model parameters: {sum(p.numel() for p in ppo_model.parameters()):,}")
    
    # Training loop
    print(f"Starting PPO training for {episodes} episodes on {device}")
    print(f"Using multi-ego training mode (all robots serve as ego each timestep)")
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # Collect experience for one episode
        while step_count < max_steps_per_episode:  # Max steps per episode
            # Select action with policy outputs for visualization
            actions, log_probs, values = trainer.select_action(state)
            
            # Visualize GNN input graph if it's time
            if enable_visualization and episode % visualize_frequency == 0 and step_count in visualize_steps:
                try:
                    # print(f"🎯 Visualizing GNN input graph - Episode {episode}, Step {step_count}")
                    visualize_gnn_input(state, episode=episode, timestep=step_count, current_state=state)
                except Exception as e:
                    # print(f"Warning: GNN visualization failed: {e}")
                    pass
            
            # Take environment step
            next_state, reward, done, info = env.step(actions)
            
            # Store experience
            experience = PPOExperience(
                graph_data=state,
                action=actions,
                reward=reward,
                log_prob=log_probs,
                value=values,
                done=done,
                next_graph_data=next_state
            )
            trainer.add_experience(experience)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # Finish the episode
        trainer.finish_episode()
        
        # Update policy when we have enough episodes collected
        losses = trainer.update_policy()  # This handles the min_episodes check internally
        
        episode_rewards.append(episode_reward)
        
        # Show experience buffer status
        total_experiences = sum(len(ep) for ep in trainer.episode_experiences)
        buffer_episodes = len(trainer.episode_experiences)
        
        # Brief episode completion logging with buffer status
        print(f"Episode {episode:4d} completed | Reward: {episode_reward:7.2f} | Steps: {step_count:3d} | Buffer: {total_experiences:3d} exp / {buffer_episodes:2d} eps")
        
        # Detailed logging after policy updates
        if losses:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            current_lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer, 'scheduler') else trainer.optimizer.param_groups[0]['lr']
            print(f"🔄 POLICY UPDATE after Episode {episode}")
            print(f"   Avg Reward (last 10): {avg_reward:8.2f} | Current: {episode_reward:8.2f}")
            print(f"   Policy Loss: {losses.get('policy_loss', 0):8.4f} | Value Loss: {losses.get('value_loss', 0):8.2f}")
            print(f"   Entropy: {losses.get('entropy_loss', 0):8.3f} | Learning Rate: {current_lr:.6f}")
            print(f"   Updates: Policy={losses.get('num_policy_updates', 0)} Value={losses.get('num_value_updates', 0)}")
            print("-" * 60)
        
        # Periodic learning progress summary
        if episode % 10 == 0 and episode > 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            reward_trend = "↗️" if len(episode_rewards) > 5 and np.mean(episode_rewards[-5:]) > np.mean(episode_rewards[-10:-5]) else "➡️"
            print(f"📊 Episode {episode:3d} Summary: Avg Reward = {np.mean(recent_rewards):6.2f} {reward_trend} | Buffer: {total_experiences} experiences")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'model_state_dict': ppo_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'episode_rewards': episode_rewards
            }, save_path)
    
    print(f"\nTraining completed!")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average final 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Model saved to: {save_path}")
    
    return episode_rewards


def main():
    """Main function"""
    import sys
    
    # Parse command line arguments
    device = 'auto'
    episodes = 1000
    enable_visualization = True
    analyze_embeddings_mode = False
    model_path = 'ppo_trust_gnn.pth'
    
    # Environment configuration parameters  
    num_robots = 5
    num_targets = 10
    adversarial_ratio = 0.5
    world_size = (100, 100)
    false_positive_rate = 0.3  # Reduced from 0.5 for more realistic simulation
    false_negative_rate = 0.1  # Increased from 0.0 to add missed detections  
    movement_speed = 1.0
    proximal_range = 100.0      # Reduced from 100.0 to match world size better
    fov_range = 50.0           # Reduced from 50.0 for more realistic observation patterns
    fov_angle = np.pi/3
    
    # Parse all arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ['--no-viz', '--disable-visualization', '--no-visualization']:
            enable_visualization = False
            # print("📊 Visualization disabled for large-scale training")
        elif arg in ['--analyze-embeddings', '--embedding-analysis', '--analyze']:
            analyze_embeddings_mode = True
        elif arg in ['--compare-embeddings', '--compare', '--compare-trained-vs-fresh']:
            analyze_embeddings_mode = 'compare'
        elif arg.startswith('--model-path='):
            model_path = arg.split('=', 1)[1]
        elif arg in ['cpu', 'cuda', 'auto'] or arg.startswith('cuda:'):
            device = arg
        elif arg.isdigit():
            episodes = int(arg)
        i += 1
    
    # Check if we should run embedding analysis instead of training
    if analyze_embeddings_mode == True:
        print("🔬 Running embedding analysis mode")
        run_embedding_analysis_on_trained_model(model_path, device)
        return
    elif analyze_embeddings_mode == 'compare':
        print("⚖️  Running embedding comparison mode")
        compare_trained_vs_untrained_embeddings(model_path, device)
        return
    
    try:
        # Run PPO training with environment parameters
        rewards = train_gnn_with_ppo(
            episodes=episodes,
            device=device,
            save_path='ppo_trust_gnn.pth',
            enable_visualization=enable_visualization,
            # Environment parameters
            num_robots=num_robots,
            num_targets=num_targets,
            adversarial_ratio=adversarial_ratio,
            world_size=world_size,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            movement_speed=movement_speed,
            proximal_range=proximal_range,
            fov_range=fov_range,
            fov_angle=fov_angle
        )
        
        # Plot results
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average
        window_size = 50
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg)
            plt.title(f'Moving Average ({window_size} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training results saved to: ppo_training_results.png")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()