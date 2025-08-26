#!/usr/bin/env python3
"""
GNN Model Training Script

This script trains the neural symbolic GNN model using collected training data.
"""

import torch
import time
import os
from typing import List

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt

from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm


def show_gpu_status():
    """Display status of all available GPUs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("📊 GPU STATUS:")
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
        free_gb = total_gb - allocated_gb
        
        status = "✅ Available" if free_gb > 2.0 else "⚠️  Limited" if free_gb > 1.0 else "❌ Busy"
        
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {allocated_gb:.1f}/{total_gb:.1f} GB used ({free_gb:.1f} GB free) {status}")
    print()

class GNNTrainer:
    """Simple and focused GNN trainer"""
    
    def _find_best_gpu(self):
        """Find GPU with most free memory"""
        if not torch.cuda.is_available():
            return None
        
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free_memory = props.total_memory - allocated
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        return best_gpu
    
    def __init__(self, device: str = "auto"):
        # Handle different device specifications
        if device == "auto":
            if torch.cuda.is_available():
                # Find GPU with most free memory
                best_gpu = self._find_best_gpu()
                self.device = torch.device(f'cuda:{best_gpu}')
                gpu_id = best_gpu
                print(f"🔍 Auto-selected GPU {best_gpu} (most free memory)")
            else:
                self.device = torch.device('cpu')
                gpu_id = None
        elif device == "cpu":
            self.device = torch.device('cpu')
            gpu_id = None
        elif device.startswith("cuda"):
            # Handle cuda, cuda:0, cuda:1, etc.
            self.device = torch.device(device)
            if ":" in device:
                gpu_id = int(device.split(":")[1])
            else:
                gpu_id = 0
        else:
            # Try to parse as GPU ID number
            try:
                gpu_id = int(device)
                self.device = torch.device(f'cuda:{gpu_id}')
            except ValueError:
                print(f"❌ Invalid device '{device}'. Use 'cpu', 'cuda', 'cuda:N', or GPU number")
                raise ValueError(f"Invalid device: {device}")
        
        # Verify GPU availability and set device
        if gpu_id is not None:
            if not torch.cuda.is_available():
                print(f"⚠️  CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
                gpu_id = None
            elif gpu_id >= torch.cuda.device_count():
                print(f"⚠️  GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)")
                print(f"   Falling back to CPU to avoid interfering with other GPUs")
                self.device = torch.device('cpu')
                gpu_id = None
            else:
                # Check if GPU has sufficient free memory (at least 2GB recommended)
                torch.cuda.set_device(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                free_memory = total_memory - allocated_memory
                free_memory_gb = free_memory / 1024**3
                
                if free_memory_gb < 2.0:  # Less than 2GB free
                    allocated_gb = allocated_memory / 1024**3
                    total_gb = total_memory / 1024**3
                    print(f"⚠️  GPU {gpu_id} has limited memory: {free_memory_gb:.1f} GB free ({allocated_gb:.1f}/{total_gb:.1f} GB used)")
                    print(f"   Falling back to CPU to avoid memory issues")
                    self.device = torch.device('cpu')
                    gpu_id = None
        
        # Print device information
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)  # Set current GPU
            print(f"🚀 Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
            print(f"   Memory Usage: {torch.cuda.memory_allocated(gpu_id)/1024**3:.2f} GB allocated")
        else:
            print(f"🔥 Using CPU")
        
        print(f"🔥 Training device: {self.device}")
    
    def train_model(self,
                   training_data_file: str,
                   epochs: int = 50,
                   output_model_file: str = "trained_gnn_model.pth") -> List[float]:
        """
        Train the GNN model
        
        Args:
            training_data_file: Path to training data file
            epochs: Number of training epochs
            output_model_file: Output model file path
            
        Returns:
            List of training losses
        """
        print("🎯 GNN MODEL TRAINING")
        print("=" * 50)
        
        # Check if training data exists
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Training data file not found: {training_data_file}")
        
        print(f"Training data: {training_data_file}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Note: Learning rate and batch size are handled internally by the GNN trainer")
        
        # Initialize neural symbolic algorithm
        neural_algorithm = NeuralSymbolicTrustAlgorithm(learning_mode=True)
        
        # Move model to device and optimize for GPU if available
        neural_algorithm.model = neural_algorithm.model.to(self.device)
        
        if self.device.type == 'cuda':
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            print(f"   🔧 Enabled cuDNN benchmark for faster training")
        
        # Re-create trainer with correct device after moving model
        from neural_symbolic_trust_algorithm import GNNTrainer as Trainer
        neural_algorithm.trainer = Trainer(neural_algorithm.model, device=self.device)
        
        # Get model info
        model_info = neural_algorithm.get_model_info()
        print(f"\n🔧 Model Information:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Parameters: {model_info['num_parameters']:,}")
        
        # Train the model
        print(f"\n🚀 Starting training...")
        
        # Monitor GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear cache before training
            print(f"   💾 GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
        
        start_time = time.time()
        
        losses = neural_algorithm.train_model(
            training_data_file=training_data_file,
            epochs=epochs
        )
        
        training_time = time.time() - start_time
        
        # Report GPU memory usage after training
        if self.device.type == 'cuda':
            print(f"   💾 GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
            print(f"   💾 Peak GPU memory usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
        # Save trained model
        neural_algorithm.save_model(output_model_file)
        
        print(f"\n✅ Training completed!")
        print(f"   ⏱️  Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        print(f"   📉 Initial loss: {losses[0]:.6f}")
        print(f"   📉 Final loss: {losses[-1]:.6f}")
        print(f"   📈 Improvement: {((losses[0] - losses[-1])/losses[0]*100):.1f}%")
        print(f"   💾 Model saved to: {output_model_file}")
        
        # Create training curve visualization
        self._plot_training_curve(losses, output_model_file)
        
        return losses
    
    def _plot_training_curve(self, losses: List[float], model_file: str):
        """Create training curve visualization"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2, alpha=0.8)
        plt.title('GNN Training Loss Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add annotations
        plt.annotate(f'Initial: {losses[0]:.6f}', 
                    xy=(0, losses[0]), 
                    xytext=(len(losses)*0.1, losses[0]*2),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')
        
        plt.annotate(f'Final: {losses[-1]:.6f}', 
                    xy=(len(losses)-1, losses[-1]), 
                    xytext=(len(losses)*0.6, losses[-1]*3),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                    fontsize=10, color='green')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = model_file.replace('.pth', '_training_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close figure to free memory
        
        print(f"   📊 Training curve saved to: {plot_file}")


def main():
    """Main training function"""
    import sys
    
    # Check for device argument
    device = "auto"
    if len(sys.argv) > 1:
        device_arg = sys.argv[1]
        
        # Special case: show GPU status
        if device_arg == "status":
            show_gpu_status()
            return
        
        # Valid device options: cpu, cuda, cuda:N, auto, or just GPU number
        valid_devices = ["cpu", "cuda", "auto"] + [f"cuda:{i}" for i in range(8)] + [str(i) for i in range(8)]
        
        if device_arg in valid_devices or device_arg.startswith("cuda:"):
            device = device_arg
        else:
            print(f"❌ Invalid device '{device_arg}'")
            print(f"Valid options:")
            print(f"  'auto' - Auto-detect best GPU")
            print(f"  'cpu'  - Force CPU usage")  
            print(f"  'cuda' - Use GPU 0")
            print(f"  'cuda:1' - Use specific GPU (cuda:0, cuda:1, etc.)")
            print(f"  '1' - Use GPU by number (0, 1, 2, 3)")
            print(f"  'status' - Show GPU status")
            print()
            show_gpu_status()
            return
    
    try:
        trainer = GNNTrainer(device=device)
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        print()
        show_gpu_status()
        return
    
    # Default training parameters
    training_data_file = "gnn_training_data.pkl"
    
    # Check if training data exists
    if not os.path.exists(training_data_file):
        print(f"❌ Training data not found: {training_data_file}")
        print(f"Run 'python gnn_training_data_collector.py' first to collect training data")
        return
    
    try:
        # Train the model
        losses = trainer.train_model(
            training_data_file=training_data_file,
            epochs=50,
            output_model_file="trained_gnn_model.pth"
        )
        
        print(f"\n🎉 GNN model training completed successfully!")
        print(f"Use 'python compare_algorithms.py' to compare with paper algorithm")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"Please check your training data and try again")


if __name__ == "__main__":
    main()