#!/usr/bin/env python3
"""
Server Compatibility Check

This script verifies that all required dependencies are available
and the environment is properly set up for running the GNN training pipeline.
"""

import sys
import os
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 PYTHON VERSION CHECK")
    print("=" * 50)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_required_packages():
    """Check if all required packages are available"""
    print(f"\n🔍 PACKAGE AVAILABILITY CHECK")
    print("=" * 50)
    
    # Core packages required for our scripts
    required_packages = [
        # Core scientific computing
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        
        # PyTorch ecosystem
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('torchaudio', 'torchaudio'),
        
        # Graph neural networks
        ('torch_geometric', 'torch-geometric'),
        ('torch_scatter', 'torch-scatter'),
        ('torch_sparse', 'torch-sparse'),
        
        # Data manipulation
        ('pandas', 'pandas'),
        ('json', 'built-in'),
        ('time', 'built-in'),
        ('os', 'built-in'),
        
        # Additional utilities
        ('typing', 'built-in'),
    ]
    
    missing_packages = []
    
    for package_name, install_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (install with: pip install {install_name})")
            missing_packages.append(install_name)
    
    return len(missing_packages) == 0, missing_packages

def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    print(f"\n🔍 PYTORCH GPU CHECK")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute: {props.major}.{props.minor}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch GPU check failed: {e}")
        return False

def check_matplotlib_backend():
    """Check matplotlib backend for headless server"""
    print(f"\n🔍 MATPLOTLIB BACKEND CHECK")
    print("=" * 50)
    
    try:
        import matplotlib
        backend = matplotlib.get_backend()
        print(f"Current backend: {backend}")
        
        # Test if we can set Agg backend (required for headless servers)
        matplotlib.use('Agg')
        print("✅ Agg backend available (headless server compatible)")
        
        # Test basic plotting
        import matplotlib.pyplot as plt
        import numpy as np
        
        x = np.linspace(0, 2*np.pi, 10)
        y = np.sin(x)
        
        plt.figure()
        plt.plot(x, y)
        plt.savefig('/tmp/test_plot.png')
        plt.close()
        
        # Clean up test file
        if os.path.exists('/tmp/test_plot.png'):
            os.remove('/tmp/test_plot.png')
            print("✅ Plot generation and saving works")
        
        return True
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        return False

def check_required_files():
    """Check if required project files exist"""
    print(f"\n🔍 PROJECT FILES CHECK")
    print("=" * 50)
    
    required_files = [
        'simulation_environment.py',
        'paper_trust_algorithm.py', 
        'neural_symbolic_trust_algorithm.py',
        'gnn_training_data_collector.py',
        'train_gnn.py',
        'compare_algorithms.py'
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0, missing_files

def check_write_permissions():
    """Check if we can write files in current directory"""
    print(f"\n🔍 WRITE PERMISSIONS CHECK")
    print("=" * 50)
    
    try:
        # Test writing a temporary file
        test_file = 'test_write_permission.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("✅ Write permissions available")
        return True
    except Exception as e:
        print(f"❌ Write permission test failed: {e}")
        return False

def main():
    """Run all compatibility checks"""
    print("🚀 SERVER COMPATIBILITY CHECK")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", lambda: check_required_packages()[0]),
        ("PyTorch GPU", check_pytorch_gpu),
        ("Matplotlib Backend", check_matplotlib_backend),
        ("Required Files", lambda: check_required_files()[0]),
        ("Write Permissions", check_write_permissions)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
            results[check_name] = False
    
    # Print summary
    print(f"\n🏁 COMPATIBILITY SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Server is ready for GNN training.")
        print("\nNext steps:")
        print("1. python gnn_training_data_collector.py")
        print("2. python train_gnn.py 1  # Use GPU 1")
        print("3. python compare_algorithms.py 1")
    else:
        print("⚠️  Some checks failed. Please resolve the issues above.")
        
        # Show specific guidance for common issues
        if not results.get("Required Packages", True):
            _, missing = check_required_packages()
            print(f"\nInstall missing packages:")
            print(f"pip install {' '.join(set(missing))}")
        
        if not results.get("Required Files", True):
            _, missing = check_required_files()
            print(f"\nMissing files: {', '.join(missing)}")
            print("Make sure you're in the correct project directory.")

if __name__ == "__main__":
    main()