"""
Pre-Overnight Run Checklist

Verifies all components are ready for an overnight training run.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_scripts():
    """Check all training scripts exist."""
    scripts = {
        'baseline': 'train.py',
        'h1': 'experiments/h1_option_structure/train.py',
        'h3': 'experiments/h3_prompt_headers/train.py',
        'h5': 'experiments/h5_state_ablation/train.py',
        'h6': 'experiments/h6_transition_reward/train.py',
        'h7': 'experiments/h7_hybrid_bert/train.py',
    }
    
    print("=" * 80)
    print("CHECKING TRAINING SCRIPTS")
    print("=" * 80)
    
    all_exist = True
    for name, script_path in scripts.items():
        path = project_root / script_path
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {script_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_directories():
    """Check required directories exist."""
    dirs = [
        'major_results',
        'training_logs',
        'src',
        'experiments',
    ]
    
    print("\n" + "=" * 80)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 80)
    
    all_exist = True
    for dir_name in dirs:
        path = project_root / dir_name
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {dir_name}/")
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check critical imports work."""
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('transformers', 'Transformers'),
        ('gymnasium', 'Gymnasium'),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_ok = False
    
    return all_ok

def check_major_results_setup():
    """Check major_results manager works."""
    print("\n" + "=" * 80)
    print("CHECKING major_results SETUP")
    print("=" * 80)
    
    try:
        from src.utils.major_results_manager import MajorResultsManager
        manager = MajorResultsManager()
        base_dir = manager.base_dir
        
        if base_dir.exists():
            print(f"  [OK] major_results/ directory exists")
            print(f"  [OK] Path: {base_dir}")
        else:
            print(f"  [WARNING] major_results/ directory does not exist (will be created)")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to import MajorResultsManager: {e}")
        return False

def check_checkpoint_saving():
    """Check checkpoint saving is configured."""
    print("\n" + "=" * 80)
    print("CHECKING CHECKPOINT CONFIGURATION")
    print("=" * 80)
    
    try:
        # Check if training loop has checkpoint saving
        import inspect
        from src.training.training_loop import HRLTrainingLoop
        
        # Check for checkpoint methods
        has_save = hasattr(HRLTrainingLoop, '_save_checkpoint')
        has_load = hasattr(HRLTrainingLoop, '_load_checkpoint')
        
        print(f"  [OK] Checkpoint saving: {has_save}")
        print(f"  [INFO] Checkpoint loading: {has_load} (optional for resuming)")
        
        # Checkpoint saving is required, loading is optional
        return has_save
    except Exception as e:
        print(f"  [ERROR] Failed to check checkpoint configuration: {e}")
        return False

def check_error_handling():
    """Check error handling in train_all_variations."""
    print("\n" + "=" * 80)
    print("CHECKING ERROR HANDLING")
    print("=" * 80)
    
    script_path = project_root / 'train_all_variations.py'
    if not script_path.exists():
        print(f"  [ERROR] train_all_variations.py not found")
        return False
    
    content = script_path.read_text()
    
    checks = {
        'try/except blocks': 'except' in content,
        'continues on failure': 'Continuing with next variation' in content,
        'KeyboardInterrupt handling': 'KeyboardInterrupt' in content,
    }
    
    all_ok = True
    for check, result in checks.items():
        status = "[OK]" if result else "[MISSING]"
        print(f"  {status} {check}")
        if not result:
            all_ok = False
    
    return all_ok

def main():
    """Run all checks."""
    print("\n" + "=" * 80)
    print("PRE-OVERNIGHT RUN CHECKLIST")
    print("=" * 80)
    print()
    
    results = {
        'Scripts': check_scripts(),
        'Directories': check_directories(),
        'Dependencies': check_dependencies(),
        'major_results Setup': check_major_results_setup(),
        'Checkpoint Saving': check_checkpoint_saving(),
        'Error Handling': check_error_handling(),
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for overnight run!")
        print("\nRecommended command:")
        print("  python train_all_variations.py --episodes 600 --map-interval 10 --device cuda")
        print("\nTo run without confirmation prompt, use:")
        print("  echo y | python train_all_variations.py --episodes 600 --map-interval 10 --device cuda")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before running overnight")
    
    print()

if __name__ == '__main__':
    main()

