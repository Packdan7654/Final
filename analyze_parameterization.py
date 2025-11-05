"""
Parameterization Analysis Script

Run this script after training to analyze reward weights and generate recommendations
for tuning RL parameters.

Usage:
    python analyze_parameterization.py <experiment_number>
    python analyze_parameterization.py 1  # Analyze experiment 1
"""

import sys
from pathlib import Path
from src.utils.parameterization_analyzer import ParameterizationAnalyzer


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_parameterization.py <experiment_number>")
        print("Example: python analyze_parameterization.py 1")
        sys.exit(1)
    
    exp_num = sys.argv[1]
    
    # Find experiment directory
    exp_dir = Path(f"training_logs/experiments/exp_{int(exp_num):03d}_*")
    matching_dirs = list(Path("training_logs/experiments").glob(f"exp_{int(exp_num):03d}_*"))
    
    if not matching_dirs:
        print(f"❌ Experiment {exp_num} not found!")
        print(f"   Looking in: training_logs/experiments/")
        sys.exit(1)
    
    experiment_dir = matching_dirs[0]
    print(f"📊 Analyzing experiment: {experiment_dir.name}")
    print()
    
    # Run analysis
    analyzer = ParameterizationAnalyzer(experiment_dir)
    report = analyzer.generate_full_report()
    
    print("✅ Analysis complete!")
    print(f"   Results saved to: {experiment_dir / 'parameterization_results'}")
    print(f"   - parameterization_report.json (detailed data)")
    print(f"   - parameterization_summary.txt (readable summary)")
    print()
    
    # Print summary
    summary_file = experiment_dir / "parameterization_results" / "parameterization_summary.txt"
    if summary_file.exists():
        print("=" * 80)
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print("=" * 80)


if __name__ == "__main__":
    main()
