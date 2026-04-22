"""
Run all EDA scripts in sequence
File: scripts/run_all.py
"""

import subprocess
import os
import sys

def run_all_scripts():
    """Execute all Python scripts in order"""
    
    scripts = [
        '01_load_data.py',
        '02_inspect_data.py',
        '03_statistics.py',
        '04_univariate_analysis.py',
        '05_bivariate_analysis.py',
        '06_correlation.py',
        '07_species_comparison.py',
        '08_final_visualization.py',
        '09_findings.py'
    ]
    
    print("=" * 60)
    print("RUNNING ALL EDA SCRIPTS")
    print("=" * 60)
    
    for script in scripts:
        print(f"\n\n{'=' * 60}")
        print(f"Running: {script}")
        print('=' * 60)
        
        result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"\n❌ Error in {script}. Stopping execution.")
            break
        else:
            print(f"\n✅ Completed: {script}")
    
    print("\n" + "=" * 60)
    print("ALL SCRIPTS COMPLETED!")
    print("=" * 60)
    print("\nNow open report.html in your browser to view the complete report.")

if __name__ == "__main__":
    run_all_scripts()
    