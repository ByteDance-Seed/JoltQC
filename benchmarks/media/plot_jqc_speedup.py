#!/usr/bin/env python3
"""
Generate speedup comparison figure between JQC and GPU4PySCF for wb97m-v calculations.
Uses only data from the media folder and displays chemical formulas.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import re


def load_benchmark_data(json_file):
    """Load benchmark data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def extract_chemical_formula(xyz_filename):
    """Extract chemical formula from XYZ file and convert to LaTeX format."""
    try:
        xyz_path = f"../molecules/{xyz_filename}"
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
            # Look for chemical formula in the header (line 2)
            if len(lines) >= 2:
                header = lines[1]
                # Extract formula from patterns like "name=C11H9ClFN5O2_..."
                match = re.search(r'name=([A-Za-z0-9]+)', header)
                if match:
                    formula = match.group(1)
                    # Clean up formula by removing numbers after underscores
                    formula = formula.split('_')[0]
                    # Convert to LaTeX format with subscripts
                    latex_formula = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1$_{\2}$', formula)
                    return latex_formula
        # Fallback: try to extract from filename
        return xyz_filename.replace('.xyz', '').replace('-', ' ')
    except Exception:
        # Fallback: use filename
        return xyz_filename.replace('.xyz', '').replace('-', ' ')


def extract_timing_data(benchmark_data):
    """Extract molecule names and timing data from benchmark results."""
    molecules = []
    formulas = []
    wall_times = []
    gpu_times = []

    for mol_data in benchmark_data['molecules']:
        if mol_data.get('success', False):
            xyz_filename = mol_data['molecule']
            formula = extract_chemical_formula(xyz_filename)

            molecules.append(xyz_filename)
            formulas.append(formula)
            wall_times.append(mol_data['wall_time_s'])
            gpu_times.append(mol_data['gpu_time_ms'] / 1000.0)  # Convert to seconds

    return molecules, formulas, wall_times, gpu_times


def create_speedup_plot():
    """Create speedup comparison plot using media folder data only."""

    # Load data files from media folder
    jqc_file = "benchmark_wb97mv_def2-tzvpd_jqc_20250918_124436.json"
    gpu4pyscf_file = "benchmark_wb97mv_def2-tzvpd_20250911_091723.json"

    jqc_data = load_benchmark_data(jqc_file)
    gpu4pyscf_data = load_benchmark_data(gpu4pyscf_file)

    # Extract timing data with chemical formulas
    jqc_molecules, jqc_formulas, jqc_wall, jqc_gpu = extract_timing_data(jqc_data)
    gpu4pyscf_molecules, gpu4pyscf_formulas, gpu4pyscf_wall, gpu4pyscf_gpu = extract_timing_data(gpu4pyscf_data)

    # Ensure molecules match between datasets
    assert jqc_molecules == gpu4pyscf_molecules, "Molecule lists don't match between datasets"
    assert jqc_formulas == gpu4pyscf_formulas, "Formula lists don't match between datasets"

    # Extract energy data for comparison
    jqc_energies = []
    gpu4pyscf_energies = []
    for mol_data in jqc_data['molecules']:
        if mol_data.get('success', False):
            jqc_energies.append(mol_data['energy'])
    for mol_data in gpu4pyscf_data['molecules']:
        if mol_data.get('success', False):
            gpu4pyscf_energies.append(mol_data['energy'])

    # Calculate energy differences (in millihartree for better readability)
    energy_diff_mh = np.array([(jqc - gpu4pyscf) * 1000 for jqc, gpu4pyscf in zip(jqc_energies, gpu4pyscf_energies)])

    # Calculate speedups
    wall_speedup = np.array(gpu4pyscf_wall) / np.array(jqc_wall)

    # Create figure with single panel and dual y-axes
    plt.rc('text', usetex=False)  # Use mathtext instead of full LaTeX for better compatibility
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Set explicit limits to prevent rendering issues
    plt.rcParams['figure.max_open_warning'] = 0

    x_pos = np.arange(len(jqc_formulas))
    width = 0.25

    # ===== LEFT Y-AXIS: TIMING COMPARISON =====
    # Create bars for timing (narrower to make room for energy bars)
    ax1.bar(x_pos - width*1.5, gpu4pyscf_wall, width,
            label='GPU4PySCF', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos - width*0.5, jqc_wall, width,
            label='JoltQC', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize left y-axis (timing)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=18, fontweight='bold', color='black')
    ax1.set_xlabel('Molecular Formula', fontsize=18, fontweight='bold')
    ax1.set_title('JoltQC vs GPU4PySCF Performance & Accuracy Comparison\nwb97m-v/def2-tzvpd on RTX 5090',
                 fontsize=20, fontweight='bold', pad=25)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(jqc_formulas, fontsize=16, rotation=0, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_yscale('log')

    # Add speedup labels on timing bars (positioned optimally)
    for i, speedup in enumerate(wall_speedup):
        max_height = max(gpu4pyscf_wall[i], jqc_wall[i])
        label_height = max_height * 1.3  # Positioned above bars but not too high
        ax1.text(i - width, label_height, f'{speedup:.1f}×',
                ha='center', va='bottom', fontweight='bold',
                fontsize=14, color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95,
                         edgecolor='darkred', linewidth=1.5))

    # ===== RIGHT Y-AXIS: ABSOLUTE ENERGY DIFFERENCES =====
    ax2 = ax1.twinx()

    # Calculate absolute energy differences (add small value to avoid log(0))
    abs_energy_diff_mh = np.abs(energy_diff_mh)
    # Add a small value to prevent log(0) issues
    abs_energy_diff_mh_log = np.maximum(abs_energy_diff_mh, 1e-6)

    # Plot absolute energy differences as dots and dashed line (shifted to middle)
    ax2.plot(x_pos, abs_energy_diff_mh_log, 'o--', color='purple', linewidth=2, markersize=10,
             alpha=0.8, label='|ΔE| (JoltQC - GPU4PySCF)', markerfacecolor='purple', markeredgecolor='black')

    # Customize right y-axis (energy difference) with log scale
    ax2.set_ylabel('Absolute Energy Difference |ΔE| [mHa]', fontsize=16, fontweight='bold', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=14)
    ax2.set_yscale('log')

    # Set reasonable limits for log scale energy difference axis
    min_val = max(abs_energy_diff_mh_log.min(), 1e-6)
    max_val = max(abs_energy_diff_mh_log.max(), 1e-3)
    ax2.set_ylim(min_val * 0.1, max_val * 10)

    # Add energy difference value labels (shifted to middle, between timing bars)
    for i, abs_diff in enumerate(abs_energy_diff_mh):
        # Position at the middle of the x-axis position (between timing bars)
        x_middle = i
        # For log scale, position labels above the points
        log_height = abs_energy_diff_mh_log[i] * 2

        # Show actual value in scientific notation
        display_value = abs_diff if abs_diff > 1e-6 else abs_diff
        ax2.text(x_middle, log_height, f'{display_value:.2e}',
                ha='center', va='bottom', fontweight='bold', fontsize=11,
                color='purple', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # ===== COMBINED LEGEND =====
    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Create combined legend with better positioning
    ax1.legend(handles1 + handles2, labels1 + labels2,
              fontsize=12, loc='upper left', frameon=True,
              fancybox=True, shadow=True, framealpha=0.9,
              bbox_to_anchor=(0.02, 0.98))

    # Calculate speedup stats for console output only
    avg_speedup = np.mean(wall_speedup)
    total_gpu4pyscf_time = sum(gpu4pyscf_wall)
    total_jqc_time = sum(jqc_wall)
    total_speedup = total_gpu4pyscf_time / total_jqc_time

    # Adjust layout manually to avoid tight_layout issues and prevent title cropping
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

    # Save figure with explicit DPI and format
    output_file = 'jqc_vs_gpu4pyscf_speedup.png'
    plt.savefig(output_file, dpi=150, format='png', facecolor='white', edgecolor='none')
    print(f"Speedup plot saved to: {output_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE & ACCURACY ANALYSIS - JoltQC vs GPU4PySCF")
    print("="*80)
    print(f"{'Chemical Formula':<25} {'GPU4PySCF (s)':<12} {'JoltQC (s)':<10} {'Speedup':<10} {'ΔE (mHa)':<10}")
    print("-"*80)

    for i, formula in enumerate(jqc_formulas):
        print(f"{formula:<25} {gpu4pyscf_wall[i]:<12.1f} {jqc_wall[i]:<10.1f} {wall_speedup[i]:<10.1f}× {energy_diff_mh[i]:<10.2f}")

    print("-"*80)
    print(f"{'AVERAGE':<25} {np.mean(gpu4pyscf_wall):<12.1f} {np.mean(jqc_wall):<10.1f} {avg_speedup:<10.1f}× {np.mean(np.abs(energy_diff_mh)):<10.2f}")
    print(f"{'TOTAL':<25} {total_gpu4pyscf_time:<12.1f} {total_jqc_time:<10.1f} {total_speedup:<10.1f}× {np.max(np.abs(energy_diff_mh)):<10.2f}")

    print(f"\n{'='*80}")
    print("INDIVIDUAL SPEEDUP & ENERGY DIFFERENCE BREAKDOWN:")
    for i, (formula, speedup, energy_diff) in enumerate(zip(jqc_formulas, wall_speedup, energy_diff_mh)):
        print(f"  {formula}: {speedup:.1f}× faster, ΔE = {energy_diff:.2f} mHa")

    print(f"\n{'='*80}")
    print("ENERGY ACCURACY SUMMARY:")
    print(f"  Mean absolute energy difference: {np.mean(np.abs(energy_diff_mh)):.2f} mHa")
    print(f"  Max absolute energy difference:  {np.max(np.abs(energy_diff_mh)):.2f} mHa")
    print(f"  RMS energy difference:           {np.sqrt(np.mean(energy_diff_mh**2)):.2f} mHa")

    # Show plot (will be ignored in headless mode)
    plt.show()

    return wall_speedup, avg_speedup, jqc_formulas, energy_diff_mh


if __name__ == "__main__":
    try:
        speedups, avg_speedup, formulas, energy_diffs = create_speedup_plot()
        print(f"\n✓ Successfully generated performance & accuracy comparison plot")
        print(f"✓ Average speedup: {avg_speedup:.1f}× improvement")
        print(f"✓ Average energy difference: {np.mean(np.abs(energy_diffs)):.2f} mHa")
        print(f"✓ Analyzed {len(formulas)} molecules with chemical formulas")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()