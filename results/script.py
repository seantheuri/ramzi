import pandas as pd
import numpy as np
import csv

def calculate_metrics(df):
    """Calculates a dictionary of metrics from a DataFrame of experimental results."""
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # --- General Metrics ---
    # Calculate the percentage of scripts that used EQ or Filter commands
    # We check if the string 'eq' or 'filter' appears in the technique_highlights column
    df['used_eq_filter'] = df['technique_highlights'].str.contains('eq', case=False) | \
                           df['technique_highlights'].str.contains('filter', case=False)
    
    # Calculate the percentage of scripts that used Loops or FX
    df['used_loop_fx'] = df['technique_highlights'].str.contains('loop', case=False) | \
                         df['technique_highlights'].str.contains('fx', case=False) | \
                         df['technique_highlights'].str.contains('repeat', case=False)

    # --- Metric Calculations ---
    avg_command_count = df['command_count'].mean()
    percent_eq_filter = (df['used_eq_filter'].sum() / len(df)) * 100
    percent_loop_fx = (df['used_loop_fx'].sum() / len(df)) * 100
    
    return {
        "Avg Command Count": f"{avg_command_count:.1f}",
        "% Scripts w/ EQ/Filter": f"{percent_eq_filter:.1f}%",
        "% Scripts w/ Loops/FX": f"{percent_loop_fx:.1f}%"
    }


def analyze_ablation_study(df):
    """Analyzes the ablation study results."""
    
    # Filter for the ablation experiment
    ablation_df = df[df['experiment_name'] == 'exp3_ablation'].copy()
    
    # Define what constitutes a "good" structural mix point
    good_mix_points = ['intro', 'outro', 'breakdown']
    
    # Calculate if a 'good' structural point was used for mixing in or out
    ablation_df['used_good_structure'] = ablation_df['mix_out_label'].isin(good_mix_points) | \
                                         ablation_df['mix_in_label'].isin(good_mix_points)
    
    # Group by the ablation condition and calculate metrics
    results = ablation_df.groupby('ablation_condition').agg(
        avg_command_count=('command_count', 'mean'),
        percent_good_structure=('used_good_structure', lambda x: (x.sum() / len(x)) * 100)
    ).reindex(['full_data', 'no_lyrics', 'no_structure']) # Ensure correct order

    return results


def analyze_human_likeness(df_exp1, df_exp2):
    """Analyzes metrics to compare against human DJ statistics."""
    
    # Combine all valid GPT-4o runs from all experiments for a robust sample
    gpt4o_df = pd.concat([
        df_exp1[df_exp1['model_name'] == 'gpt-4o'],
        df_exp2[df_exp2['model_name'] == 'gpt-4o']
    ]).copy()
    
    # Clean the transition_length_beats column, converting non-numeric to NaN and then dropping them
    gpt4o_df['transition_length_beats'] = pd.to_numeric(gpt4o_df['transition_length_beats'], errors='coerce')
    gpt4o_df.dropna(subset=['transition_length_beats'], inplace=True)
    
    # --- Metric 1: Transition Length ---
    avg_transition_length = gpt4o_df['transition_length_beats'].mean()
    
    # --- Metric 2: Key Transposition Rate ---
    total_mixes = len(gpt4o_df)
    mixes_with_key_shift = gpt4o_df[gpt4o_df['key_shift_count'] > 0].shape[0]
    key_shift_rate = (mixes_with_key_shift / total_mixes) * 100 if total_mixes > 0 else 0
    
    # --- Metric 3: Structural Cue Usage ---
    # Let's define a "logical" transition as one that uses an intro, outro, or breakdown
    good_mix_points = ['intro', 'outro', 'breakdown']
    gpt4o_df['used_good_structure'] = gpt4o_df['mix_out_label'].isin(good_mix_points) | \
                                      gpt4o_df['mix_in_label'].isin(good_mix_points)
    structural_cue_usage_rate = (gpt4o_df['used_good_structure'].sum() / total_mixes) * 100 if total_mixes > 0 else 0

    return {
        "Avg Transition Length (beats)": f"{avg_transition_length:.1f}",
        "Key Transposition Rate": f"{key_shift_rate:.1f}%",
        "Structural Cue Usage Rate": f"{structural_cue_usage_rate:.1f}%"
    }


def main():
    """Main function to load data and print all report tables."""

    try:
        # --- Load Data from separate CSV files ---
        print("Loading data from experimental_results.csv (Exp1)...")
        df_exp1 = pd.read_csv('experimental_results.csv')

        print("Loading data from experimental_results2.csv (Exp2)...")
        # Now that the header is in the file, we can read it directly.
        # Using engine='python' and quoting helps with complex, quoted text fields.
        df_exp2 = pd.read_csv('experimental_results2.csv', sep=',', engine='python', quoting=csv.QUOTE_ALL)

        print("Loading data from experimental_results3.csv (Exp3)...")
        df_exp3 = pd.read_csv('experimental_results3.csv')
        
        # --- Data Cleaning and Type Conversion ---
        print("Cleaning and converting data types...")
        numeric_cols = ['command_count', 'harmonic_score', 'key_shift_count', 'bpm_change_a_percent', 'bpm_change_b_percent']
        for col in numeric_cols:
            if col in df_exp1.columns: df_exp1[col] = pd.to_numeric(df_exp1[col], errors='coerce')
            if col in df_exp2.columns: df_exp2[col] = pd.to_numeric(df_exp2[col], errors='coerce')
            if col in df_exp3.columns: df_exp3[col] = pd.to_numeric(df_exp3[col], errors='coerce')

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure experimental_results.csv, experimental_results2.csv, and experimental_results3.csv are present.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    print("="*60)
    print("ANALYSIS FOR RESEARCH PAPER")
    print("="*60)

    # --- Table 1: Model Reasoning and Prompting Style Comparison ---
    print("\n--- TABLE 1: Model Reasoning & Prompting Style ---")
    
    # We only need data from Experiment 1 for this table
    gpt4o_direct = calculate_metrics(df_exp1[(df_exp1['model_name'] == 'gpt-4o') & (df_exp1['prompting_style'] == 'direct')])
    gpt4o_cot = calculate_metrics(df_exp1[(df_exp1['model_name'] == 'gpt-4o') & (df_exp1['prompting_style'] == 'cot')])
    
    print(f"\nGPT-4o (Direct):")
    for key, value in gpt4o_direct.items():
        print(f"  {key}: {value}")
        
    print(f"\nGPT-4o (CoT):")
    for key, value in gpt4o_cot.items():
        print(f"  {key}: {value}")
    
    # --- Table 2: Ablation Study on Input Data ---
    print("\n\n--- TABLE 2: Ablation Study on Input Data ---")
    # This uses the df_exp3 DataFrame, which is now clean
    ablation_results = analyze_ablation_study(df_exp3)
    
    print(ablation_results.to_string())

    # --- Table for Section 5.4: Benchmarking Against Human DJ Practices ---
    print("\n\n--- BENCHMARKING: Comparison with Human DJ Practices ---")
    
    # This uses data from all valid GPT-4o runs from Exp1 and Exp2
    human_likeness_stats = analyze_human_likeness(df_exp1, df_exp2)

    print("\nRAMZI System (GPT-4o) Metrics:")
    for key, value in human_likeness_stats.items():
        print(f"  {key}: {value}")
    
    print("\nHuman DJ Benchmark (Kim et al., 2020):")
    print("  Avg Transition Length (beats): Peaks around 32, 64")
    print("  Key Transposition Rate: ~2.5%")
    print("  Structural Cue Usage Rate: High (qualitative), uses intros/outros")
    print("  Tempo Change < 5%: ~86.1%")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()