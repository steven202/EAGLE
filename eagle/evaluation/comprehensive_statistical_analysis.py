#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for EAGLE Speed Comparisons

This script performs advanced statistical analysis across multiple benchmarks
and methods, including:
- Multiple comparison corrections (Bonferroni, FDR)
- Effect size analysis across benchmarks
- Meta-analysis of results
- Visualization of statistical results
- Comprehensive reporting

Usage:
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir log/20250727_0412_optimized_ppo \
        --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir statistical_analysis_results
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats.mstats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveStatisticalAnalyzer:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.results = defaultdict(dict)
        self.comparisons = []
        
    def load_speed_data(self, file_path):
        """Load speed data from a JSONL file."""
        speeds = []
        question_ids = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                qid = json_obj["question_id"]
                tokens = sum(json_obj["choices"][0]['new_tokens'])
                times = sum(json_obj["choices"][0]['wall_time'])
                speed = tokens / times if times > 0 else 0
                speeds.append(speed)
                question_ids.append(qid)
        
        return np.array(speeds), question_ids
    
    def calculate_effect_size(self, method1_speeds, method2_speeds):
        """Calculate Cohen's d effect size."""
        differences = method1_speeds - method2_speeds
        mean_diff = np.mean(differences)
        pooled_std = np.sqrt((np.var(method1_speeds) + np.var(method2_speeds)) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def perform_single_comparison(self, method1_file, method2_file, method1_name, method2_name, benchmark):
        """Perform statistical comparison between two methods."""
        try:
            # Load data
            speeds1, ids1 = self.load_speed_data(method1_file)
            speeds2, ids2 = self.load_speed_data(method2_file)
            
            # Match question IDs
            dict1 = dict(zip(ids1, speeds1))
            dict2 = dict(zip(ids2, speeds2))
            common_ids = sorted(set(ids1) & set(ids2))
            
            if len(common_ids) == 0:
                return None
            
            # Extract matched speeds
            matched_speeds1 = np.array([dict1[qid] for qid in common_ids])
            matched_speeds2 = np.array([dict2[qid] for qid in common_ids])
            
            # Basic statistics
            mean1, mean2 = np.mean(matched_speeds1), np.mean(matched_speeds2)
            std1, std2 = np.std(matched_speeds1, ddof=1), np.std(matched_speeds2, ddof=1)
            speed_ratio = mean1 / mean2
            
            # Statistical tests
            t_stat, p_value_t = ttest_rel(matched_speeds1, matched_speeds2)
            
            try:
                w_stat, p_value_w = wilcoxon(matched_speeds1, matched_speeds2, alternative='two-sided')
                wilcoxon_success = True
            except ValueError:
                w_stat, p_value_w = np.nan, np.nan
                wilcoxon_success = False
            
            # Effect size
            cohens_d = self.calculate_effect_size(matched_speeds1, matched_speeds2)
            
            # Confidence interval
            differences = matched_speeds1 - matched_speeds2
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - t_critical * se
            ci_upper = mean_diff + t_critical * se
            
            return {
                'benchmark': benchmark,
                'method1': method1_name,
                'method2': method2_name,
                'n_samples': n,
                'method1_mean': mean1,
                'method2_mean': mean2,
                'method1_std': std1,
                'method2_std': std2,
                'speed_ratio': speed_ratio,
                't_statistic': t_stat,
                't_p_value': p_value_t,
                'wilcoxon_statistic': w_stat if wilcoxon_success else np.nan,
                'wilcoxon_p_value': p_value_w if wilcoxon_success else np.nan,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_difference': mean_diff,
                'significant_t_test': p_value_t < 0.05,
                'significant_wilcoxon': p_value_w < 0.05 if wilcoxon_success else False
            }
            
        except Exception as e:
            print(f"Error comparing {method1_name} vs {method2_name} on {benchmark}: {e}")
            return None
    
    def analyze_results_directory(self, results_dir, benchmarks):
        """Analyze all results in a directory."""
        results_dir = Path(results_dir)
        
        # Find all policy directories
        policy_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('optimized_')]
        
        for policy_dir in policy_dirs:
            policy_name = policy_dir.name
            print(f"Analyzing policy: {policy_name}")
            
            for benchmark in benchmarks:
                # Define file paths
                policy_file = policy_dir / "evaluation" / f"{benchmark}_results.jsonl"
                eagle3_file = policy_dir / "baseline_results" / f"{benchmark}_LLaMA3.1-8B_eagle3.jsonl"
                baseline_file = policy_dir / "baseline_results" / f"{benchmark}_LLaMA3.1-8B_baseline.jsonl"
                
                # Check if files exist
                if not policy_file.exists():
                    print(f"  Warning: {policy_file} not found")
                    continue
                
                # Perform comparisons
                comparisons = []
                
                # Policy vs EAGLE3
                if eagle3_file.exists():
                    comp = self.perform_single_comparison(
                        policy_file, eagle3_file, 
                        policy_name, "EAGLE3", benchmark
                    )
                    if comp:
                        comparisons.append(comp)
                
                # Policy vs Baseline
                if baseline_file.exists():
                    comp = self.perform_single_comparison(
                        policy_file, baseline_file, 
                        policy_name, "Baseline", benchmark
                    )
                    if comp:
                        comparisons.append(comp)
                
                # EAGLE3 vs Baseline
                if eagle3_file.exists() and baseline_file.exists():
                    comp = self.perform_single_comparison(
                        eagle3_file, baseline_file, 
                        "EAGLE3", "Baseline", benchmark
                    )
                    if comp:
                        comparisons.append(comp)
                
                # Store results
                self.results[policy_name][benchmark] = comparisons
    
    def apply_multiple_comparison_corrections(self):
        """Apply multiple comparison corrections."""
        # Collect all p-values
        all_p_values = []
        comparison_info = []
        
        for policy_name, benchmark_results in self.results.items():
            for benchmark, comparisons in benchmark_results.items():
                for comp in comparisons:
                    all_p_values.append(comp['t_p_value'])
                    comparison_info.append({
                        'policy': policy_name,
                        'benchmark': benchmark,
                        'method1': comp['method1'],
                        'method2': comp['method2']
                    })
        
        if not all_p_values:
            return
        
        # Apply corrections
        p_values = np.array(all_p_values)
        
        # Bonferroni correction
        bonferroni_p_values = p_values * len(p_values)
        bonferroni_p_values = np.minimum(bonferroni_p_values, 1.0)
        
        # FDR correction (Benjamini-Hochberg)
        sorted_indices = np.argsort(p_values)
        fdr_p_values = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            fdr_p_values[idx] = p_values[idx] * len(p_values) / (i + 1)
        fdr_p_values = np.minimum(fdr_p_values, 1.0)
        
        # Store corrected p-values
        for i, (policy_name, benchmark_results) in enumerate(self.results.items()):
            for benchmark, comparisons in benchmark_results.items():
                for j, comp in enumerate(comparisons):
                    comp_idx = i * len(benchmark_results) + j
                    if comp_idx < len(bonferroni_p_values):
                        comp['bonferroni_p_value'] = bonferroni_p_values[comp_idx]
                        comp['fdr_p_value'] = fdr_p_values[comp_idx]
                        comp['significant_bonferroni'] = bonferroni_p_values[comp_idx] < 0.05
                        comp['significant_fdr'] = fdr_p_values[comp_idx] < 0.05
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Overall statistics
        total_comparisons = sum(len(comps) for comps in self.results.values() 
                              for comps in comps.values())
        significant_comparisons = sum(
            sum(1 for comp in comps if comp.get('significant_t_test', False))
            for comps in self.results.values() 
            for comps in comps.values()
        )
        
        report.append(f"Total comparisons analyzed: {total_comparisons}")
        report.append(f"Significant comparisons (p < 0.05): {significant_comparisons}")
        report.append(f"Significance rate: {significant_comparisons/total_comparisons*100:.1f}%")
        report.append("")
        
        # Results by policy
        for policy_name, benchmark_results in self.results.items():
            report.append(f"POLICY: {policy_name}")
            report.append("-" * 50)
            
            policy_significant = 0
            policy_total = 0
            
            for benchmark, comparisons in benchmark_results.items():
                report.append(f"  Benchmark: {benchmark}")
                
                for comp in comparisons:
                    policy_total += 1
                    if comp.get('significant_t_test', False):
                        policy_significant += 1
                    
                    report.append(f"    {comp['method1']} vs {comp['method2']}:")
                    report.append(f"      Speed ratio: {comp['speed_ratio']:.4f}x")
                    report.append(f"      p-value: {comp['t_p_value']:.6f}")
                    report.append(f"      Effect size (Cohen's d): {comp['cohens_d']:.4f}")
                    report.append(f"      Significant: {'YES' if comp.get('significant_t_test', False) else 'NO'}")
                    
                    if 'bonferroni_p_value' in comp:
                        report.append(f"      Bonferroni p-value: {comp['bonferroni_p_value']:.6f}")
                        report.append(f"      FDR p-value: {comp['fdr_p_value']:.6f}")
                    
                    report.append("")
            
            if policy_total > 0:
                report.append(f"  Policy significance rate: {policy_significant/policy_total*100:.1f}%")
            report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir):
        """Create visualizations of the statistical results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        for policy_name, benchmark_results in self.results.items():
            for benchmark, comparisons in benchmark_results.items():
                for comp in comparisons:
                    plot_data.append({
                        'Policy': policy_name,
                        'Benchmark': benchmark,
                        'Comparison': f"{comp['method1']} vs {comp['method2']}",
                        'Speed Ratio': comp['speed_ratio'],
                        'P-value': comp['t_p_value'],
                        'Effect Size': comp['cohens_d'],
                        'Significant': comp.get('significant_t_test', False)
                    })
        
        if not plot_data:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Speed ratio distribution
        ax1 = axes[0, 0]
        significant_data = df[df['Significant']]
        non_significant_data = df[~df['Significant']]
        
        if len(significant_data) > 0:
            ax1.hist(significant_data['Speed Ratio'], alpha=0.7, label='Significant', bins=10)
        if len(non_significant_data) > 0:
            ax1.hist(non_significant_data['Speed Ratio'], alpha=0.7, label='Non-significant', bins=10)
        
        ax1.set_xlabel('Speed Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Speed Ratios')
        ax1.legend()
        ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        # 2. P-value distribution
        ax2 = axes[0, 1]
        ax2.hist(df['P-value'], bins=20, alpha=0.7)
        ax2.set_xlabel('P-value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of P-values')
        ax2.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax2.legend()
        
        # 3. Effect size distribution
        ax3 = axes[1, 0]
        ax3.hist(df['Effect Size'], bins=15, alpha=0.7)
        ax3.set_xlabel("Cohen's d")
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Effect Sizes')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 4. Speed ratio by benchmark
        ax4 = axes[1, 1]
        benchmark_means = df.groupby('Benchmark')['Speed Ratio'].mean()
        benchmark_means.plot(kind='bar', ax=ax4)
        ax4.set_xlabel('Benchmark')
        ax4.set_ylabel('Mean Speed Ratio')
        ax4.set_title('Mean Speed Ratio by Benchmark')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir / 'statistical_analysis_plots.png'}")
    
    def save_detailed_results(self, output_dir):
        """Save detailed results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        for policy_name, benchmark_results in self.results.items():
            for benchmark, comparisons in benchmark_results.items():
                for comp in comparisons:
                    row = {
                        'Policy': policy_name,
                        'Benchmark': benchmark,
                        'Method1': comp['method1'],
                        'Method2': comp['method2'],
                        'N_Samples': comp['n_samples'],
                        'Method1_Mean': comp['method1_mean'],
                        'Method2_Mean': comp['method2_mean'],
                        'Speed_Ratio': comp['speed_ratio'],
                        'T_Statistic': comp['t_statistic'],
                        'T_P_Value': comp['t_p_value'],
                        'Wilcoxon_P_Value': comp.get('wilcoxon_p_value', np.nan),
                        'Cohens_D': comp['cohens_d'],
                        'CI_Lower': comp['ci_lower'],
                        'CI_Upper': comp['ci_upper'],
                        'Significant_T_Test': comp['significant_t_test'],
                        'Significant_Wilcoxon': comp.get('significant_wilcoxon', False)
                    }
                    
                    if 'bonferroni_p_value' in comp:
                        row.update({
                            'Bonferroni_P_Value': comp['bonferroni_p_value'],
                            'FDR_P_Value': comp['fdr_p_value'],
                            'Significant_Bonferroni': comp['significant_bonferroni'],
                            'Significant_FDR': comp['significant_fdr']
                        })
                    
                    csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_dir / 'detailed_statistical_results.csv', index=False)
            print(f"Detailed results saved to {output_dir / 'detailed_statistical_results.csv'}")
        else:
            print("No data available for CSV export")

def main():
    parser = argparse.ArgumentParser(
        description='Perform comprehensive statistical analysis on EAGLE speed comparisons'
    )
    parser.add_argument(
        '--results-dir', 
        type=str, 
        required=True,
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--tokenizer-path', 
        type=str, 
        required=True,
        help='Path to tokenizer model'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='statistical_analysis_results',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--benchmarks', 
        type=str, 
        nargs='+',
        default=['mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum', 'qa'],
        help='Benchmarks to analyze'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveStatisticalAnalyzer(args.tokenizer_path)
        
        # Analyze results
        print(f"Analyzing results in {args.results_dir}...")
        analyzer.analyze_results_directory(args.results_dir, args.benchmarks)
        
        # Apply multiple comparison corrections
        print("Applying multiple comparison corrections...")
        analyzer.apply_multiple_comparison_corrections()
        
        # Generate report
        print("Generating comprehensive report...")
        report = analyzer.generate_summary_report()
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'statistical_analysis_report.txt', 'w') as f:
            f.write(report)
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.create_visualizations(output_dir)
        
        # Save detailed results
        print("Saving detailed results...")
        analyzer.save_detailed_results(output_dir)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}")
        print("\nSummary report:")
        print(report)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 