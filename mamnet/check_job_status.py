"""
Identify Failed Fine-tuning Jobs

Scans output directories and log files to identify:
- Jobs with no results.json (failed to start or crashed early)
- Jobs with incomplete training (fewer epochs than expected)
- Jobs cancelled due to time limit
- Jobs with other errors

Generates CSV report of all jobs with their status.
"""

import os
import json
import glob
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Expected configuration
CITIES = ['chicago', 'miami', 'phoenix']
RESOLUTIONS = ['highres', 'midres']
STRATEGIES = ['random', 'clustered', 'dispersed']
N_VALUES = [0, 25, 50, 100, 200, 350, 450]
SEEDS = [1, 2, 3, 4, 5]

# Total expected jobs
TOTAL_JOBS = len(CITIES) * len(RESOLUTIONS) * len(STRATEGIES) * len(N_VALUES) * len(SEEDS)


def parse_folder_name(folder_name: str) -> Dict:
    """
    Parse finetune folder name to extract configuration
    
    Format: finetune_{city}_{res}_{strategy}_N{n}_seed{seed}_{timestamp}
    """
    try:
        parts = folder_name.split('_')
        
        if len(parts) < 7:
            return None
        
        config = {
            'city': parts[1],
            'resolution': parts[2],
            'strategy': parts[3],
            'n_samples': int(parts[4][1:]),  # Remove 'N' prefix
            'seed': int(parts[5][4:]),  # Remove 'seed' prefix
            'timestamp': '_'.join(parts[6:])
        }
        
        return config
    except:
        return None

def parse_log_name(log_name: str) -> Dict:
    """
    Parse split log file name to extract configuration
    
    Formats: 
    - Regular: split_{city}_{res}_{strategy}_N{n}_s{seed}.out
    - N=600: split_{city}_{res}_original_N600.out (no seed)
    """
    try:
        # Remove .out extension
        name = log_name.replace('.out', '')
        parts = name.split('_')
        
        # Check for N=600 case (no seed suffix)
        if len(parts) == 5 and parts[3] == 'original' and parts[4].startswith('N'):
            config = {
                'city': parts[1],
                'resolution': parts[2],
                'strategy': parts[3],
                'n_samples': int(parts[4][1:]),  # Remove 'N' prefix
                'seed': 1  # N=600 uses seed=1 as canonical
            }
            return config
        
        # Regular case
        if len(parts) < 6:
            return None
        
        config = {
            'city': parts[1],
            'resolution': parts[2],
            'strategy': parts[3],
            'n_samples': int(parts[4][1:]),  # Remove 'N' prefix
            'seed': int(parts[5][1:])  # Remove 's' prefix
        }
        
        return config
    except:
        return None
    
def get_expected_split_filename(city: str, resolution: str, strategy: str, 
                                n_samples: int, seed: int) -> str:
    """
    Get the expected split filename accounting for special cases
    
    Returns:
        Expected split filename
    """
    # Special case: N=600 uses original split (no strategy/seed variation)
    if n_samples == 600:
        return f"{city}_{resolution}_original_N{n_samples:03d}.json"
    
    # Special case: Dispersed uses random split as fallback
    # Miami: N>=350, Chicago/Phoenix: N=450
    if strategy == 'dispersed':
        if (city == 'miami' and n_samples >= 350) or (city in ['chicago', 'phoenix'] and n_samples == 450):
            return f"{city}_{resolution}_random_N{n_samples:03d}_seed{seed}.json"
    
    # Regular case
    return f"{city}_{resolution}_{strategy}_N{n_samples:03d}_seed{seed}.json"

def check_split_file(split_path: str) -> Dict:
    """
    Check if split file exists and is valid
    
    Returns:
        Dict with status info
    """
    if not os.path.exists(split_path):
        return {
            'has_split': False,
            'split_valid': False,
            'split_error': 'File does not exist'
        }
    
    try:
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        # Check required fields
        required_fields = ['train_filenames', 'val_filenames', 'spatial_metrics', 
                          'strategy', 'n_samples']
        
        for field in required_fields:
            if field not in split_data:
                return {
                    'has_split': True,
                    'split_valid': False,
                    'split_error': f'Missing field: {field}'
                }
        
        # Check if we have files
        n_train = len(split_data['train_filenames'])
        n_val = len(split_data['val_filenames'])
        
        if n_train == 0 and n_val == 0 and split_data['n_samples'] > 0:
            return {
                'has_split': True,
                'split_valid': False,
                'split_error': 'No train/val files in split'
            }
        
        return {
            'has_split': True,
            'split_valid': True,
            'split_error': None,
            'n_train': n_train,
            'n_val': n_val
        }
        
    except json.JSONDecodeError:
        return {
            'has_split': True,
            'split_valid': False,
            'split_error': 'Invalid JSON'
        }
    except Exception as e:
        return {
            'has_split': True,
            'split_valid': False,
            'split_error': f'Error reading file: {str(e)}'
        }

def check_job_status(output_dir: str, log_file: str = None) -> Dict:
    """
    Check status of a single job
    
    Returns dict with:
    - status: 'completed', 'partial', 'failed', 'no_results'
    - reason: Description of issue
    - epochs_completed: Number of epochs completed (if applicable)
    - has_results: Whether results.json exists
    - has_checkpoint: Whether checkpoint exists
    """
    status_info = {
        'status': 'unknown',
        'reason': '',
        'epochs_completed': 0,
        'has_results': False,
        'has_checkpoint': False,
        'has_spatial_metrics': False,
        'has_visualization': False,
        'error_message': ''
    }
    
    # Check if results.json exists
    results_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_path):
        status_info['has_results'] = True
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Check if we have valid metrics
            if 'test_metrics' in results and 'F1' in results['test_metrics']:
                status_info['status'] = 'completed'
                status_info['reason'] = 'Successfully completed'
            else:
                status_info['status'] = 'partial'
                status_info['reason'] = 'Results file incomplete'
        except:
            status_info['status'] = 'failed'
            status_info['reason'] = 'Results file corrupted'
    else:
        status_info['has_results'] = False
        status_info['status'] = 'no_results'
        status_info['reason'] = 'No results.json found'
    
    # Check for checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    if os.path.exists(checkpoint_path):
        status_info['has_checkpoint'] = True
    
    # Check for spatial metrics
    spatial_path = os.path.join(output_dir, 'spatial_metrics.json')
    if os.path.exists(spatial_path):
        status_info['has_spatial_metrics'] = True
    
    # Check for visualization
    viz_path = os.path.join(output_dir, 'sampling_visualization.png')
    if os.path.exists(viz_path):
        status_info['has_visualization'] = True
    
    # Check log file for errors if provided
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Check for specific error patterns
            if 'CANCELLED' in log_content and 'TIME LIMIT' in log_content:
                status_info['status'] = 'failed'
                status_info['reason'] = 'Time limit exceeded'
            elif 'ERROR' in log_content or 'Error' in log_content:
                status_info['status'] = 'failed'
                # Try to extract error message
                error_lines = [line for line in log_content.split('\n') 
                             if 'error' in line.lower() or 'exception' in line.lower()]
                if error_lines:
                    status_info['error_message'] = error_lines[-1][:200]  # Last error, truncated
                    status_info['reason'] = 'Error in execution'
            elif 'Traceback' in log_content:
                status_info['status'] = 'failed'
                status_info['reason'] = 'Python exception'
                # Extract exception type
                traceback_lines = log_content.split('\n')
                for i, line in enumerate(traceback_lines):
                    if 'Traceback' in line and i + 1 < len(traceback_lines):
                        # Get next few lines
                        status_info['error_message'] = ' '.join(traceback_lines[i:i+3])[:200]
                        break
            
            # Try to count epochs completed from log
            epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', log_content)
            if epoch_matches:
                completed_epochs = [int(m[0]) for m in epoch_matches]
                if completed_epochs:
                    status_info['epochs_completed'] = max(completed_epochs)
        except:
            pass
    
    return status_info

# def check_job_status(log_file: str) -> Dict:
#     """
#     Check status of a single job based on log file only
#     """
#     status_info = {
#         'status': 'unknown',
#         'reason': '',
#         'has_log': False,
#         'error_message': ''
#     }
    
#     if not os.path.exists(log_file):
#         status_info['status'] = 'missing'
#         status_info['reason'] = 'Log file not found'
#         return status_info
    
#     status_info['has_log'] = True
    
#     try:
#         with open(log_file, 'r') as f:
#             log_content = f.read()
        
#         # Check for successful completion - look for your specific completion message
#         # Adjust these patterns based on what your split script outputs
#         if 'Successfully saved splits' in log_content or 'Split generation complete' in log_content:
#             status_info['status'] = 'completed'
#             status_info['reason'] = 'Successfully completed'
#         elif 'CANCELLED' in log_content and 'TIME LIMIT' in log_content:
#             status_info['status'] = 'failed'
#             status_info['reason'] = 'Time limit exceeded'
#         elif 'ERROR' in log_content or 'Error' in log_content:
#             status_info['status'] = 'failed'
#             error_lines = [line for line in log_content.split('\n') 
#                          if 'error' in line.lower() or 'exception' in line.lower()]
#             if error_lines:
#                 status_info['error_message'] = error_lines[-1][:200]
#                 status_info['reason'] = 'Error in execution'
#         elif 'Traceback' in log_content:
#             status_info['status'] = 'failed'
#             status_info['reason'] = 'Python exception'
#             traceback_lines = log_content.split('\n')
#             for i, line in enumerate(traceback_lines):
#                 if 'Traceback' in line and i + 1 < len(traceback_lines):
#                     status_info['error_message'] = ' '.join(traceback_lines[i:i+3])[:200]
#                     break
#         else:
#             # If no clear completion or error message, assume running or incomplete
#             status_info['status'] = 'incomplete'
#             status_info['reason'] = 'No completion message found'
#     except Exception as e:
#         status_info['status'] = 'failed'
#         status_info['reason'] = f'Could not read log file: {str(e)}'
    
#     return status_info

def scan_all_jobs(output_dir: str, log_dir: str = None) -> pd.DataFrame:
    """
    Scan all jobs and create comprehensive report
    
    Args:
        output_dir: Directory containing finetune_* folders
        log_dir: Directory containing SLURM log files (optional)
    
    Returns:
        DataFrame with all jobs and their status
    """
    print("="*70)
    print("Scanning Fine-tuning Jobs")
    print("="*70)
    
    # Find all finetune directories
    pattern = os.path.join(output_dir, 'finetune_*')
    finetune_dirs = glob.glob(pattern)
    
    print(f"\nFound {len(finetune_dirs)} finetune directories")
    
    # Create expected job list
    expected_jobs = []
    for city in CITIES:
        for res in RESOLUTIONS:
            for strategy in STRATEGIES:
                for n in N_VALUES:
                    for seed in SEEDS:
                        expected_jobs.append({
                            'city': city,
                            'resolution': res,
                            'strategy': strategy,
                            'n_samples': n,
                            'seed': seed
                        })
    
    print(f"Expected {len(expected_jobs)} total jobs")
    
    # Scan existing jobs
    job_reports = []
    
    for ft_dir in finetune_dirs:
        folder_name = os.path.basename(ft_dir)
        config = parse_folder_name(folder_name)
        
        if config is None:
            print(f"Warning: Could not parse folder name: {folder_name}")
            continue
        
        # Find corresponding log file
        log_file = None
        if log_dir:
            log_pattern = os.path.join(log_dir, 
                f"ft_{config['city']}_{config['resolution']}_{config['strategy']}_N{config['n_samples']}_s{config['seed']}.out")
            log_files = glob.glob(log_pattern)
            if log_files:
                log_file = log_files[0]
            else:
                # Try alternative pattern
                log_pattern2 = os.path.join(log_dir, f"ft_*_{config['city']}_{config['resolution']}_{config['strategy']}_N{config['n_samples']:03d}_s{config['seed']}.out")
                log_files = glob.glob(log_pattern2)
                if log_files:
                    log_file = log_files[0]
        
        # Check job status
        status_info = check_job_status(ft_dir, log_file)
        
        # Combine config and status
        report = {
            'city': config['city'],
            'resolution': config['resolution'],
            'strategy': config['strategy'],
            'n_samples': config['n_samples'],
            'seed': config['seed'],
            'timestamp': config['timestamp'],
            'output_dir': ft_dir,
            'log_file': log_file if log_file else 'Not found',
            'overall_status': status_info['status'],
            'overall_reason': status_info['reason'],
            'epochs_completed': status_info['epochs_completed'],
            'has_results': status_info['has_results'],
            'has_checkpoint': status_info['has_checkpoint'],
            'has_spatial_metrics': status_info['has_spatial_metrics'],
            'has_visualization': status_info['has_visualization'],
            'error_message': status_info['error_message']
        }
        
        job_reports.append(report)
    
    # Create DataFrame
    df = pd.DataFrame(job_reports)
    # Keep only the latest job for each configuration (in case of reruns)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset=['city', 'resolution', 'strategy', 'n_samples', 'seed'], keep='last')
    
    # Check for missing jobs
    existing_keys = set(
        df.apply(lambda x: f"{x['city']}_{x['resolution']}_{x['strategy']}_{x['n_samples']}_{x['seed']}", axis=1)
    )
    
    missing_jobs = []
    for job in expected_jobs:
        key = f"{job['city']}_{job['resolution']}_{job['strategy']}_{job['n_samples']}_{job['seed']}"
        if key not in existing_keys:
            missing_jobs.append({
                **job,
                'overall_status': 'missing',
                'overall_reason': 'Job never started or directory not created',
                'output_dir': 'N/A',
                'log_file': 'N/A'
            })
    
    if missing_jobs:
        missing_df = pd.DataFrame(missing_jobs)
        df = pd.concat([df, missing_df], ignore_index=True)
    
    # Sort by configuration
    df = df.sort_values(['city', 'resolution', 'strategy', 'n_samples', 'seed']).reset_index(drop=True)
    
    return df

# def scan_all_jobs(log_dir: str, splits_dir: str) -> pd.DataFrame:
#     """
#     Scan all split jobs from log files AND split files
    
#     Args:
#         log_dir: Directory containing split_*.out log files
#         splits_dir: Directory containing split JSON files
    
#     Returns:
#         DataFrame with all jobs and their status
#     """
#     print("="*70)
#     print("Scanning Split Generation Jobs")
#     print("="*70)
    
#     # Find all split log files
#     pattern = os.path.join(log_dir, 'split_*.out')
#     log_files = glob.glob(pattern)
    
#     print(f"\nFound {len(log_files)} split log files")
    
#     # Create expected job list
#     expected_jobs = []
#     for city in CITIES:
#         for res in RESOLUTIONS:
#             for strategy in STRATEGIES:
#                 for n in N_VALUES:
#                     for seed in SEEDS:
#                         expected_jobs.append({
#                             'city': city,
#                             'resolution': res,
#                             'strategy': strategy,
#                             'n_samples': n,
#                             'seed': seed
#                         })

#     # Add N=600 jobs (all strategies/seeds use same split)
#     for city in CITIES:
#         for res in RESOLUTIONS:
#             for strategy in STRATEGIES:
#                 for seed in SEEDS:
#                     expected_jobs.append({
#                         'city': city,
#                         'resolution': res,
#                         'strategy': strategy,
#                         'n_samples': 600,
#                         'seed': seed
#                     })
    
#     print(f"Expected {len(expected_jobs)} total jobs")
    
#     # Scan existing jobs
#     job_reports = []
    
#     for log_file in log_files:
#         log_name = os.path.basename(log_file)
#         config = parse_log_name(log_name)
        
#         if config is None:
#             print(f"Warning: Could not parse log file name: {log_name}")
#             continue
        
#         # Check log status
#         status_info = check_job_status(log_file)
        
#         # Check split file
#         split_filename = get_expected_split_filename(
#             config['city'], config['resolution'], config['strategy'],
#             config['n_samples'], config['seed']
#         )
#         split_path = os.path.join(splits_dir, split_filename)
#         split_info = check_split_file(split_path)
        
#         # Determine overall status (both log and split must be good)
#         if status_info['status'] == 'completed' and split_info['split_valid']:
#             overall_status = 'ready'
#             overall_reason = 'Log completed and split file valid'
#         elif not split_info['has_split']:
#             overall_status = 'no_split'
#             overall_reason = 'Split file missing'
#         elif not split_info['split_valid']:
#             overall_status = 'invalid_split'
#             overall_reason = f"Split file invalid: {split_info['split_error']}"
#         else:
#             overall_status = status_info['status']
#             overall_reason = status_info['reason']
        
#         # Combine config and status
#         report = {
#             'city': config['city'],
#             'resolution': config['resolution'],
#             'strategy': config['strategy'],
#             'n_samples': config['n_samples'],
#             'seed': config['seed'],
#             'log_file': log_file,
#             'split_file': split_path,
#             'expected_split': split_filename,
#             'overall_status': overall_status,
#             'overall_reason': overall_reason,
#             'log_status': status_info['status'],
#             'log_reason': status_info['reason'],
#             **split_info
#         }
        
#         job_reports.append(report)
    
#     # Create DataFrame
#     df = pd.DataFrame(job_reports)
    
#     # Check for missing jobs
#     existing_keys = set(
#         df.apply(lambda x: f"{x['city']}_{x['resolution']}_{x['strategy']}_{x['n_samples']}_{x['seed']}", axis=1)
#     )
    
#     missing_jobs = []
#     for job in expected_jobs:
#         key = f"{job['city']}_{job['resolution']}_{job['strategy']}_{job['n_samples']}_{job['seed']}"
#         if key not in existing_keys:
#             # Check if split file exists even without log
#             split_filename = get_expected_split_filename(
#                 job['city'], job['resolution'], job['strategy'],
#                 job['n_samples'], job['seed']
#             )
#             split_path = os.path.join(splits_dir, split_filename)
#             split_info = check_split_file(split_path)
            
#             if split_info['split_valid']:
#                 overall_status = 'ready'
#                 overall_reason = 'Split file exists (log missing but OK)'
#             elif split_info['has_split']:
#                 overall_status = 'invalid_split'
#                 overall_reason = f"Split invalid: {split_info['split_error']}"
#             else:
#                 overall_status = 'missing'
#                 overall_reason = 'Both log and split file missing'
            
#             missing_jobs.append({
#                 **job,
#                 'overall_status': overall_status,
#                 'overall_reason': overall_reason,
#                 'log_status': 'missing',
#                 'log_reason': 'Log file not created',
#                 'log_file': 'N/A',
#                 'split_file': split_path,
#                 'expected_split': split_filename,
#                 **split_info
#             })
    
#     if missing_jobs:
#         missing_df = pd.DataFrame(missing_jobs)
#         df = pd.concat([df, missing_df], ignore_index=True)
    
#     # Sort by configuration
#     df = df.sort_values(['city', 'resolution', 'strategy', 'n_samples', 'seed']).reset_index(drop=True)
    
#     return df

## Changes in here
def print_summary(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    print(f"\nTotal jobs expected: {TOTAL_JOBS}")
    print(f"Total jobs found: {len(df)}")
    
    print(f"\nOverall Status breakdown:")
    status_counts = df['overall_status'].value_counts()
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {status:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # Failed jobs breakdown
    failed_df = df[~df['overall_status'].isin(['ready', 'completed'])]

# Uncomment
    # print("\nSplit file status:")
    # split_status_counts = failed_df.groupby(['has_split', 'split_valid']).size()
    # for (has_split, valid), count in split_status_counts.items():
    #     status_str = f"Has split: {has_split}, Valid: {valid}"
    #     print(f"  {status_str:40s}: {count}")
    
    if len(failed_df) > 0:
        print(f"\n{len(failed_df)} jobs need attention:")
        print("\nReasons:")
        reason_counts = failed_df['overall_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason:40s}: {count:4d}")
        
        # Breakdown by configuration
        print("\nFailed jobs by city:")
        for city in CITIES:
            city_failed = len(failed_df[failed_df['city'] == city])
            print(f"  {city:10s}: {city_failed}")
        
        print("\nFailed jobs by strategy:")
        for strategy in STRATEGIES:
            strategy_failed = len(failed_df[failed_df['strategy'] == strategy])
            print(f"  {strategy:10s}: {strategy_failed}")
        
        print("\nFailed jobs by N:")
        for n in N_VALUES:
            n_failed = len(failed_df[failed_df['n_samples'] == n])
            print(f"  N={n:3d}: {n_failed}")
    else:
        print("\n✓ All jobs completed successfully!")
    
    # Time limit failures
    time_limit_df = df[df['overall_reason'] == 'Time limit exceeded']
    if len(time_limit_df) > 0:
        print(f"\n⚠ {len(time_limit_df)} jobs cancelled due to time limit")
        print("  Consider increasing --time in finetune.sh")


def save_reports(df: pd.DataFrame, save_dir: str):
    """Save detailed reports"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full report
    full_report_path = os.path.join(save_dir, 'job_status_full.csv')
    df.to_csv(full_report_path, index=False)
    print(f"\n✓ Full report saved: {full_report_path}")
    
    # Save failed jobs only
    failed_df = df[~df['overall_status'].isin(['ready', 'completed'])]
    if len(failed_df) > 0:
        failed_report_path = os.path.join(save_dir, 'job_status_failed.csv')
        failed_df.to_csv(failed_report_path, index=False)
        print(f"✓ Failed jobs report saved: {failed_report_path}")
        
        # Create resubmission script
        resubmit_script_path = os.path.join(save_dir, 'resubmit_failed_jobs.sh')
        create_resubmit_script(failed_df, resubmit_script_path)
        print(f"✓ Resubmission script saved: {resubmit_script_path}")
    
    # Save completed jobs
    completed_df = df[df['overall_status'] == 'ready']
    if len(completed_df) > 0:
        completed_report_path = os.path.join(save_dir, 'job_status_ready.csv')
        completed_df.to_csv(completed_report_path, index=False)
        print(f"✓ Completed jobs report saved: {completed_report_path}")


def create_resubmit_script(failed_df: pd.DataFrame, output_path: str):
    """Create bash script to resubmit failed jobs"""
    
    # Filter out missing jobs (those need different handling)
    resubmit_df = failed_df[failed_df['status'] != 'missing']
    
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('# Resubmit Failed Fine-tuning Jobs\n')
        f.write('# Generated automatically by check_job_status.py\n\n')
        
        f.write('BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"\n')
        f.write('OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"\n')
        f.write('LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/finetune"\n\n')
        
        f.write('echo "Resubmitting failed jobs..."\n')
        f.write('echo ""\n\n')
        
        for idx, row in resubmit_df.iterrows():
            name = f"ft_{row['city']}_{row['resolution']}_{row['strategy']}_N{row['n_samples']}_s{row['seed']}"
            outputfile = f"${{LOG_DIR}}/{name}_retry.out"
            
            f.write(f"# Retry: {row['reason']}\n")
            f.write(f"echo 'Resubmitting: {name}'\n")
            f.write(f"sbatch --output={outputfile} \\\n")
            f.write(f"       --job-name={name}_retry \\\n")
            f.write(f"       --export=TARGET_CITY={row['city']},RESOLUTION={row['resolution']},"
                   f"N_SAMPLES={row['n_samples']},STRATEGY={row['strategy']},"
                   f"RANDOM_SEED={row['seed']},BASE_DATA_ROOT=${{BASE_DATA_ROOT}},"
                   f"OUTPUT_DIR=${{OUTPUT_DIR}} \\\n")
            f.write(f"       finetune.sh\n\n")
        
        f.write('echo ""\n')
        f.write(f'echo "Resubmitted {len(resubmit_df)} jobs"\n')
    
    # Make executable
    os.chmod(output_path, 0o755)

# def create_resubmit_script(failed_df: pd.DataFrame, output_path: str):
#     """Create bash script to resubmit failed jobs"""
    
#     # Filter out missing jobs (those need different handling)
#     resubmit_df = failed_df[~failed_df['overall_status'].isin(['ready'])]
    
#     with open(output_path, 'w') as f:
#         f.write('#!/bin/bash\n\n')
#         f.write('# Resubmit Failed Fine-tuning Jobs\n')
#         f.write('# Generated automatically by check_job_status.py\n\n')
        
#         f.write('BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"\n')
#         f.write('OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"\n')
#         f.write('LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/finetune"\n\n')
        
#         f.write('echo "Resubmitting failed jobs..."\n')
#         f.write('echo ""\n\n')
        
#         for idx, row in resubmit_df.iterrows():
#             # name = f"ft_{row['city']}_{row['resolution']}_{row['strategy']}_N{row['n_samples']}_s{row['seed']}"
#             name = f"split_{row['city']}_{row['resolution']}_{row['strategy']}_N{row['n_samples']}_s{row['seed']}"
#             outputfile = f"${{LOG_DIR}}/{name}_retry.out"
            
#             f.write(f"# Retry: {row['overall_reason']}\n")
#             f.write(f"echo 'Resubmitting: {name}'\n")
#             f.write(f"sbatch --output={outputfile} \\\n")
#             f.write(f"       --job-name={name}_retry \\\n")
#             f.write(f"       --export=TARGET_CITY={row['city']},RESOLUTION={row['resolution']},"
#                    f"N_SAMPLES={row['n_samples']},STRATEGY={row['strategy']},"
#                    f"RANDOM_SEED={row['seed']},BASE_DATA_ROOT=${{BASE_DATA_ROOT}},"
#                    f"OUTPUT_DIR=${{OUTPUT_DIR}} \\\n")
#             f.write(f"       finetune.sh\n\n")
        
#         f.write('echo ""\n')
#         f.write(f'echo "Resubmitted {len(resubmit_df)} jobs"\n')
    
#     # Make executable
#     os.chmod(output_path, 0o755)

# Changes to Make
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check fine-tuning job status')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing finetune_* folders')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory containing SLURM log files')
    parser.add_argument('--save_dir', type=str, default='./job_reports',
                       help='Directory to save reports')
    parser.add_argument('--splits_dir', type=str, required=True,
                       help='Directory containing split JSON files')
    
    args = parser.parse_args()
    
    # Scan jobs
    df = scan_all_jobs(args.output_dir, args.log_dir)
    # df = scan_all_jobs(args.log_dir, args.splits_dir)
    
    # Print summary
    print_summary(df)
    
    # Save reports
    save_reports(df, args.save_dir)
    
    print("\n" + "="*70)
    print("Job status check complete!")
    print("="*70)
    
    # Return exit code based on failures
    failed_count = len(df[~df['overall_status'].isin(['ready', 'completed'])])
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    exit(main())