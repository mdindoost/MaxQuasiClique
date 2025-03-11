#!/usr/bin/env python3
"""
Benchmark script for the MaxQuasiClique algorithm.
This script runs multiple tests and generates performance comparison charts.

Usage:
    python benchmark.py --executable=max_quasi_clique --sizes=1000,2000,5000 --seeds=10 --threads=4
"""

import subprocess
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
from tabulate import tabulate

def save_results(results, filename):
    """Save benchmark results to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def load_results(filename):
    """Load benchmark results from a JSON file"""
    if not os.path.exists(filename):
        print(f"Error: Results file {filename} not found")
        return None
    
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MaxQuasiClique algorithms")
    parser.add_argument("--executable", type=str, default="max_quasi_clique", 
                        help="Executable to benchmark")
    parser.add_argument("--optimized", type=str, default=None,
                        help="Optimized executable to compare with")
    parser.add_argument("--sizes", type=str, default="1000,2000,5000",
                        help="Comma-separated list of graph sizes to benchmark")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds to try for each run")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads to use (for optimized version)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout in seconds for each benchmark run")
    parser.add_argument("--load", action="store_true",
                        help="Load results from previous runs instead of running new benchmarks")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from saved results")
    args = parser.parse_args()
    
    # Parse graph sizes
    try:
        graph_sizes = [int(s) for s in args.sizes.split(",")]
    except ValueError:
        print("Error: Invalid graph sizes format. Use comma-separated integers.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = f"benchmark_results_{timestamp}"
    
    # Generate graphs if needed and not in plot-only mode
    if not args.plot_only and not args.load:
        if not generate_graphs(graph_sizes):
            print("Error generating graphs. Exiting.")
            sys.exit(1)
    
    results_list = []
    labels = []
    
    # Run or load base algorithm benchmarks
    base_results_file = f"{results_base}_base.json"
    if args.load or args.plot_only:
        base_results = load_results("benchmark_results_base.json")
        if base_results:
            results_list.append(base_results)
            labels.append("Base Algorithm")
    elif not args.plot_only:
        base_results = run_benchmarks(args.executable, graph_sizes, args.seeds, None, args.timeout, "Base Algorithm")
        save_results(base_results, base_results_file)
        results_list.append(base_results)
        labels.append("Base Algorithm")
    
    # Run or load optimized algorithm benchmarks if specified
    if args.optimized:
        opt_results_file = f"{results_base}_optimized.json"
        if args.load or args.plot_only:
            opt_results = load_results("benchmark_results_optimized.json")
            if opt_results:
                results_list.append(opt_results)
                labels.append("Optimized Algorithm")
        elif not args.plot_only:
            opt_results = run_benchmarks(args.optimized, graph_sizes, args.seeds, 
                                         args.threads, args.timeout, "Optimized Algorithm")
            save_results(opt_results, opt_results_file)
            results_list.append(opt_results)
            labels.append("Optimized Algorithm")
    
    # Print results table
    if results_list:
        print_results_table(results_list, labels)
    
    # Generate comparison plot
    if len(results_list) > 0:
        plot_file = f"benchmark_comparison_{timestamp}.png"
        compare_results(results_list, labels, 
                        f"MaxQuasiClique Algorithm Performance Comparison (Seeds: {args.seeds})",
                        plot_file)
    
    print("Benchmarking complete.")

def print_results_table(results_list, labels):
    """Print a formatted table of benchmark results"""
    headers = ["Algorithm", "Graph Size", "Solution Size", "Density", "Time (s)", "Success"]
    table_data = []
    
    for results, label in zip(results_list, labels):
        for r in results:
            row = [
                label,
                r['graph_size'],
                r.get('solution_size', 'N/A'),
                f"{r.get('density', 'N/A'):.4f}" if isinstance(r.get('density'), (int, float)) else 'N/A',
                f"{r.get('execution_time', 'N/A'):.2f}" if isinstance(r.get('execution_time'), (int, float)) else 'N/A',
                "Yes" if r.get('success', False) else "No"
            ]
            table_data.append(row)
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

def run_command(command, timeout=None):
    """Run a command and return stdout, stderr, and return code"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        return "", "Timeout expired", -1

def parse_results(stdout):
    """Parse the output of the MaxQuasiClique program"""
    results = {}
    
    # Extract solution size
    for line in stdout.split('\n'):
        if "Vertices:" in line:
            try:
                results['solution_size'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        
        if "Edges:" in line:
            try:
                parts = line.split(':')[1].strip().split('/')
                results['solution_edges'] = int(parts[0])
                results['possible_edges'] = int(parts[1])
            except (ValueError, IndexError):
                pass
        
        if "Density:" in line:
            try:
                results['density'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        
        if "Algorithm execution time:" in line:
            try:
                results['execution_time'] = float(line.split(':')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    
    return results

def generate_graphs(graph_sizes):
    """Generate test graphs of different sizes"""
    print("Generating test graphs...")
    
    for size in graph_sizes:
        output_file = f"test_graph_{size}.txt"
        if not os.path.exists(output_file):
            print(f"Generating graph with {size} vertices...")
            command = f"python3 generate_test_graph.py --neurons {size} --output {output_file} --seed 42"
            stdout, stderr, returncode = run_command(command)
            
            if returncode != 0:
                print(f"Error generating graph: {stderr}")
                return False
        else:
            print(f"Graph {output_file} already exists, skipping generation")
    
    return True

def run_benchmarks(executable, graph_sizes, num_seeds=10, num_threads=None, timeout=1800, label=None):
    """Run benchmarks on different graph sizes"""
    if label:
        print(f"\n=== Running benchmarks for {label} ===")
    else:
        print(f"\n=== Running benchmarks for {executable} ===") 
    results = []
    
    for size in graph_sizes:
        graph_file = f"test_graph_{size}.txt"
        
        if not os.path.exists(graph_file):
            print(f"Error: Graph file {graph_file} not found")
            continue
        
        print(f"Running benchmark on graph with {size} vertices...")
        
        thread_arg = f" {num_threads}" if num_threads else ""
        command = f"./{executable} {graph_file} {num_seeds}{thread_arg}"
        
        start_time = time.time()
        stdout, stderr, returncode = run_command(command, timeout)
        total_time = time.time() - start_time
        
        if returncode != 0:
            print(f"Error running benchmark: {stderr}")
            result = {
                'graph_size': size,
                'solution_size': 0,
                'execution_time': timeout if total_time >= timeout else total_time,
                'success': False,
                'timeout': total_time >= timeout
            }
        else:
            parsed = parse_results(stdout)
            parsed['graph_size'] = size
            parsed['success'] = True
            parsed['timeout'] = False
            result = parsed
        
        results.append(result)
        print(f"Result: {result}")
    
    return results

def compare_results(results_list, labels, title, output_file):
    """Compare multiple algorithm runs"""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        # Filter out failed runs
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print(f"No successful benchmark runs for {label}")
            continue
        
        # Prepare data
        graph_sizes = [r['graph_size'] for r in successful_results]
        solution_sizes = [r['solution_size'] for r in successful_results]
        execution_times = [r['execution_time'] for r in successful_results]
        densities = [r.get('density', 0) for r in successful_results]
        
        color = colors[i % len(colors)]
        
        # Plot solution size vs graph size
        ax1.plot(graph_sizes, solution_sizes, 'o-', linewidth=2, markersize=8, 
                 color=color, label=label)
        
        # Plot execution time vs graph size
        ax2.plot(graph_sizes, execution_times, 'o-', linewidth=2, markersize=8, 
                 color=color, label=label)
        
        # Plot density vs graph size
        ax3.plot(graph_sizes, densities, 'o-', linewidth=2, markersize=8, 
                 color=color, label=label)
    
    ax1.set_title('Solution Size vs Graph Size')
    ax1.set_xlabel('Graph Size (vertices)')
    ax1.set_ylabel('Solution Size (vertices)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Execution Time vs Graph Size')
    ax2.set_xlabel('Graph Size (vertices)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_title('Solution Density vs Graph Size')
    ax3.set_xlabel('Graph Size (vertices)')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    ax3.legend()
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def plot_results(results, title, output_file):
    """Generate performance plots for a single algorithm run"""
    # Filter out failed runs
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful benchmark runs to plot")
        return
    
    # Prepare data
    graph_sizes = [r['graph_size'] for r in successful_results]
    solution_sizes = [r['solution_size'] for r in successful_results]
    execution_times = [r['execution_time'] for r in successful_results]
    densities = [r.get('density', 0) for r in successful_results]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot solution size vs graph size
    ax1.plot(graph_sizes, solution_sizes, 'o-', linewidth=2, markersize=8)
    ax1.set_title('Solution Size vs Graph Size')
    ax1.set_xlabel('Graph Size (vertices)')
    ax1.set_ylabel('Solution Size (vertices)')
    ax1.grid(True)
    
    # Plot execution time vs graph size
    ax2.plot(graph_sizes, execution_times, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_title('Execution Time vs Graph Size')
    ax2.set_xlabel('Graph Size (vertices)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.grid(True)
    
    # Plot density vs graph size
    ax3.plot(graph_sizes, densities, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_title('Solution Density vs Graph Size')
    ax3.set_xlabel('Graph Size (vertices)')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()