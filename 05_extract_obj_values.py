"""
05_extract_obj_values.py — Extract objective values from processed dataset.

Reads all sample_*.pkl files in a given directory, extracts the solution
objective value for each sample, and saves the results to a CSV file.

Usage:
    python 05_extract_obj_values.py data/samples/setcover/500r_1000c_0.05d/train
    python 05_extract_obj_values.py data/samples/SC/train -o my_output.csv
"""

import os
import re
import sys
import argparse
import gzip
import pickle
import csv


def extract_obj_values(data_dir, output_csv):
    """
    Read all sample_*.pkl files in data_dir, extract objective values,
    and write results to output_csv.
    """
    def _numeric_key(path):
        nums = re.findall(r'\d+', os.path.basename(path))
        return int(nums[-1]) if nums else 0

    pkl_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir)
         if f.startswith('sample_') and f.endswith('.pkl')],
        key=_numeric_key,
    )

    if not pkl_files:
        print(f"No sample_*.pkl files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} sample files in {data_dir}")

    fieldnames = ['sample_file', 'instance', 'status', 'obj_val',
                  'primal_bound', 'dual_bound', 'solving_time', 'n_nodes']

    n_success = 0
    n_no_obj = 0
    n_error = 0

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pkl_path in enumerate(pkl_files):
            basename = os.path.basename(pkl_path)
            try:
                with gzip.open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                solution = data.get('solution', {})
                instance = data.get('instance', '')

                row = {
                    'sample_file': basename,
                    'instance': os.path.basename(instance) if instance else '',
                    'status': solution.get('status', ''),
                    'obj_val': solution.get('obj_val'),
                    'primal_bound': solution.get('primal_bound'),
                    'dual_bound': solution.get('dual_bound'),
                    'solving_time': solution.get('solving_time'),
                    'n_nodes': solution.get('n_nodes'),
                }
                writer.writerow(row)

                obj = solution.get('obj_val')
                if obj is not None:
                    n_success += 1
                else:
                    n_no_obj += 1

                if (i + 1) % 100 == 0 or (i + 1) == len(pkl_files):
                    print(f"  [{i+1}/{len(pkl_files)}] processed", flush=True)

            except Exception as e:
                print(f"  [ERROR] {basename}: {e}")
                n_error += 1

    print(f"\nDone. Results saved to {output_csv}")
    print(f"  Total: {len(pkl_files)}, with obj: {n_success}, "
          f"no obj: {n_no_obj}, errors: {n_error}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract objective values from sample_*.pkl dataset files.')
    parser.add_argument(
        'data_dir',
        help='Path to directory containing sample_*.pkl files.',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file path. Default: <data_dir>/obj_values.csv',
        default=None,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip('/')
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a directory.")
        sys.exit(1)

    output_csv = args.output or os.path.join(data_dir, 'obj_values.csv')

    extract_obj_values(data_dir, output_csv)
