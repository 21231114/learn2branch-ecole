"""
读取 train 目录下所有 .pkl 文件，输出 sample编号 -> 实例文件名 的对应关系。
用法：python check_sample_instance_map.py [--dir DIR] [--out OUT]
"""
import os
import re
import gzip
import pickle
import argparse
from pathlib import Path


def read_instance_field(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('instance', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data/samples/setcover/500r_1000c_0.05d/train',
                        help='包含 sample_*.pkl 的目录')
    parser.add_argument('--out', default=None,
                        help='结果输出到文件（默认只打印到终端）')
    args = parser.parse_args()

    sample_dir = Path(args.dir)
    files = sorted(sample_dir.glob('sample_*.pkl'),
                   key=lambda p: int(re.search(r'sample_(\d+)\.pkl', p.name).group(1)))

    print(f"共找到 {len(files)} 个文件，开始读取...\n")

    rows = []
    for path in files:
        sample_id = int(re.search(r'sample_(\d+)\.pkl', path.name).group(1))
        try:
            instance = read_instance_field(path)
            instance_name = os.path.basename(instance) if instance else 'UNKNOWN'
            # 从文件名提取实例编号，例如 instance_59.lp -> 59
            m = re.search(r'instance_(\d+)\.lp', instance_name)
            instance_id = int(m.group(1)) if m else -1
            match = '  ' if sample_id == instance_id else '!!'  # !! 表示编号不对应
            rows.append((sample_id, instance_id, instance_name, match))
        except Exception as e:
            rows.append((sample_id, -1, f'ERROR: {e}', '??'))

    # 统计
    mismatches = [(s, i, n) for s, i, n, m in rows if m != '  ']

    lines = []
    lines.append(f"{'sample_id':>10}  {'instance_id':>11}  {'instance_file':<35}  {'状态'}")
    lines.append('-' * 75)
    for sample_id, instance_id, instance_name, match in rows:
        lines.append(f"{sample_id:>10}  {instance_id:>11}  {instance_name:<35}  {match}")
    lines.append('-' * 75)
    lines.append(f"总计 {len(rows)} 个文件，其中编号不对应: {len(mismatches)} 个")

    if mismatches:
        lines.append("\n编号不对应的文件：")
        lines.append(f"{'sample_id':>10}  {'instance_id':>11}  {'instance_file'}")
        for s, i, n in mismatches:
            lines.append(f"{s:>10}  {i:>11}  {n}")

    output = '\n'.join(lines)
    print(output)

    if args.out:
        with open(args.out, 'w') as f:
            f.write(output + '\n')
        print(f"\n结果已写入 {args.out}")


if __name__ == '__main__':
    main()
