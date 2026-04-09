"""查看 SC/train 中解的目标值的平均值"""
import pickle
import gzip
import glob
import numpy as np

data_dir = "data/samples/SC/train"
files = sorted(glob.glob(f"{data_dir}/sample_*.pkl"))

obj_vals = []
for f in files:
    with gzip.open(f, 'rb') as fp:
        data = pickle.load(fp)
    obj_vals.append(data['solution']['obj_val'])

obj_vals = np.array(obj_vals)
print(f"样本数: {len(obj_vals)}")
print(f"目标值均值: {obj_vals.mean():.4f}")
print(f"目标值标准差: {obj_vals.std():.4f}")
print(f"目标值最小值: {obj_vals.min():.4f}")
print(f"目标值最大值: {obj_vals.max():.4f}")
