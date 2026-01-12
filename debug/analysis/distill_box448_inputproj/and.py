import pandas as pd

a = pd.read_csv('/mnt/data/ccy/EasyR1/debug/analysis/distill_box448_inputproj/inputproj_group_stats_filtered_0.5.csv')
b = pd.read_csv('/mnt/data/ccy/EasyR1/debug/analysis/normalized_traj_text/group_stats_filtered_0.1.csv')
df = pd.concat([a, b]).sort_values('group_size', ascending=False)
df = df.drop_duplicates('token', keep='first')
df.to_csv('group_stats_filtered_0.5_merged.csv', index=False)
