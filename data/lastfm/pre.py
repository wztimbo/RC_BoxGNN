import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

DATA_DIR = r'D:\PythonFiles\RC\data\lastfm'
OUT_DIR  = r'D:\PythonFiles\RC\data\lastfm'
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# 1. 读取 train，建立连续 user / item 映射（核心）
# =====================================================
train_df = pd.read_csv(f'{DATA_DIR}/train.csv', sep='\t')

users = sorted(train_df['user_id'].unique())
items = sorted(train_df['item_id'].unique())

user_map = {u: i for i, u in enumerate(users)}
item_map = {i: j for j, i in enumerate(items)}

n_users = len(user_map)
n_items = len(item_map)

# =====================================================
# 2. 生成 train / dev / test.txt（统一 remap）
# =====================================================
def save_ui(csv_file, out_file):
    df = pd.read_csv(csv_file, sep='\t')
    df = df[['user_id', 'item_id']].drop_duplicates()

    # 只保留 train 中出现过的 user / item
    df = df[df['user_id'].isin(user_map) & df['item_id'].isin(item_map)]

    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)

    df.to_csv(out_file, sep='\t', index=False, header=True)

save_ui(f'{DATA_DIR}/train.csv', f'{OUT_DIR}/train.txt')
save_ui(f'{DATA_DIR}/dev.csv',   f'{OUT_DIR}/dev.txt')
save_ui(f'{DATA_DIR}/test.csv',  f'{OUT_DIR}/test.txt')

print('Saved train/dev/test.txt')

# =====================================================
# 3. 构建 UI 矩阵（基于 remap 后的 train）
# =====================================================
train_df = pd.read_csv(f'{OUT_DIR}/train.txt', sep='\t')

row = train_df['user_id'].to_numpy()
col = train_df['item_id'].to_numpy()

UI = csr_matrix(
    (np.ones(len(train_df)), (row, col)),
    shape=(n_users, n_items)
)

# =====================================================
# 4. item → tag（KMeans）
# =====================================================
item_co = (UI.T @ UI).toarray()
np.fill_diagonal(item_co, 0)

N_TAGS = 50

item_tags = KMeans(
    n_clusters=N_TAGS,
    random_state=42,
    n_init=10
).fit_predict(item_co)

item2tag = pd.DataFrame({
    'item_id': np.arange(n_items),
    'tag_id': item_tags
})


item2tag.to_csv(
    f'{OUT_DIR}/item2tag.txt',
    sep='\t', index=False, header=True
)

# =====================================================
# 5. user → tag（KMeans）
# =====================================================
user_co = (UI @ UI.T).toarray()
np.fill_diagonal(user_co, 0)

user_tags = KMeans(
    n_clusters=N_TAGS,
    random_state=42,
    n_init=10
).fit_predict(user_co)

user2tag = pd.DataFrame({
    'user_id': np.arange(n_users),
    'tag_id': user_tags
})


user2tag.to_csv(
    f'{OUT_DIR}/user2tag.txt',
    sep='\t', index=False, header=True
)

print('Saved user2tag.txt / item2tag.txt')
