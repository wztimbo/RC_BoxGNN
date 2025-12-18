# 使用ReChorus复现BoxGNN模型

## 项目简介

本项目在 ReChorus 推荐系统框架上实现 BoxGNN，
用于探索图结构表示学习在推荐系统中的应用。

项目添加了BoxGNN.py文件、BoxGNNReader.py、BoxGNNRunner.py文件

该代码主要用于课程实验与科研复现。

## 环境与配置

- python 3.10.4
- torch 2.1.0
- torch-scatter 2.1.2+pt21cpu
- torch-sparse 0.6.18+pt21cpu
```bash
# 使用conda创建虚拟环境

conda create -n rc_boxgnn python=3.10.4
```
```bash
# 安装依赖
cd RC

pip install -r requirements.txt

# 安装torch_scatter,torch_sparse
pip install torch_scatter torch_sparse  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```
### 不同数据集下运行BoxGNN
```bash
cd src
python main.py --model BoxGNN --dataset Grocery_and_Gourmet_Food --beta 0.3
```
```bash
python main.py --model BoxGNN --dataset MovieLens --beta 0.2
```
```bash
python main.py --model BoxGNN --dataset lastfm --beta 0.3
```
### 其它模型
```bash
python main.py --model LightGCN --dataset Grocery_and_Gourmet_Food 
```