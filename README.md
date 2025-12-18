# 使用ReChorus复现BoxGNN模型

## 项目简介

本项目在 ReChorus 推荐系统框架上实现 BoxGNN，
用于探索图结构表示学习在推荐系统中的应用。

项目主要添加了BoxGNN.py文件、BoxGNNReader.py、BoxGNNRunner.py文件

该代码主要用于课程实验与科研复现。

## 环境与配置

- python 3.10.4

```bash
# 使用conda创建虚拟环境

conda create -n rc_boxgnn python=3.10.4
conda activate rc_boxgnn
```
```bash
# 下载与安装依赖
git clone https://github.com/wztimbo/RC_BoxGNN.git

cd RC

# CPU运行
pip install -r requirements_cpu.txt
pip install torch_scatter torch_sparse  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# GPU运行
pip install -r requirements_gpu.txt
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```
### 不同数据集下运行BoxGNN
```bash
cd src
python main.py --model_name BoxGNN --dataset Grocery_and_Gourmet_Food --beta 0.3
```
```bash
python main.py --model_name BoxGNN --dataset MovieLens --beta 0.2
```
```bash
python main.py --model_name BoxGNN --dataset lastfm --beta 0.3
```
### 其它模型
```bash
python main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food 
```