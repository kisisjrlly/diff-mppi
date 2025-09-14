# LMPPI 分离式训练流程使用指南

这个改进后的例子将数据收集、模型训练和测试分为三个独立的步骤，支持大规模数据收集和长时间训练。

## 📋 使用流程

### 步骤 1: 数据收集 📊

收集大量轨迹数据并保存到本地磁盘：

```bash
# 基本使用 - 收集5000条轨迹
python step1_collect_data.py

# 大规模数据收集 - 收集50000条轨迹
python step1_collect_data.py --num_trajectories 50000 --horizon 60

# 自定义参数
python step1_collect_data.py \
    --num_trajectories 20000 \
    --horizon 50 \
    --save_dir ./my_data \
    --pd_ratio 0.7 \
    --batch_size 2000
```

**参数说明:**
- `--num_trajectories`: 收集的轨迹数量 (默认: 5000)
- `--horizon`: 每条轨迹的长度 (默认: 50)
- `--save_dir`: 数据保存目录 (默认: ./data)
- `--pd_ratio`: PD控制器轨迹比例 (默认: 0.8)
- `--batch_size`: 批处理大小 (默认: 1000)
- `--device`: 计算设备 cpu/cuda/auto (默认: auto)

**输出:**
- `pendulum_data_YYYYMMDD_HHMMSS.pkl`: 轨迹数据文件
- `pendulum_data_YYYYMMDD_HHMMSS_metadata.pkl`: 数据元信息

### 步骤 2: VAE 训练 🧠

使用收集的数据训练VAE模型：

```bash
# 基本训练
python step2_train_vae.py --data_path ./data/pendulum_data_20250913_143022.pkl

# 长时间充分训练
python step2_train_vae.py \
    --data_path ./data/pendulum_data_20250913_143022.pkl \
    --epochs 500 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --latent_dim 32 \
    --hidden_dims 1024 512 256 128 \
    --patience 50
```

**参数说明:**
- `--data_path`: 数据文件路径 (必需)
- `--epochs`: 训练轮数 (默认: 200)
- `--batch_size`: 批大小 (默认: 64)
- `--learning_rate`: 学习率 (默认: 1e-3)
- `--latent_dim`: 潜空间维度 (默认: 16)
- `--hidden_dims`: 隐藏层维度 (默认: 512 256 128 64)
- `--patience`: 早停耐心值 (默认: 30)
- `--train_ratio`: 训练集比例 (默认: 0.8)

**输出:**
- `./trained_models/vae_training_YYYYMMDD_HHMMSS/`: 训练目录
  - `checkpoint_best.pth`: 最佳模型
  - `checkpoint_latest.pth`: 最新模型
  - `training_curves.png`: 训练曲线
  - `model_config.json`: 模型配置
  - `training_metrics.json`: 训练指标

### 步骤 3: 控制器测试 🎯

测试训练好的LMPPI控制器：

```bash
# 基本测试
python step3_test_lmppi.py --model_path ./trained_models/vae_training_20250913_143022/checkpoint_best.pth

# 详细测试
python step3_test_lmppi.py \
    --model_path ./trained_models/vae_training_20250913_143022/checkpoint_best.pth \
    --num_episodes 50 \
    --horizon 30 \
    --num_samples 200 \
    --save_dir ./my_test_results
```

**参数说明:**
- `--model_path`: 训练好的模型路径 (必需)
- `--num_episodes`: 测试回合数 (默认: 20)
- `--horizon`: 控制时域 (默认: 20)
- `--num_samples`: MPPI采样数 (默认: 100)
- `--save_dir`: 结果保存目录 (默认: ./test_results)

**输出:**
- `./test_results/test_results_YYYYMMDD_HHMMSS/`: 测试结果目录
  - `comparison_results.png`: 性能对比图
  - `sample_trajectories.png`: 样本轨迹图
  - `detailed_results.pkl`: 详细结果数据
  - `summary_stats.json`: 汇总统计

## 🚀 推荐工作流程

### 大规模数据收集与训练

```bash
# 1. 收集大量数据 (可以分批进行)
python step1_collect_data.py --num_trajectories 100000 --horizon 60 --batch_size 5000

# 2. 长时间充分训练
python step2_train_vae.py \
    --data_path ./data/pendulum_data_*.pkl \
    --epochs 1000 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --latent_dim 64 \
    --hidden_dims 2048 1024 512 256 128 \
    --patience 100

# 3. 全面性能测试
python step3_test_lmppi.py \
    --model_path ./trained_models/vae_training_*/checkpoint_best.pth \
    --num_episodes 100 \
    --horizon 40 \
    --num_samples 500
```

### 快速验证流程

```bash
# 1. 小规模数据收集
python step1_collect_data.py --num_trajectories 1000 --horizon 30

# 2. 快速训练
python step2_train_vae.py \
    --data_path ./data/pendulum_data_*.pkl \
    --epochs 50 \
    --latent_dim 8

# 3. 快速测试
python step3_test_lmppi.py \
    --model_path ./trained_models/vae_training_*/checkpoint_best.pth \
    --num_episodes 10
```

## 📈 监控训练进度

训练过程中可以通过以下方式监控进度：

1. **实时输出**: 终端显示每轮训练的损失和指标
2. **训练曲线**: 每10轮自动保存训练曲线图
3. **检查点**: 定期保存模型检查点
4. **早停机制**: 验证损失不改善时自动停止

## 🔧 高级用法

### 并行数据收集
```bash
# 可以开启多个终端并行收集数据
# 终端1
python step1_collect_data.py --num_trajectories 25000 --save_dir ./data_batch1

# 终端2  
python step1_collect_data.py --num_trajectories 25000 --save_dir ./data_batch2

# 终端3
python step1_collect_data.py --num_trajectories 25000 --save_dir ./data_batch3

# 终端4
python step1_collect_data.py --num_trajectories 25000 --save_dir ./data_batch4
```

### 训练配置优化
```bash
# GPU训练，大批量，高学习率
python step2_train_vae.py \
    --data_path ./data/large_dataset.pkl \
    --device cuda \
    --batch_size 256 \
    --learning_rate 2e-3 \
    --epochs 2000 \
    --latent_dim 128
```

### 模型对比测试
```bash
# 测试不同的模型
python step3_test_lmppi.py --model_path ./models/model_v1/checkpoint_best.pth --save_dir ./results_v1
python step3_test_lmppi.py --model_path ./models/model_v2/checkpoint_best.pth --save_dir ./results_v2
python step3_test_lmppi.py --model_path ./models/model_v3/checkpoint_best.pth --save_dir ./results_v3
```

## 💡 最佳实践

1. **数据多样性**: 使用不同的控制器策略和噪声水平收集多样化数据
2. **渐进训练**: 先用小数据集快速验证，再用大数据集充分训练  
3. **模型保存**: 定期保存检查点，防止训练中断导致的损失
4. **超参调优**: 系统地调试学习率、批大小、网络结构等超参数
5. **性能监控**: 使用验证集监控过拟合，及时调整训练策略

现在您可以收集非常多的数据并进行长时间的充分训练！🎉
