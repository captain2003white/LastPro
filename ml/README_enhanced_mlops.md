# Enhanced MLOps with Multiple Experiments and PKL Files

## 概述
本项目已成功集成了增强版MLOps功能，包括多个实验版本、PKL文件生成和完整的实验跟踪。

## 新增功能

### 1. 多实验版本生成
- **10个实验运行**: 每个配置运行多次实验
- **4种配置组合**:
  - V1数据 + 基线模型 (3次运行)
  - V1数据 + 改进模型 (2次运行)  
  - V2数据 + 基线模型 (3次运行)
  - V2数据 + 改进模型 (2次运行)

### 2. PKL文件生成
每个实验都会生成以下文件：
- `model.pkl`: 训练好的模型
- `scaler.pkl`: 数据预处理器
- `metadata.json`: 模型元数据

### 3. 模型文件结构
```
models/
├── SVM_v1_baseline_v1_baseline/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── SVM_v1_improved_v1_improved/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── SVM_v2_baseline_v2_baseline/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
└── SVM_v2_improved_v2_improved/
    ├── model.pkl
    ├── scaler.pkl
    └── metadata.json
```

## 实验结果总结

### 最佳模型
**V2基线SVM模型** - 最佳性能
- **F1分数**: 0.7324
- **准确率**: 0.7324
- **精确率**: 0.7222
- **召回率**: 0.7429

### 所有实验结果
| 配置 | 数据版本 | 模型类型 | 运行次数 | 平均F1 | 平均准确率 |
|------|----------|----------|----------|--------|------------|
| 1 | V1 | 基线 | 3 | 0.5087 | 0.7837 |
| 2 | V1 | 改进 | 2 | 0.5143 | 0.7837 |
| 3 | V2 | 基线 | 3 | **0.7324** | **0.7324** |
| 4 | V2 | 改进 | 2 | 0.6667 | 0.6901 |

## 使用方法

### 1. 运行多实验训练
```bash
cd ml
python enhanced_train_with_mlops.py
```

### 2. 查看MLflow实验结果
```bash
# 启动MLflow UI
mlflow ui

# 或者使用命令行查看
python view_experiments.py
```

### 3. 设置DVC模型跟踪
```bash
# 添加模型文件到DVC
dvc add models

# 提交到Git
git add models.dvc
git commit -m "Add model files to DVC tracking"
```

### 4. 推送到DagsHub
```bash
# 推送数据到DagsHub
dvc push

# 推送代码到GitHub
git push origin dev
```

## 在DagsHub中查看

### 1. 数据版本
访问: https://dagshub.com/whitecaptain2003/LastPro
- 查看数据版本历史
- 比较不同数据集版本
- 下载特定版本数据

### 2. 模型文件
- 所有PKL文件都通过DVC跟踪
- 可以在DagsHub中查看模型版本
- 支持模型文件的版本比较

### 3. 实验跟踪
- MLflow实验记录保存在本地
- 可以通过MLflow UI查看详细结果
- 每个实验都有完整的参数和指标记录

## MLflow UI功能

### 1. 实验比较
- 比较不同实验的性能
- 查看参数对性能的影响
- 识别最佳超参数组合

### 2. 模型管理
- 查看所有训练过的模型
- 下载模型文件
- 比较模型性能

### 3. 指标跟踪
- 准确率、精确率、召回率、F1分数
- 训练和测试样本数量
- 数据版本和Git提交信息

## 文件说明

### 核心脚本
- `enhanced_train_with_mlops.py`: 增强版训练脚本
- `view_experiments.py`: 实验查看器
- `setup_model_tracking.py`: DVC模型跟踪设置

### 配置文件
- `models.dvc`: DVC模型文件跟踪配置
- `metadata.json`: 每个模型的元数据

## 最佳实践

### 1. 模型选择
推荐使用V2基线SVM模型，因为：
- 最高的F1分数 (0.7324)
- 平衡的精确率和召回率
- 在增强数据集上训练

### 2. 实验管理
- 每次实验都有唯一的时间戳
- 包含Git提交信息用于代码版本跟踪
- 完整的元数据记录

### 3. 版本控制
- 使用DVC跟踪大型文件
- Git跟踪代码和配置
- 完整的可重现性

## 下一步计划

1. **模型部署**: 将最佳模型部署到生产环境
2. **监控设置**: 建立模型性能监控
3. **A/B测试**: 在生产环境中测试不同模型
4. **自动化**: 设置CI/CD流程自动运行实验

## 技术栈
- **数据版本控制**: DVC
- **实验跟踪**: MLflow
- **模型序列化**: Pickle
- **远程存储**: DagsHub
- **版本控制**: Git
- **机器学习**: scikit-learn

## 联系信息
- **GitHub**: https://github.com/captain2003white/LastPro
- **DagsHub**: https://dagshub.com/whitecaptain2003/LastPro
