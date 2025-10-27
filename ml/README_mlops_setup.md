# MLOps Setup Guide

## 项目概述
本项目已成功集成了完整的MLOps流程，包括数据版本控制(DVC)和实验跟踪(MLflow)。

## 已完成的MLOps功能

### 1. 数据版本控制 (DVC)
- ✅ **DVC初始化**: 项目已配置DVC
- ✅ **远程存储**: 配置DagsHub作为远程存储
- ✅ **数据版本**: 创建了v1和v2两个数据集版本
- ✅ **数据跟踪**: 使用DVC跟踪所有数据文件

#### 数据集版本
- **V1 (基础版本)**: 1309个样本，9个特征
  - 基础数据清洗
  - 移除zero列
  - 前向填充缺失值
  
- **V2 (增强版本)**: 346个样本，11个特征
  - 包含V1的所有处理
  - 异常值移除(IQR方法)
  - 特征工程(年龄分组)
  - 类别平衡采样

### 2. 实验跟踪 (MLflow)
- ✅ **MLflow集成**: 完整的实验跟踪系统
- ✅ **实验设计**: 4个对比实验
- ✅ **指标记录**: 准确率、精确率、召回率、F1分数
- ✅ **模型版本**: 自动记录模型和预处理器

#### 实验结果总结
| 实验 | 数据集 | 模型类型 | F1分数 | 准确率 | 状态 |
|------|--------|----------|--------|--------|------|
| 1 | V1 | 基线SVM | 0.5087 | 0.7837 | ✅ |
| 2 | V1 | 调优SVM | 0.5143 | 0.7837 | ✅ |
| 3 | V2 | 基线SVM | 0.7324 | 0.7324 | ✅ |
| 4 | V2 | 调优SVM | 0.6667 | 0.6901 | ✅ |

### 3. 生产就绪模型选择
**推荐模型**: V2数据集的基线SVM模型
- **F1分数**: 0.7324 (最高)
- **准确率**: 0.7324
- **选择理由**: 
  - 在平衡数据集上表现最佳
  - 特征工程和异常值移除提升了性能
  - 类别平衡解决了数据不平衡问题

## 文件结构
```
ml/
├── README_data.md              # 数据版本文档
├── README_experiments.md       # 实验跟踪文档
├── README_mlops_setup.md       # MLOps设置指南
├── create_data_versions.py     # 数据版本创建脚本
├── train_with_mlops.py         # MLOps训练脚本
└── create_sample_data.py       # 示例数据生成(未使用)

data/
├── train_and_test2.csv         # 原始数据
├── train_and_test2.csv.dvc     # 原始数据DVC文件
├── train_and_test2_v1.csv      # V1数据集
├── train_and_test2_v1.csv.dvc  # V1数据DVC文件
├── train_and_test2_v2.csv      # V2数据集
└── train_and_test2_v2.csv.dvc  # V2数据DVC文件
```

## 使用方法

### 1. 运行实验
```bash
cd ml
python train_with_mlops.py
```

### 2. 查看MLflow UI
```bash
mlflow ui
# 然后在浏览器中打开 http://localhost:5000
```

### 3. 数据版本管理
```bash
# 拉取特定数据版本
dvc pull data/train_and_test2_v1.csv.dvc
dvc pull data/train_and_test2_v2.csv.dvc

# 推送数据到远程
dvc push
```

## DVC远程存储配置

### DagsHub配置
- **仓库地址**: https://dagshub.com/whitecaptain2003/LastPro
- **DVC远程**: https://dagshub.com/whitecaptain2003/LastPro.dvc
- **认证**: 需要配置用户名和密码

### 配置DVC认证
```bash
# 设置用户名
dvc remote modify storage --local user whitecaptain2003

# 设置密码
dvc remote modify storage --local password YOUR_DAGSHUB_TOKEN

# 或者使用环境变量
export DVC_REMOTE_USER=whitecaptain2003
export DVC_REMOTE_PASSWORD=YOUR_DAGSHUB_TOKEN
```

## 关键指标说明

### 主要优化指标: F1分数
选择F1分数作为主要指标的原因：
1. **类别不平衡**: 原始数据存在类别不平衡问题
2. **业务影响**: 假阳性和假阴性都有重要影响
3. **平衡性能**: F1分数平衡了精确率和召回率
4. **模型比较**: 便于不同预处理方法的公平比较

## 下一步计划

1. **模型部署**: 将最佳模型部署到生产环境
2. **监控设置**: 建立模型性能监控
3. **CI/CD集成**: 集成到持续集成流程
4. **A/B测试**: 在生产环境中进行A/B测试

## 技术栈
- **数据版本控制**: DVC
- **实验跟踪**: MLflow
- **机器学习**: scikit-learn
- **数据处理**: pandas, numpy
- **远程存储**: DagsHub
- **版本控制**: Git

## 联系信息
- **GitHub**: https://github.com/captain2003white/LastPro
- **DagsHub**: https://dagshub.com/whitecaptain2003/LastPro
