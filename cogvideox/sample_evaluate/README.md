# CogVideo Batch Sampler - Simple

这是一个用于 CogVideo 视频生成和评估的工具集。

## 项目结构

```
simple/
├── run.sh                              # 主要运行脚本
├── simple_multiprocess_sampler.py      # 多进程采样器
├── calc_finnal_score.py                # 最终得分计算脚本
├── modify_cogvideo.py                  # CogVideo 修改脚本
├── test_block_sparse_attention.py      # 块稀疏注意力测试
├── configs/                            # 配置文件目录
├── VBench/                             # VBench 评估工具(需要手动clone到这里）
├── Cogvideo5b_4steps_0.84/            # 生成结果示例
└── Triton/                             # Triton 相关代码

```

## run.sh 脚本使用说明

`run.sh` 是整个工作流程的主脚本，包含从视频生成到评估的完整流程。

### 工作流程

#### 步骤 1: 激活环境

```bash
git clone https://github.com/Vchitect/VBench.git
source /workspace/Vbench_EVA/vbench_env/bin/activate
```

**注意**: 需要根据你的实际环境路径修改此行。

#### 步骤 2: 视频采样生成

```bash
python simple_multiprocess_sampler.py --config configs/8.29_cogvideo4steps_2.json
```

**配置说明**:
- 需要新建或修改配置文件（参考 `configs/8.29_cogvideo4steps_2.json`）
- 配置文件中需要修改的路径:
  - `naming_prompt_file`: 提示词文件路径（默认: `/workspace/Vbench_EVA/VBench/prompts/all_dimension.txt`）
  - `sampling_prompt_file`: 增强提示词文件路径（默认: `/workspace/Vbench_EVA/VBench/prompts/augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt`）

**提示**: 只需修改路径中的 VBench 位置，后半部分保持不变。

#### 步骤 3: 运行评估

```bash
bash /workspace/Vbench_EVA/cogvideo_batch_sampler/simple/VBench/vbench2_beta_long/evaluate_long.sh "videopath"
```

**注意事项**:
- 确保已激活对应的环境
- 需要修改脚本内部的 `output_base_dir`，评估结果将存放在该路径下的新建文件夹中
- 将 `"videopath"` 替换为实际的视频路径

#### 步骤 4: 计算最终得分

```bash
python /workspace/Vbench_EVA/cogvideo_batch_sampler/simple/calc_finnal_score.py --result_dir /path/to/evaluate_result
```

**参数说明**:
- `--result_dir`: 步骤 3 中生成的评估结果路径

**示例**:
```bash
python calc_finnal_score.py --result_dir ./Cogvideo5b_4steps_0.84/evaluate_result/checkpoint_120_fake_without_sparsity
```

## 快速开始

1. **修改环境路径**: 编辑 `run.sh` 第 3 行，设置正确的虚拟环境路径

2. **准备配置文件**: 复制并修改 `configs/` 下的配置文件，更新其中的路径设置

3. **执行采样**: 运行采样脚本生成视频

4. **评估视频**: 使用 VBench 评估脚本对生成的视频进行评估

5. **查看结果**: 运行得分计算脚本获取最终评分

## 环境配置

本项目需要配置两个主要环境：CogVideo 推理环境和 VBench 评估环境。

### 1. CogVideo 推理环境

参考 CogVideo 官方仓库配置推理环境：
- 仓库地址：https://github.com/THUDM/CogVideo
- 安装 PyTorch、diffusers 等依赖
- 下载 CogVideo 模型权重

基本安装命令：
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
```

### 2. VBench 评估环境

参考 VBench 官方仓库配置评估环境：
- 仓库地址：https://github.com/Vchitect/VBench
- 使用 `VBench/requirements.txt` 安装依赖

基本安装命令：
```bash
pip install vbench
# 或从 requirements.txt 安装
pip install -r VBench/requirements.txt
```

主要依赖包括：
- Pillow, numpy, opencv-python
- transformers, timm, pyiqa
- decord (视频处理)
- 其他评估相关库

### 环境说明

- 可以创建统一的虚拟环境同时包含两者的依赖
- 或分别创建两个环境，在不同步骤切换使用
- 推荐使用 Python 3.8+ 和 CUDA 环境以支持 GPU 加速

## 注意事项

- 所有涉及路径的地方都需要根据实际部署情况进行修改
- 确保有足够的磁盘空间存储生成的视频和评估结果
- 采样过程可能需要较长时间，建议使用 GPU 加速

## 相关文件说明

- **simple_multiprocess_sampler.py**: 多进程视频采样器，支持批量生成
- **calc_finnal_score.py**: 计算 VBench 评估的最终得分
- **modify_cogvideo.py**: 用于修改 CogVideo 模型的工具脚本
- **test_block_sparse_attention.py**: 块稀疏注意力机制的测试脚本
