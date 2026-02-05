# Spec-Bench NPU 适配规格文档

## 概述

对 Spec-Bench 框架实现华为 Ascend NPU 适配，使其能够在 NPU 上运行推理评测。

## 适配范围

### 目标方法
- baseline: 基础自回归推理
- eagle: EAGLE 投机解码方法

### 排除方法
以下方法不在本次适配范围内：
- eagle2, eagle3
- medusa, hydra
- sps, pld, rest
- lookahead, space
- recycling, samd

### 支持的模型架构
- LLaMA (Dense 模型)
- Mixtral (MoE 模型)

## 技术规格

### NPU 环境
- 硬件: Ascend 910B
- 软件栈: CANN 8.x
- Python 库: torch_npu

### 精度支持
- float16
- bfloat16
- float32

### 设备检测策略
```python
# 自动检测设备
if torch.npu.is_available():
    device = "npu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

### 错误处理
- 当指定使用 NPU 但 NPU 不可用时：直接报错退出
- 不支持自动回退到 CUDA

## 实现计划

### Phase 1: 公共层适配

**目标**: 建立 NPU/CUDA 兼容的基础设施

**用户故事 US-1.1**: 设备检测模块
- 创建 `evaluation/device_utils.py` 公共模块
- 实现 `get_device()` 函数，自动检测可用设备
- 实现 `get_device_synchronize()` 函数，封装 `torch.cuda.synchronize()` / `torch.npu.synchronize()`
- 验收标准: 在 NPU 环境下 `get_device()` 返回 "npu"

**用户故事 US-1.2**: 环境变量处理
- 支持 `ASCEND_RT_VISIBLE_DEVICES` 环境变量
- 兼容现有 `CUDA_VISIBLE_DEVICES` 处理逻辑
- 验收标准: 设置 `ASCEND_RT_VISIBLE_DEVICES=0` 后程序能正确识别

**Git Commit**: `feat(npu): add device detection and utility module`

### Phase 2: Baseline 适配

**目标**: 使基础推理流程在 NPU 上运行

**用户故事 US-2.1**: 修改 evaluation/eval.py
- 将 `.to("cuda")` 替换为 `.to(device)`，device 从公共模块获取
- 将 `torch.cuda.synchronize()` 替换为公共函数调用
- 验收标准: `python -m evaluation.inference_baseline` 在 NPU 上能完成推理

**用户故事 US-2.2**: 修改 evaluation/inference_baseline.py
- 适配设备相关代码
- 验收标准: 生成的 jsonl 文件包含正确的推理结果

**Git Commit**: `feat(npu): adapt baseline inference for NPU`

### Phase 3: EAGLE 适配

**目标**: 使 EAGLE 投机解码在 NPU 上运行

**用户故事 US-3.1**: 修改 model/eagle/ea_model.py
- 适配设备相关代码
- 确保 KVCache 操作兼容 NPU
- 验收标准: EAGLE 模型能在 NPU 上加载

**用户故事 US-3.2**: 修改 model/eagle/modeling_llama_kv.py
- 适配 attention 计算中的设备相关代码
- 确保 tree_mask 处理兼容 NPU
- 验收标准: EAGLE 推理流程完整运行

**用户故事 US-3.3**: 修改 model/eagle/modeling_mixtral_kv.py (如果存在)
- 适配 MoE 路由计算
- 确保专家选择逻辑兼容 NPU
- 验收标准: Mixtral 模型在 NPU 上能完成推理

**用户故事 US-3.4**: 修改 evaluation/inference_eagle.py
- 适配设备相关代码
- 验收标准: `python -m evaluation.inference_eagle` 在 NPU 上能完成推理

**Git Commit**: `feat(npu): adapt EAGLE speculative decoding for NPU`

## 修改策略

### 代码组织
- 抽取设备无关代码到公共模块 `evaluation/device_utils.py`
- 原地修改现有文件，不新建 NPU 专用版本
- 通过运行时检测切换设备

### 关键修改点

| 文件 | 修改内容 |
|------|----------|
| evaluation/device_utils.py | 新建: 设备检测、同步函数 |
| evaluation/eval.py | .to("cuda") -> .to(device), synchronize |
| evaluation/inference_baseline.py | 设备适配 |
| evaluation/inference_eagle.py | 设备适配 |
| model/eagle/ea_model.py | 设备适配、KVCache 兼容 |
| model/eagle/modeling_llama_kv.py | attention/tree_mask 适配 |
| model/eagle/modeling_mixtral_kv.py | MoE 路由适配 |

## 验收标准

### 功能验收
- 能在 NPU 上正常运行 baseline 和 eagle 推理
- 能产出正确的推理结果文件
- 暂不要求性能对标 CUDA

### 测试方法
1. rsync 整个 Spec-Bench 文件夹到 `ssh root@192.168.0.180:/home/z00929669/Spec-Bench`
2. 进入 192.168.0.180 的 spec-bench 容器
3. 进入 `/home/z00929669/Spec-Bench`
4. 运行 `sh run.sh` 测试

## Git 工作流

### Commit 规范
- 使用 Conventional Commits 格式
- 示例:
  - `feat(npu): add device detection and utility module`
  - `feat(npu): adapt baseline inference for NPU`
  - `feat(npu): adapt EAGLE speculative decoding for NPU`

### 工作流程
1. 每个 Phase 完成后进行 git commit
2. 仅本地 commit，不自动 push
3. 最后由用户手动决定何时 push

## 依赖要求

### 新增依赖
```
torch_npu  # 需要与 CANN 版本匹配
```

### 现有依赖保持不变
- torch>=2.1.1
- transformers==4.37.1
- 其他 requirements.txt 中的依赖

## 风险与注意事项

1. **算子兼容性**: 部分 CUDA 算子可能在 NPU 上没有直接对应，需要调研 torch_npu API
2. **精度差异**: NPU 和 CUDA 的浮点计算可能存在微小差异
3. **无参考代码**: 需要从头调研 torch_npu 的使用方式
