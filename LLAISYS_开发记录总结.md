# LLAISYS 开发记录总结

> 本文档汇总了 LLAISYS 框架从 Tensor 基础操作、CPU 算子实现、Qwen2 模型推理到双平台 GPU 适配的完整开发记录。

---

## 项目全景

LLAISYS 是一个自研深度学习推理框架，本次开发通过四个递进式作业，最终在 NVIDIA RTX 5090 和沐曦曦云 C500 两款 GPU 上实现了 DeepSeek-R1-Distill-Qwen-1.5B 模型的端到端推理，且推理输出与 HuggingFace 参考实现完全一致。

### 作业关系与技术栈

```
作业1：Tensor 基础操作        → 内存视图、维度变换（框架基石）
    ↓
作业2：7 个 CPU 算子实现      → argmax/embedding/linear/rms_norm/rope/self_attention/swiglu
    ↓
作业3：Qwen2 模型推理         → C++ 前向 + C API + Python 绑定 + KV Cache 增量推理
    ↓
作业4：双平台 GPU 适配（两部分）
    ├── 4.1 NVIDIA RTX 5090 32GB（CUDA 原生 kernel）
    └── 4.2 曦云 C500 16GB（MXMACA Host-Side 混合实现）
```

### 最终性能对比

| 平台 | 推理时间（128 tokens） | Token 一致性 | 备注 |
|------|----------------------|-------------|------|
| CPU（作业3） | ~18 分钟 | 完全一致 ✅ | 逐 token bf16 计算，无优化 |
| NVIDIA RTX 5090（作业4.1） | 0.30s | 完全一致 ✅ | 比 HuggingFace GPU 快 5.7x |
| HuggingFace GPU（参考） | 1.72s | — | PyTorch 参考实现 |
| 曦云 C500（作业4.2） | 9.29s | 完全一致 ✅ | ~13.8 tokens/s，Host-Side 实现 |

---

## 作业1：Tensor 基础操作

### 任务概述

实现 `src/tensor/tensor.cpp` 中的 5 个函数，构成框架张量操作基石：

| 函数 | 功能 | 难度 |
|------|------|------|
| `load` | 从主机内存拷贝数据到张量（支持 CPU/GPU） | 基础 |
| `isContiguous` | 判断张量是否在内存中连续 | 基础 |
| `permute` | 维度重排（不搬数据，只改 meta） | 基础 |
| `slice` | 沿指定维度切片（不搬数据，改 offset 和 shape） | 基础 |
| `view` | 改变张量形状（连续直接算 strides，非连续需双指针检查兼容性） | ⭐ 较难 |

测试链路：`(3,4,5)` i64 张量的 `load → view(6,10) → permute(2,0,1) → slice(2,1,4)`。

### 实现要点

- **load**：参考 `debug()` 的 D2H 拷贝写法，先 `setDevice` 切设备，再 `memcpy_sync` 拷贝。CPU 走 H2H，GPU 走 H2D。
- **isContiguous**：从最内层维度向外遍历，检查非 size-1 维度的 stride 是否等于"内层维度大小之积"。
- **permute**：按 `order` 重排 shape 和 strides，不搬数据，需校验 order 是合法排列。
- **slice**：被切维度 shape 改为 `end - start`，strides 不变，`_offset` 加字节偏移。
- **view（最复杂）**：
  - 快速路径：连续张量从内向外计算新 strides。
  - 非连续路径：双指针算法从右向左扫描新旧维度，合并/拆分旧维度匹配新形状，核心是维护"已累积元素数 r2"和"基准步长 stride"，检查旧维度在内存上是否连续。能正确拒绝 `(2,3,5) strides(30,10,1) → view(2,15)` 这类不兼容情况。

### 踩坑记录

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| 代码写在 .hpp | 编译报错 `not declared` | 实现写进头文件，不在类作用域 | 实现放 `.cpp`，`.hpp` 只保留声明 |
| heredoc 粘贴截断 | 大段内容粘贴乱码 | 终端粘贴缓冲区有限 | 用 nano 编辑器或 Python 脚本写入 |
| nano 旧代码未删净 | 编译通过但报 `Unimplemented` | 新旧代码并存 | nano 全选删除再粘贴，grep 确认 |
| Python 加载旧 .so（核心坑） | 测试仍报旧代码错误 | `pip install -e .` 装到系统路径，`xmake install` 只更新项目目录 | 覆盖系统路径 .so 或用 `PYTHONPATH` 优先加载项目目录 |

### 经验总结

1. 分清 `.hpp`（声明）和 `.cpp`（实现）。
2. 大段代码别用 heredoc，终端粘贴缓冲区有限。
3. 改完代码用 `grep` 验证文件内容生效，别只看编译通过。
4. 注意 Python 模块加载路径，`pip install -e .` 后系统路径有旧副本，调试时先 `print(llaisys.__file__)` 确认加载哪个。
5. xmake 的 `cache compiling` 不一定是缓存问题，可能只是 ccache 正常输出。

---

## 作业2：CPU 算子实现

### 任务概述

实现 LLAISYS 框架中 7 个算子的 CPU 版本，支持 Float32、Float16、BFloat16 三种数据类型：argmax、embedding、linear、rms_norm、rope、self_attention（含 GQA + Causal Mask）、swiglu。

### 各算子实现要点

| 算子 | 核心公式/逻辑 | 注意事项 |
|------|-------------|---------|
| **argmax** | 遍历找最大值及索引 | 用 float 中间值比较避免半精度精度问题；max_idx 为 int64 |
| **embedding** | `out[i] = weight[index[i]]` | index 必须是 int64；按行拷贝 embedding 向量 |
| **linear** | `Y = X · Wᵀ + b` | weight 形状 [N,K] 不需转置存储，按 W[n][k] 索引 |
| **rms_norm** | `y = x * rsqrt(mean(x²)+eps) * weight` | 按行归一化，先求平方均值再开方取倒 |
| **rope** | 对半维度做旋转变换 | 角度公式 `pos / theta^(2j/d)`；前半维 a 和后半维 b 做旋转 |
| **self_attention** | `softmax(Q·Kᵀ·scale + mask) · V` | 支持 GQA 头映射；causal mask 需考虑 KV cache 偏移 |
| **swiglu** | `out = up * gate * sigmoid(gate)` | 逐元素运算，sigmoid 用 `1/(1+exp(-x))` |

### 踩坑记录

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| xmake 禁止 root | 报错不支持 root | 安全考虑 | 加 `--root` 参数 |
| install 目标名错误 | `'install' is not a valid target` | xmake.lua 重写了 install 逻辑 | 指定 target 名 `xmake install llaisys` |
| 改代码测试跑旧结果 | 仍报 `Unimplemented` | C++ 编译型，需重新编译→复制 so→重装 | 每次改完走完整三步 |
| unused variable 编译失败 | `-Werror=unused-variable` | 项目开启 `-Werror` | 删掉未使用变量 |
| rearrange 链接失败 | `undefined symbol` | 非作业要求的算子桩函数未生成符号 | 补空实现保证链接通过 |
| **Causal Mask 逻辑错误** | qlen=5, kvlen=11 结果不对 | KV Cache 场景 mask 不是简单 `j > i` | mask 偏移量改为 `causal_offset = kvlen - qlen` |
| Git Push DNS 问题 | `Could not resolve host` | 容器网络不稳定 | 多试/固定 IP 到 hosts/关闭 SSL 验证 |

### 经验总结

1. **半精度运算统一转 float**：f16/bf16 计算时先 cast 到 float，算完再 cast 回去。
2. **编译型语言改完必须重新编译**：C++ 改完要走完整编译→链接→部署流程。
3. **注意编译器警告**：`-Werror` 下所有 warning 都是 error。
4. **算子实现前先读测试代码**：PyTorch 参考实现就是标准答案，特别是 mask、维度顺序细节。
5. **Causal Mask 要考虑 KV Cache**：推理时 `kvlen >= qlen`，mask 偏移是 `kvlen - qlen`。
6. **GQA 头映射**：`kv_h = q_h / (n_heads / n_kv_heads)`，相邻 query 头共享同一组 kv 头。

---

## 作业3：Qwen2 模型推理实现

### 任务概述

基于 llaisys 框架实现 DeepSeek-R1-Distill-Qwen-1.5B（Qwen2 架构）的端到端推理，包括 C++ 模型前向、C API 封装、Python ctypes 绑定、权重加载、增量推理 + KV Cache。最终推理输出与 HuggingFace 参考实现**完全一致**。

### 架构概览

```
C++ 层:
  src/ops/           # 作业2：7个算子
  src/models/qwen2/  # 作业3：模型实现
    ├── model.hpp    # Qwen2Model 类定义
    └── model.cpp    # 前向传播、KV Cache、增量推理
  src/llaisys/qwen2.cc  # C API 封装

Python 层:
  python/llaisys/libllaisys/models.py  # ctypes 结构体定义
  python/llaisys/models/qwen2.py       # 模型加载、权重映射、generate
```

### 实现要点

- **model.hpp**：定义 Qwen2Model 类，包含所有中间 tensor buffer、KV cache、权重结构体。
- **model.cpp**：
  - `initBuffers()`：按 seq_len=1 分配所有中间张量，KV cache 按 `[maxseq, n_kv_head, head_dim]` 分配。
  - `forwardLayer()`：单层前向（RMSNorm → QKV Linear → RoPE → KV Cache 更新 → SelfAttention → O Linear → 残差 → RMSNorm → Gate/Up Linear → SwiGLU → Down Linear → 残差）。
  - `infer()`：逐 token 循环处理（prefill 和 decode 统一），最后 RMSNorm + LM Head + Argmax。
- **qwen2.cc**：C API 封装（Create/Weights/Infer/Destroy）。
- **Python 绑定**：ctypes 定义 Meta/Weights 结构体，用 torch 加载 safetensors 权重。

### 踩坑记录

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| **权重指针不同步（核心 bug）** | 权重加载成功但推理段错误 | `getWeights()` 在初始化时调用，全局权重是 nullptr 被复制进 struct，Python 写入 struct 指针但模型成员变量未更新 | `infer()` 开头同步三个全局权重指针 |
| bfloat16 权重加载 | `TypeError: bfloat16 not understood` | numpy 不原生支持 bfloat16 | 用 `framework="pt"` torch 加载，`view(uint16).numpy()` 读原始字节 |
| so 软链接指向错误目录 | 跑起来还是旧代码 | 软链接指向另一个 build 目录 | `readlink -f` 查真实路径，直接覆盖 |
| xmake 缓存致旧代码 | 编译秒过但行为没变 | ccache 缓存 | `xmake f -c` 清缓存 + `xmake -r` 强制重编 |
| getTensor 空指针解引用 | 段错误地址 0x38 | attention 层无 bias，nullptr 解引用 | `return h ? h->tensor : nullptr` |
| safetensors slice API 不一致 | `no attribute 'shape'/'data'` | numpy 后端 slice API 非 numpy 风格 | 放弃 numpy 后端，用 `framework="pt"` |
| 模型路径是 HF 缓存子目录 | `FileNotFoundError: config.json` | HF 缓存结构 `snapshots/<hash>/` | 指向 snapshots 子目录 |

### 调试技巧

1. **addr2line 定位段错误**：`addr2line -e libllaisys.so 0x1cb44 -f`（release 版可能显示 `??:?`）。
2. **分层验证权重加载**：单独写脚本检查每个权重指针非空。
3. **检查 so 是否更新**：`ls -la` 看时间戳，`readlink -f` 看软链接真实路径。
4. **逐 token 推理简化调试**：prefill 也改逐 token 处理（seq_len=1）。

### 最终结果

- 模型：DeepSeek-R1-Distill-Qwen-1.5B（28层，hidden=1536，GQA 12头/2 KV头，bf16）
- 输入："Who are you?"
- 输出：与 HuggingFace 参考实现 tokens 完全一致
- 速度：CPU 上约 18 分钟（无优化，逐 token bf16 计算）

---

## 作业4：双平台 GPU 适配

作业4 分为两个部分，分别在 NVIDIA RTX 5090 和沐曦曦云 C500 上完成 GPU 适配。

### 4.1 NVIDIA RTX 5090 32GB CUDA 适配

#### 目标平台

- GPU：NVIDIA RTX 5090 32GB（Blackwell，sm_120）
- CUDA：12.8（V12.8.61）
- 编译器：nvcc 12.8 / g++
- 构建系统：xmake

#### 开发内容

1. **xmake 构建配置**：添加 `nv-gpu` 选项，在 `llaisys` 共享库目标中直接编译 `.cu` 文件，设置 `sm_120` 架构、`-Xcompiler -fPIC`、cuBLAS 链接。

2. **CUDA Runtime API 实现**（`src/device/nvidia/nvidia_runtime_api.cu`）：实现设备管理、流管理、内存分配/释放、同步/异步拷贝等 12 个 API。

3. **cuBLAS 资源管理**：`Resource` 类持有 `cublasHandle_t`，线程局部句柄绑定流。

4. **8 个 CUDA 算子**（原生 device kernel）：

| 算子 | 实现要点 |
|------|---------|
| add | 逐元素加法，1D grid/block |
| argmax | warp shuffle 归约求最大值索引 |
| embedding | 查表内核，每个 token 一个线程 |
| linear | cuBLAS `cublasGemmEx` + bias add kernel |
| rms_norm | 两遍内核（求 mean squares → 归一化） |
| rope | 旋转位置编码，每两维一组 |
| self_attention | causal GQA，shared memory softmax |
| swiglu | SiLU 激活 + 逐元素乘法 |

5. **模型代码适配**：添加设备感知的 `copyTensor`（自动选 D2D 或 H2H）和 `copyToHost`，替换 CPU 专用的 `std::memcpy`。

#### 踩坑记录

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| macOS 资源派生文件 | `._*` 文件致编译错误 | macOS tar 附加 AppleDouble | `COPYFILE_DISABLE=1` 打包 + `find . -name "._*" -delete` |
| 包含路径深度错误 | `fatal error: No such file` | 算子在 3 层目录，include 写 2 层 | `../../` 改 `../../../` |
| cuda_bfloat16.h 找不到 | `fatal error` | CUDA 12.8 头文件名是 `cuda_bf16.h` | 改为 `#include <cuda_bf16.h>` |
| nvcc 缺 -fPIC | 链接报 `relocation TPOFF32` | `add_cxflags` 不作用于 nvcc | `add_cuflags("-Xcompiler -fPIC", {force=true})` |
| **device-link 缺失（核心）** | `undefined symbol: __cudaRegisterLinkedBinary` | .cu 在静态库中不执行 device-link | .cu 文件直接放共享库目标编译 |
| add_cuflags 被忽略 | warning: is ignored | xmake 检查 cuflag 合法性 | 加 `{force = true}` |
| torch mask 设备不匹配 | `mask on cpu and self on cuda` | `torch.ones` 未指定 device | 添加 `device=query.device` |
| --root 干扰配置命令 | `xmake f` 配置失败 | `--root` 被误解析为配置参数 | 用 `XMAKE_ROOT=y` 环境变量 |
| 远程 CUDA 路径不在 PATH | `nvcc: command not found` | 非交互 SSH 不加载 .bashrc | 每条命令前手动 export |

#### 验证结果

- `test_runtime.py --device nvidia` ✅
- 8 个算子测试（F32/F16/BF16）✅
- `test_infer.py --device nvidia --test`（128 tokens 与 HuggingFace 完全一致）✅
- 推理速度：LLAISYS GPU 0.30s vs HuggingFace 1.72s

---

### 4.2 曦云 C500 16GB MXMACA 适配

#### 目标平台

| 项目 | 规格 |
|------|------|
| GPU | 曦云 C500（XCORE 1.0 架构） |
| 显存 | 16GB HBM2e |
| Warp Size | 64（NVIDIA 为 32） |
| 软件栈 | MXMACA 3.2.1（兼容 CUDA 11.x/12.x） |
| 编译器 | clang++-15 |

#### 总体策略：Host-Side 混合实现

由于 MXMACA 的 `mxcc` 编译器脚本存在语法错误（Python f-string 未闭合），且 `clang++-15 -x cuda` 编译时出现找不到 `libdevice`、目标 triple 不识别等问题，无法直接编译包含 `__global__` kernel 的 `.cu` 文件。

因此采用 **Host-Side 混合实现策略**：
- 将 `.cu` 文件重命名为 `.cpp`，以普通 C++ 代码编译
- 算子计算在主机端完成，通过 `mcMemcpy` 在设备/主机间拷贝数据
- `linear` 算子 GEMM 部分使用 MCBLAS 在设备端完成

#### 开发内容

1. **xmake 配置**：添加 `mx-gpu` 选项，创建 `.mx_build/` 目录（从 `.cu` 重命名为 `.cpp`），绕过 xmake CUDA 规则检测。

2. **CUDA 兼容层**（`cuda_compat.hpp`）：将 CUDA API 映射到 MXMACA 的 `mc_*` API，包括类型映射（`half → mcblas_half`）、函数映射（`cudaMalloc → mcMalloc`）、Warp Size（64）、warp shuffle 兼容宏等。

3. **8 个算子 Host-Side 实现**：

| 算子 | 实现方式 | 特殊处理 |
|------|---------|----------|
| add | Host-Side 逐元素加法 | 支持 F32/F16/BF16 |
| argmax | Host-Side 扫描求最大值 | 结果通过 `mxm_copy_to_device` 写回 |
| embedding | Host-Side 查表拷贝 | 先拷贝 index 到主机，再查找 weight 行 |
| linear | MCBLAS GEMM + Host bias | GEMM 在设备端，bias 加法在主机端 |
| rms_norm | Host-Side 均方根归一化 | 逐行计算 |
| rope | Host-Side 三角函数计算 | 逐位置逐头计算 |
| self_attention | Host-Side 软注意力 | 含 causal mask 和 softmax |
| swiglu | Host-Side 逐元素激活 | SiLU 激活函数 |

4. **测试框架适配**：
   - `test_utils.py`：PyTorch 参考数据在 CPU 上生成，使用 H2D 拷贝。
   - `test_infer.py`：PyTorch 模型在 CPU 上加载，添加 `--skip-hf` 选项。
   - `qwen2.py`：用 `safetensors.torch` 替代 `safetensors.numpy` 读取权重。

#### 踩坑记录

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| mxcc 编译器脚本损坏 | 无法执行 | Python f-string 未闭合 | 绕过 mxcc，直接用 clang++-15 |
| CUDA 头文件找不到 | `cuda_runtime.h not found` | MXMACA 用 `mcr/mc_runtime.h` | cuda_compat.hpp 条件编译映射 |
| stdint 类型未定义 | `uint8_t` 未定义 | C 模式未包含 stdint | 包含 MXMACA 头文件前先包含 `<cstdint>` |
| Warp Size 差异 | 规约结果不正确 | C500 warp size 64，代码硬编码 32 | 定义 `WARP_SIZE` 为 64（Host-Side 后不再影响） |
| `__shfl_down_sync` 不兼容 | 未定义 | MXMACA 不支持 mask 参数 | 宏映射 `__shfl_down(val, offset)` |
| 严格别名违规 | 编译警告变错误 | MXMACA 类型双关代码 | 添加 `-fno-strict-aliasing` |
| Device-Side 变量未定义 | `blockIdx` 未声明 | .cu 重命名 .cpp 后非 CUDA 编译 | Host-Side 实现，避免 device 变量 |
| MCBLAS 数据类型映射错误 | `cudaDataType_t` 未声明 | MXMACA 用 `macaDataType_t` | 类型映射 + 常量宏定义 |
| argmax 设备指针直接赋值 | 测试失败 | 设备指针不能主机端解引用 | 用 `mxm_copy_to_device` 替代 |
| embedding 主机访问设备指针 | 测试失败 | 主机端遍历设备指针 | 先拷贝 index 到主机缓冲区 |
| numpy 不支持 bfloat16 | `TypeError` | numpy 1.26.4 不支持 | 改用 `safetensors.torch` |
| PyTorch 无法用 C500 GPU | 加载失败 | PyTorch 不支持 MXMACA | PyTorch 在 CPU 上运行 |
| xmake CUDA 检测干扰 | 检测 CUDA SDK 失败 | MXMACA 非标准 CUDA | `.cu` 重命名 `.cpp` 绕过检测 |

#### 验证结果

| 测试项 | 状态 |
|--------|------|
| Runtime 测试 | ✅ 通过 |
| 8 个算子（F32/F16/BF16） | ✅ 全部通过 |
| 推理测试（16/128 tokens） | ✅ Token 完全匹配 PyTorch |

性能数据：
- llaisys C500 推理（128 tokens）：9.29s，约 13.8 tokens/s
- llaisys C500 推理（16 tokens）：3.50s
- PyTorch CPU 推理（16 tokens）：74.35s
- 加速比：约 8x（vs CPU）

#### 后续优化方向

1. **Native Device Kernel**：将简单逐元素算子改为原生 MXMACA device kernel，消除 D2H/H2D 搬运开销。
2. **linear bias device kernel**：设备端添加 bias 加法，避免每层 GEMM 后往返。
3. **self_attention 在线 softmax**：Flash Attention 风格，减少中间存储。
4. **性能目标**：解码速度提升至 > 20 tokens/s。

---

## 整体经验总结

### 开发流程方法论

1. **先读测试代码再写实现**：测试中的 PyTorch 参考实现就是标准答案，特别是 mask、维度顺序等细节。
2. **分层验证**：从算子级 → 层级 → 模型级逐步验证，每层单独测试通过再集成。
3. **改完代码验证文件内容**：用 `grep` 确认改动生效，别只看编译通过。

### 跨作业复现的高频坑

| 坑类型 | 出现作业 | 核心教训 |
|--------|---------|---------|
| Python 加载旧 .so / 软链接指向错误 | 作业1、2、3 | 每次编译后确认 `readlink -f` 真实路径，覆盖系统路径 |
| xmake 缓存/重编问题 | 作业2、3 | `xmake f -c` 清缓存 + `xmake -r` 强制重编 |
| heredoc 粘贴截断 | 作业1、3 | 大段代码用编辑器或 Python 脚本，不依赖 heredoc |
| bfloat16 加载问题 | 作业3、4.2 | numpy 不支持 bf16，统一用 torch `framework="pt"` 加载 |

### GPU 适配关键经验

1. **CUDA 头文件名**：`cuda_bf16.h`（不是 `cuda_bfloat16.h`），与 `cuda_fp16.h` 命名风格不一致。
2. **xmake + CUDA 共享库**：`.cu` 文件必须直接在 `shared` 目标中编译，不能放 `static` 目标再依赖（device-link 只对共享库执行）。
3. **nvcc 的 -fPIC**：需要 `-Xcompiler -fPIC` 前缀，xmake 中需 `{force = true}`。
4. **macOS → Linux 传输**：始终用 `COPYFILE_DISABLE=1` 避免 AppleDouble 文件污染。
5. **设备感知代码**：CPU/GPU 混合代码中 `std::memcpy` 只适用 CPU，GPU 需 Runtime API 的 `memcpy_sync`。
6. **国产 GPU 适配策略**：当编译器工具链不完善时，Host-Side 混合实现是可行的降级方案，配合兼容层宏映射可复用大部分 CUDA 代码。

### 项目文件全景

| 作业 | 关键文件 | 说明 |
|------|---------|------|
| 作业1 | `src/tensor/tensor.cpp` | 5 个 Tensor 基础函数 |
| 作业2 | `src/ops/{argmax,embedding,linear,rms_norm,rope,self_attention,swiglu}/op.cpp` | 7 个 CPU 算子 |
| 作业2 | `src/ops/rearrange/op.cpp` | 补空实现保证链接 |
| 作业3 | `src/models/qwen2/model.hpp` / `model.cpp` | Qwen2Model 类与前向实现 |
| 作业3 | `src/llaisys/qwen2.cc` | C API 封装 |
| 作业3 | `python/llaisys/libllaisys/models.py` | ctypes 结构体定义 |
| 作业3 | `python/llaisys/models/qwen2.py` | Python 模型加载与推理 |
| 作业4.1 | `xmake.lua` / `xmake/nvidia.lua` | NVIDIA CUDA 构建配置 |
| 作业4.1 | `src/device/nvidia/nvidia_runtime_api.cu` | CUDA Runtime API |
| 作业4.1 | `src/device/nvidia/nvidia_resource.cu` | cuBLAS 资源管理 |
| 作业4.1 | `src/ops/*/nvidia/*.cu` | 8 个 CUDA 算子 |
| 作业4.1 | `src/models/qwen2/model.cpp` | 设备感知适配 |
| 作业4.2 | `src/device/nvidia/cuda_compat.hpp` | CUDA/MXMACA 兼容层 |
| 作业4.2 | `src/ops/*/nvidia/*.cu`（MXMACA 分支） | 8 个 Host-Side 算子 |
| 作业4.2 | `scripts/ssh_compute_c500.sh` | C500 远程访问脚本 |
