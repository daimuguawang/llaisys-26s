# 作业4：NVIDIA RTX 5090 32GB CUDA 适配开发记录

## 概述

本文档记录了 LLaisys 项目在 NVIDIA RTX 5090 32GB（Blackwell 架构，sm_120）上的 CUDA 适配开发过程，包括构建配置、Runtime API 实现、8 个算子的 CUDA 内核编写、模型代码适配，以及开发过程中遇到的各种问题与解决方案。

**目标平台**：远程算力容器（通过 `scripts/ssh_compute.sh` 连接）
- GPU：NVIDIA RTX 5090 32GB（Blackwell，sm_120）
- CUDA：12.8（V12.8.61）
- OS：Linux x86_64
- 编译器：nvcc 12.8 / g++
- 构建系统：xmake

**最终结果**：所有测试通过
- `test_runtime.py --device nvidia` ✅
- 8 个算子测试（add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu，F32/F16/BF16）✅
- `test_infer.py --device nvidia --test`（128 tokens 与 HuggingFace 完全一致）✅
- 推理速度：LLAISYS GPU 0.30s vs HuggingFace 1.72s

---

## 一、开发步骤

### 1. 远程环境验证

通过 `ssh_compute.sh` 脚本连接远程算力容器，验证：
- `nvidia-smi`：RTX 5090 32GB，驱动版本 ≥ 560.xx，CUDA Version ≥ 12.8
- `nvcc --version`：CUDA 12.8 V12.8.61
- xmake 路径：`/root/.local/bin/xmake`（需 `--root` 和 `XMAKE_ROOT=y`）
- 模型路径：`/data/huggingface_home/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`

### 2. xmake 构建配置

#### 2.1 根 `xmake.lua` 修改

添加 `nv-gpu` 选项和 NVIDIA 设备依赖：

```lua
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end
```

在 `llaisys-device` 和 `llaisys-ops` 目标中移除对 nvidia 静态库的依赖（最终方案，见踩坑第5条），改为在 `llaisys` 共享库目标中直接编译 .cu 文件：

```lua
target("llaisys")
    set_kind("shared")
    ...
    add_files("src/llaisys/*.cc")

    if has_config("nv-gpu") then
        local cuda_home = os.getenv("CUDA_HOME") or "/usr/local/cuda"
        add_files("src/device/nvidia/*.cu", "src/ops/*/nvidia/*.cu")
        add_includedirs("src", "include")
        add_includedirs(cuda_home .. "/include")
        add_cugencodes("sm_120")
        add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
        add_links("cudart", "cublas", "cublasLt")
        add_linkdirs(cuda_home .. "/lib64")
        add_rpathdirs(cuda_home .. "/lib64")
    end
```

#### 2.2 `xmake/nvidia.lua` 配置

定义 `llaisys-device-nvidia` 和 `llaisys-ops-nvidia` 静态库目标（用于独立编译验证，最终构建不直接使用）：

```lua
local cuda_home = os.getenv("CUDA_HOME") or "/usr/local/cuda"

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("../src/device/nvidia/*.cu")
    add_includedirs("../include", "../src", {public = true})
    add_includedirs(cuda_home .. "/include")
    add_cugencodes("sm_120")
    add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
    on_install(function (target) end)
target_end()
```

### 3. CUDA Runtime API 实现

**文件**：`src/device/nvidia/nvidia_runtime_api.cu`

实现以下 API：

| API | CUDA Runtime 函数 |
|-----|------------------|
| `getDeviceCount()` | `cudaGetDeviceCount` |
| `setDevice(int)` | `cudaSetDevice` |
| `deviceSynchronize()` | `cudaDeviceSynchronize` |
| `createStream()` | `cudaStreamCreate` |
| `destroyStream()` | `cudaStreamDestroy` |
| `streamSynchronize()` | `cudaStreamSynchronize` |
| `mallocDevice(size_t)` | `cudaMalloc` |
| `freeDevice(void*)` | `cudaFree` |
| `mallocHost(size_t)` | `cudaMallocHost` |
| `freeHost(void*)` | `cudaFreeHost` |
| `memcpySync(dst, src, size, kind)` | `cudaMemcpy` |
| `memcpyAsync(dst, src, size, kind, stream)` | `cudaMemcpyAsync` |

### 4. cuBLAS 资源管理

**文件**：`src/device/nvidia/nvidia_resource.cuh` / `nvidia_resource.cu`

- `Resource` 类继承 `DeviceResource`，持有 `cublasHandle_t`
- `getCublasHandle(llaisysStream_t stream)`：返回线程局部 cuBLAS 句柄并绑定流

```cpp
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
private:
    cublasHandle_t _cublas_handle;
};

cublasHandle_t getCublasHandle(llaisysStream_t stream);
```

### 5. 8 个 CUDA 算子实现

每个算子包含 `.hpp`（声明）和 `.cu`（实现）文件，支持 FP32/FP16/BF16：

| 算子 | 文件路径 | 实现要点 |
|------|---------|---------|
| add | `src/ops/add/nvidia/` | 逐元素加法，1D grid/block |
| argmax | `src/ops/argmax/nvidia/` | warp shuffle 归约求最大值索引 |
| embedding | `src/ops/embedding/nvidia/` | 查表内核，每个 token 一个线程 |
| linear | `src/ops/linear/nvidia/` | cuBLAS `cublasGemmEx` + bias add kernel |
| rms_norm | `src/ops/rms_norm/nvidia/` | 两遍内核（求 mean squares → 归一化） |
| rope | `src/ops/rope/nvidia/` | 旋转位置编码，每两维一组 |
| self_attention | `src/ops/self_attention/nvidia/` | causal GQA，shared memory softmax |
| swiglu | `src/ops/swiglu/nvidia/` | SiLU 激活 + 逐元素乘法 |

**数据类型处理**：使用模板函数 `dev_to_float<T>` / `dev_from_float<T>` 在 F16/BF16 和 float 之间转换，确保与 CPU 参考实现的数值一致性。

### 6. 算子分发逻辑

在每个算子的 `op.cpp` 中添加 NVIDIA 设备分发：

```cpp
case LLAISYS_DEVICE_NVIDIA:
    llaisys::ops::nvidia::add(out, a, b, type, stream);
    break;
```

### 7. 模型代码适配

**文件**：`src/models/qwen2/model.cpp`

添加设备感知的辅助函数，替换 CPU 专用的 `std::memcpy` 和手动循环：

```cpp
// 设备感知拷贝（自动选择 D2D 或 H2H）
static void copyTensor(tensor_t dst, tensor_t src) {
    size_t bytes = src->numel() * src->elementSize();
    if (dst->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst->data(), src->data(), bytes);
    } else {
        llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
        auto api = llaisys::core::context().runtime().api();
        api->memcpy_sync(dst->data(), src->data(), bytes, LLAISYS_MEMCPY_D2D);
    }
}

// 设备到主机拷贝
static void copyToHost(void *host_dst, tensor_t src) { ... }
```

替换位置：
- KV cache 更新：`copyTensor(k_dst, k_rope)` / `copyTensor(v_dst, v_buf)`
- 残差加法：`ops::add(hidden, hidden, o_buf)` / `ops::add(hidden, hidden, down_buf)`
- 结果拷贝：`copyToHost(&result, max_idx)`

### 8. 编译与测试

构建命令：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export XMAKE_ROOT=y
xmake f --nv-gpu=y
xmake -j8
```

测试命令：
```bash
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
# ... 其他 ops 测试
python test/test_infer.py --device nvidia --model <model_path> --test
```

---

## 二、踩坑记录

### 坑1：macOS 资源派生文件（._*）

**现象**：从 macOS 打包上传的 tar 文件在 Linux 解压后出现大量 `._*` 开头的隐藏文件，导致 nvcc/g++ 编译时报无效字符错误。

**原因**：macOS 的 `tar` 默认会附加扩展属性（`xattr`）作为 AppleDouble 文件。

**解决**：
- 打包时设置 `COPYFILE_DISABLE=1` 环境变量
- 远程解压后执行 `find . -name "._*" -delete` 清理

```bash
COPYFILE_DISABLE=1 tar czf /tmp/nvidia_ops.tar.gz src/ops/*/nvidia/ ...
```

### 坑2：CUDA 算子文件包含路径深度错误

**现象**：编译 .cu 文件时报 `fatal error: ../../core/llaisys_core.hpp: No such file or directory`

**原因**：算子文件位于 `src/ops/<op>/nvidia/` 目录，深度为 3 层，但 include 路径写的是 `../../`（2 层），应为 `../../../`（3 层）。

**解决**：所有 .cu 文件中的 `../../` 改为 `../../../`：
```cpp
// 错误
#include "../../core/llaisys_core.hpp"
// 正确
#include "../../../core/llaisys_core.hpp"
```

### 坑3：`cuda_bfloat16.h` 头文件名错误

**现象**：编译报 `fatal error: cuda_bfloat16.h: No such file or directory`

**原因**：CUDA 12.8 的 bfloat16 头文件名是 **`cuda_bf16.h`**，不是 `cuda_bfloat16.h`。`cuda_fp16.h` 存在但 `cuda_bfloat16.h` 不存在，容易混淆。

**解决**：所有 .cu 文件中将 `#include <cuda_bfloat16.h>` 改为 `#include <cuda_bf16.h>`。

**验证方法**：`ls /usr/local/cuda/include/cuda_bf*.h` 确认实际文件名。

### 坑4：nvcc 编译需要 `-Xcompiler -fPIC`

**现象**：链接 `libllaisys.so` 时报：
```
relocation R_X86_64_TPOFF32 against `...handle' can not be used when making a shared object; recompile with -fPIC
```

**原因**：nvidia 静态库中的 .cu 文件编译时没有传 `-fPIC` 给宿主编译器。xmake 的 `add_cxflags("-fPIC")` 只作用于 C++ 编译器，不作用于 nvcc。nvcc 需要通过 `-Xcompiler` 前缀传递宿主编译器选项。

**解决**：在 `add_cuflags` 中添加 `"-Xcompiler -fPIC"`：
```lua
add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
```

**注意**：xmake 会默认忽略不认识的 cuflag，需要加 `{force = true}` 才能生效。如果分开写 `"-Xcompiler", "-fPIC"` 会被当作两个独立 flag 而 `-Xcompiler` 被丢弃。

### 坑5：CUDA device-link 步骤缺失（核心问题）

**现象**：`libllaisys.so` 构建成功，但 Python 加载时报：
```
OSError: undefined symbol: __cudaRegisterLinkedBinary_43448865_19_embedding_nvidia_cu_18c10fca
```

**原因**：每个 .cu 文件编译后会产生 `__cudaRegisterLinkedBinary_<hash>` 符号，该符号是 **undefined** 状态，需要通过 CUDA **device-link** 步骤生成定义。将 .cu 文件编译进静态库（.a）再链接进共享库（.so）时，xmake 不会执行 device-link，导致符号未定义。

`nm` 验证：
```
# 静态库中符号状态（U = undefined）
U __cudaRegisterLinkedBinary_fb1651a6_24_self_attention_nvidia_cu_26a4c2ff

# libcudart.so 中不提供该符号（不是运行时库的职责）
nm -D libcudart.so | grep cudaRegisterLinked  # 无输出
```

**解决**：将 .cu 文件从静态库目标移到 `llaisys` 共享库目标直接编译。xmake 对共享库目标中的 .cu 文件会自动执行 device-link 步骤（`devlinking.release llaisys_gpucode.cu.o`）。

**修改前**（不工作）：
```lua
-- .cu 文件在静态库中
target("llaisys-ops-nvidia")
    set_kind("static")
    add_files("../src/ops/*/nvidia/*.cu")
    
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-ops-nvidia")  -- 静态库依赖，无 device-link
```

**修改后**（工作）：
```lua
-- .cu 文件直接在共享库中
target("llaisys")
    set_kind("shared")
    if has_config("nv-gpu") then
        add_files("src/device/nvidia/*.cu", "src/ops/*/nvidia/*.cu")
        add_cugencodes("sm_120")
        add_cuflags(...)
    end
```

构建日志中出现 `devlinking.release` 表示 device-link 成功执行。

### 坑6：xmake `add_cuflags` 的 `{force = true}` 要求

**现象**：xmake 输出警告：
```
warning: add_cuflags("-Xcompiler") is ignored, please pass {force = true}
```

**原因**：xmake 对 cuflag 做合法性检查，`-Xcompiler` 不在已知列表中被忽略。

**解决**：加 `{force = true}` 强制设置，或将 `-Xcompiler -fPIC` 作为一个字符串传入。

### 坑7：测试中 torch mask 设备不匹配

**现象**：`test/ops/self_attention.py --device nvidia` 报：
```
RuntimeError: expected self and mask to be on the same device, but got mask on cpu and self on cuda:0
```

**原因**：`torch_self_attention` 函数中 `torch.ones(L, S, dtype=torch.bool)` 未指定 `device` 参数，默认在 CPU 创建 mask，而 attention 在 GPU 上。

**解决**：添加 `device=query.device`：
```python
# 修改前
temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S-L)
# 修改后
temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=S-L)
```

### 坑8：xmake `--root` 标志干扰配置命令

**现象**：`xmake f --nv-gpu=y --root` 配置失败，`--root` 被误解析为配置参数。

**解决**：通过环境变量 `XMAKE_ROOT=y` 代替命令行 `--root` 标志。

### 坑9：远程环境 CUDA 路径未在 PATH 中

**现象**：远程 SSH 执行命令时 `nvcc: command not found`。

**原因**：非交互式 SSH 会话不加载 `.bashrc`，CUDA 路径未在 PATH 中。

**解决**：每条远程命令前手动 export：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 三、关键文件清单

| 文件 | 说明 |
|------|------|
| `xmake.lua` | 根构建配置，添加 nv-gpu 选项和 CUDA 编译 |
| `xmake/nvidia.lua` | NVIDIA CUDA 目标编译配置 |
| `src/device/nvidia/nvidia_runtime_api.cu` | CUDA Runtime API 实现 |
| `src/device/nvidia/nvidia_resource.cu` / `.cuh` | cuBLAS 资源管理 |
| `src/ops/{add,argmax,embedding,linear,rms_norm,rope,self_attention,swiglu}/nvidia/*.cu` | 8 个 CUDA 算子实现 |
| `src/ops/*/op.cpp` | 算子分发逻辑（添加 NVIDIA 分支） |
| `src/models/qwen2/model.cpp` | 模型推理代码适配（设备感知 memcpy） |
| `test/ops/self_attention.py` | 测试修复（mask device） |
| `scripts/ssh_compute.sh` | 远程算力容器 SSH 访问脚本 |

---

## 四、性能数据

| 指标 | HuggingFace (GPU) | LLAISYS (GPU) |
|------|-------------------|---------------|
| 推理时间（128 tokens） | 1.72s | 0.30s |
| Token 一致性 | — | 完全一致 ✅ |

---

## 五、经验总结

1. **CUDA 头文件名**：`cuda_bf16.h`（不是 `cuda_bfloat16.h`），与 `cuda_fp16.h` 命名风格不一致，需特别注意。

2. **xmake + CUDA 共享库**：.cu 文件必须直接在 `shared` 目标中编译，不能放在 `static` 目标中再依赖。xmake 只对共享库目标执行 device-link。

3. **nvcc 的 `-fPIC`**：需要 `-Xcompiler -fPIC` 前缀，且 xmake 中需要 `{force = true}`。

4. **macOS → Linux 传输**：始终使用 `COPYFILE_DISABLE=1` 避免 AppleDouble 文件污染。

5. **设备感知代码**：在 CPU/GPU 混合代码中，`std::memcpy` 只适用于 CPU，GPU 需要 `cudaMemcpy` 或 Runtime API 的 `memcpy_sync`。
