-- xmake/nvidia.lua
-- NVIDIA CUDA (Blackwell sm_120) and MXMACA (MetaX C500) build configuration.

local is_mx = has_config("mx-gpu")
local cuda_home = os.getenv("CUDA_HOME") or "/usr/local/cuda"
local maca_path = os.getenv("MXMACA_PATH") or "/opt/maca-3.2.1"
local mxcc = maca_path .. "/mxgpu_llvm/bin/mxcc"

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if is_mx then
        -- MXMACA: compile .cu files as C++ to bypass CUDA detection
        add_files("../src/device/nvidia/*.cu", {kind = "cxx"})
    else
        add_files("../src/device/nvidia/*.cu")
    end
    add_includedirs("../include", "../src", {public = true})

    if is_mx then
        -- MXMACA (MetaX C500) configuration
        add_includedirs(maca_path .. "/include")
        add_includedirs(maca_path .. "/include/mcblas")
        set_toolchains("cxx", mxcc)
        add_cxxflags("-x", "maca", "-offload-arch=native", "-O3", "-fPIC", "--maca-path=" .. maca_path, {force = true})
        add_ldflags("--maca-path=" .. maca_path, {force = true})
    else
        -- NVIDIA CUDA configuration
        add_includedirs(cuda_home .. "/include")
        add_cugencodes("sm_120")
        add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
        add_cxxflags("-O3")
    end

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if is_mx then
        -- MXMACA: compile .cu files as C++ to bypass CUDA detection
        add_files("../src/ops/*/nvidia/*.cu", {kind = "cxx"})
    else
        add_files("../src/ops/*/nvidia/*.cu")
    end
    add_includedirs("../include", "../src", {public = true})

    if is_mx then
        -- MXMACA (MetaX C500) configuration
        add_includedirs(maca_path .. "/include")
        add_includedirs(maca_path .. "/include/mcblas")
        set_toolchains("cxx", mxcc)
        add_cxxflags("-x", "maca", "-offload-arch=native", "-O3", "-fPIC", "--maca-path=" .. maca_path, {force = true})
        add_ldflags("--maca-path=" .. maca_path, {force = true})
        add_links("mcblas")
        add_linkdirs(maca_path .. "/lib")
    else
        -- NVIDIA CUDA configuration
        add_includedirs(cuda_home .. "/include")
        add_cugencodes("sm_120")
        add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
        add_cxxflags("-O3")
        add_links("cudart", "cublas", "cublasLt")
        add_linkdirs(cuda_home .. "/lib64")
    end

    on_install(function (target) end)
target_end()
