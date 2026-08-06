-- xmake/nvidia.lua
-- NVIDIA CUDA (Blackwell sm_120) build configuration for RTX 5090.

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

    -- CUDA compile config (sm_120 = Blackwell / RTX 5090)
    add_cugencodes("sm_120")
    add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
    add_cxxflags("-O3")

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

    add_files("../src/ops/*/nvidia/*.cu")
    add_includedirs("../include", "../src", {public = true})
    add_includedirs(cuda_home .. "/include")

    add_cugencodes("sm_120")
    add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
    add_cxxflags("-O3")

    -- link CUDA math libs (needed when resolving symbols used by .cu kernels)
    add_links("cudart", "cublas", "cublasLt")
    add_linkdirs(cuda_home .. "/lib64")

    on_install(function (target) end)
target_end()
