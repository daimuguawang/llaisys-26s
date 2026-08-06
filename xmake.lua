add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA / MXMACA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("mx-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX C500 GPU (MXMACA)")
option_end()

if has_config("nv-gpu") or has_config("mx-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    if has_config("mx-gpu") then
        add_defines("USE_MXMACA")
        -- For MXMACA, we don't use nvidia.lua to avoid CUDA SDK detection
        -- MXMACA compilation is handled directly in the main targets
    else
        includes("xmake/nvidia.lua")
    end
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/models/*/*.cpp")

    on_install(function (target) end)
target_end()

-- MXMACA specific targets (only built when mx-gpu is enabled)
-- For MXMACA, we compile .cu files as C++ code (using g++ directly)
-- because MXMACA provides a CUDA-compatible runtime via mc_runtime.h
--
-- Strategy: Pre-copy .cu files to a build directory with .cpp extension
-- This avoids xmake's CUDA rule detection completely

if has_config("mx-gpu") then
    local maca_path = os.getenv("MXMACA_PATH") or "/opt/maca-3.2.1"
    local build_dir = ".mx_build"
    
    -- Setup instructions:
    -- Before running xmake configure, run these commands to prepare .mx_build/:
    --   rm -rf .mx_build
    --   # Copy entire src/ directory structure
    --   cp -r src .mx_build/src
    --   # Rename .cu files to .cpp
    --   find .mx_build/src -name "*.cu" -exec sh -c 'mv "$0" "${0%.cu}.cpp"' {} \;
    
    target("llaisys-device-mx")
        set_kind("static")
        set_languages("cxx17")
        set_warnings("all", "error")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_files(build_dir .. "/src/device/nvidia/*.cpp")
        -- .mx_build/src/ is needed so that relative includes like
        -- "../../../core/llaisys_core.hpp" resolve to .mx_build/src/core/llaisys_core.hpp
        add_includedirs(build_dir .. "/src", {public = true})
        add_includedirs(maca_path .. "/include")
        add_includedirs(maca_path .. "/include/mcblas")
        -- Disable strict aliasing check because MXMACA library has strict-aliasing violations
        add_cxxflags("-O3", "-fPIC", "-fno-strict-aliasing", {force = true})
    target_end()

    target("llaisys-ops-mx")
        set_kind("static")
        set_languages("cxx17")
        set_warnings("all", "error")
        add_deps("llaisys-device-mx")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_files(build_dir .. "/src/ops/*/nvidia/*.cpp")
        -- .mx_build/src/ is needed so that relative includes resolve correctly
        add_includedirs(build_dir .. "/src", {public = true})
        add_includedirs(maca_path .. "/include")
        add_includedirs(maca_path .. "/include/mcblas")
        -- Disable strict aliasing check because MXMACA library has strict-aliasing violations
        add_cxxflags("-O3", "-fPIC", "-fno-strict-aliasing", {force = true})
        add_links("mcblas")
        add_linkdirs(maca_path .. "/lib")
    target_end()
end

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")
    if has_config("mx-gpu") then
        add_deps("llaisys-device-mx")
        add_deps("llaisys-ops-mx")
    end

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/llaisys/*.cc")

    if has_config("mx-gpu") then
        -- MXMACA (MetaX C500) build configuration for main target
        local maca_path = os.getenv("MXMACA_PATH") or "/opt/maca-3.2.1"
        add_includedirs(maca_path .. "/include")
        add_includedirs(maca_path .. "/include/mcblas")
        add_links("mcblas")
        add_linkdirs(maca_path .. "/lib")
        add_rpathdirs(maca_path .. "/lib")
    elseif has_config("nv-gpu") then
        local cuda_home = os.getenv("CUDA_HOME") or "/usr/local/cuda"
        -- Compile CUDA .cu files directly in the shared library so xmake
        -- performs the device-link step (resolves __cudaRegisterLinkedBinary_*).
        add_files("src/device/nvidia/*.cu", "src/ops/*/nvidia/*.cu")
        add_includedirs("src", "include")
        add_includedirs(cuda_home .. "/include")
        add_cugencodes("sm_120")
        add_cuflags("--expt-relaxed-constexpr", "-O3", "-Xcompiler -fPIC", {force = true})
        add_links("cudart", "cublas", "cublasLt")
        add_linkdirs(cuda_home .. "/lib64")
        add_rpathdirs(cuda_home .. "/lib64")
    end

    set_installdir(".")

    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
