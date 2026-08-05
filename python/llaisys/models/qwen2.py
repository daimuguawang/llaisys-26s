from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor
from pathlib import Path
import safetensors
import numpy as np
import ctypes
import json
import os

def _dtype_from_numpy(np_dtype):
    if np_dtype == np.float32:
        return DataType.F32
    elif np_dtype == np.float16:
        return DataType.F16
    elif np_dtype == np.bfloat16:
        return DataType.BF16
    elif np_dtype == np.int64:
        return DataType.I64
    elif np_dtype == np.int32:
        return DataType.I32
    else:
        raise ValueError(f"Unsupported dtype: {np_dtype}")

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        # Determine dtype from first safetensors file
        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        # Detect dtype
        with safetensors.safe_open(st_files[0], framework="numpy", device="cpu") as f:
            first_key = list(f.keys())[0]
            first_tensor = f.get_tensor(first_key)
            dtype = _dtype_from_numpy(first_tensor.dtype)
        # Build meta
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        intermediate_size = config["intermediate_size"]
        vocab_size = config["vocab_size"]
        max_seq = config.get("max_position_embeddings", 32768)
        eps = config.get("rms_norm_eps", 1e-6)
        theta = config.get("rope_theta", 10000.0)
        # Get end token (eos_token_id)
        end_token = config.get("eos_token_id", 2)
        if isinstance(end_token, list):
            end_token = end_token[0]
        meta = LlaisysQwen2Meta()
        meta.dtype = int(dtype)
        meta.nlayer = num_layers
        meta.hs = hidden_size
        meta.nh = num_heads
        meta.nkvh = num_kv_heads
        meta.dh = head_dim
        meta.di = intermediate_size
        meta.maxseq = max_seq
        meta.voc = vocab_size
        meta.epsilon = eps
        meta.theta = theta
        meta.end_token = int(end_token)
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            int(device),
            device_ids,
            1,
        )
        self._device = device
        self._meta = meta
        # Get weights struct
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights = weights_ptr.contents
        # Load all weights
        self._load_weights(st_files, dtype, device)

    def _load_weights(self, st_files, dtype, device):
        # Map from safetensors key to (field_name, layer_idx)
        weight_map = {}
        # Embeddings
        weight_map["model.embed_tokens.weight"] = ("in_embed", None)
        weight_map["lm_head.weight"] = ("out_embed", None)
        weight_map["model.norm.weight"] = ("out_norm_w", None)
        # Per layer
        for i in range(self._meta.nlayer):
            prefix = f"model.layers.{i}."
            weight_map[prefix + "input_layernorm.weight"] = ("attn_norm_w", i)
            weight_map[prefix + "self_attn.q_proj.weight"] = ("attn_q_w", i)
            weight_map[prefix + "self_attn.q_proj.bias"] = ("attn_q_b", i)
            weight_map[prefix + "self_attn.k_proj.weight"] = ("attn_k_w", i)
            weight_map[prefix + "self_attn.k_proj.bias"] = ("attn_k_b", i)
            weight_map[prefix + "self_attn.v_proj.weight"] = ("attn_v_w", i)
            weight_map[prefix + "self_attn.v_proj.bias"] = ("attn_v_b", i)
            weight_map[prefix + "self_attn.o_proj.weight"] = ("attn_o_w", i)
            weight_map[prefix + "post_attention_layernorm.weight"] = ("mlp_norm_w", i)
            weight_map[prefix + "mlp.gate_proj.weight"] = ("mlp_gate_w", i)
            weight_map[prefix + "mlp.up_proj.weight"] = ("mlp_up_w", i)
            weight_map[prefix + "mlp.down_proj.weight"] = ("mlp_down_w", i)
        # Process all files
        for file in st_files:
            with safetensors.safe_open(file, framework="numpy", device="cpu") as f:
                for name in f.keys():
                    if name not in weight_map:
                        # Skip unknown weights
                        continue
                    field_name, layer_idx = weight_map[name]
                    data = f.get_tensor(name)
                    # Create tensor and load data
                    t = Tensor(shape=tuple(data.shape), dtype=dtype, device=device)
                    # Make sure data is contiguous
                    data = np.ascontiguousarray(data)
                    ptr = data.ctypes.data_as(ctypes.c_void_p)
                    t.load(ptr)
                    # Assign to weights struct
                    if layer_idx is None:
                        # Global weight
                        setattr(self._weights, field_name, t.lib_tensor())
                    else:
                        # Array element
                        arr = getattr(self._weights, field_name)
                        arr[layer_idx] = t.lib_tensor()
                    # Keep a reference to avoid GC
                    if not hasattr(self, "_weight_tensors"):
                        self._weight_tensors = []
                    self._weight_tensors.append(t)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # For test mode (top_k=1, temperature=1.0), use greedy argmax
        output_ids = list(inputs)
        max_steps = max_new_tokens if max_new_tokens else 128
        # Prefill: process all input tokens at once
        arr = (ctypes.c_int64 * len(inputs))(*inputs)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            arr,
            len(inputs),
        )
        next_token = int(next_token)
        output_ids.append(next_token)
        if next_token == self._meta.end_token:
            return output_ids
        # Decode: generate one token at a time
        for step in range(max_steps - 1):
            arr = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                arr,
                1,
            )
            next_token = int(next_token)
            output_ids.append(next_token)
            if next_token == self._meta.end_token:
                break
        return output_ids

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
