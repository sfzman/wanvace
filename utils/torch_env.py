import os


_EXPANDABLE_SEGMENTS_CONF = "expandable_segments:True"


def configure_torch_cuda_allocator_env():
    """为 PyTorch CUDA allocator 启用更稳妥的碎片化配置。"""
    current_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if not current_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _EXPANDABLE_SEGMENTS_CONF
        return os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    current_parts = [part.strip() for part in current_conf.split(",") if part.strip()]
    normalized_parts = [part.lower() for part in current_parts]
    if _EXPANDABLE_SEGMENTS_CONF.lower() not in normalized_parts:
        current_parts.append(_EXPANDABLE_SEGMENTS_CONF)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(current_parts)
    return os.environ["PYTORCH_CUDA_ALLOC_CONF"]
