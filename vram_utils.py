import torch

def clear_vram():
    """释放显存"""
    global pipe
    try:
        # 清理pipeline
        if pipe is not None:
            del pipe
            pipe = None
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # 强制垃圾回收
        import gc
        gc.collect()
        
        return "显存释放完成！"
    except Exception as e:
        return f"显存释放失败：{str(e)}"

def get_vram_info():
    """获取显存信息"""
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    try:
        # 使用nvidia-smi命令获取更准确的显存信息
        import subprocess
        import re
        
        # 运行nvidia-smi命令
        result = subprocess.run(['nvidia-smi', '--query-gpu', 'memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # 解析输出，格式通常是: "1234, 24576" (已使用MB, 总计MB)
            output = result.stdout.strip()
            match = re.search(r'(\d+),\s*(\d+)', output)
            if match:
                used_mb = int(match.group(1))
                total_mb = int(match.group(2))
                used_gb = used_mb / 1024
                total_gb = total_mb / 1024
                free_gb = total_gb - used_gb
                
                return f"显存使用: {used_gb:.2f}GB / {total_gb:.2f}GB (已使用/总计) | 剩余: {free_gb:.2f}GB"
            else:
                # 如果解析失败，回退到PyTorch方法
                raise ValueError("无法解析nvidia-smi输出")
        else:
            # 如果nvidia-smi失败，回退到PyTorch方法
            raise RuntimeError(f"nvidia-smi命令失败: {result.stderr}")
            
    except Exception as e:
        # 回退到PyTorch方法作为备选方案
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return f"显存使用: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB (已分配/已保留/总计) [PyTorch备选]"
        except Exception as fallback_e:
            return f"获取显存信息失败: {str(e)} | 备选方案也失败: {str(fallback_e)}"
