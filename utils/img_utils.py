from PIL import Image

def get_image_info(image):
    """获取图片信息"""
    if not image:
        return "未上传图片"
    
    try:
        if isinstance(image, str):
            # 如果是文件路径
            img = Image.open(image)
        else:
            # 如果是PIL Image对象
            img = image
        
        width, height = img.size
        mode = img.mode
        format_name = img.format if hasattr(img, 'format') else "未知"
        
        return f"尺寸: {width}x{height}, 模式: {mode}, 格式: {format_name}"
    except Exception as e:
        return f"获取图片信息失败: {str(e)}"