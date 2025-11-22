# describe_image_qwen_vl.py
"""
使用本地 Qwen-VL-32B 多模态模型描述图片

配置：
- 模型: qwen-vl-32b
- 地址: http://10.248.60.233:5001/v1/chat/completions
- API Key: 从环境变量 QWEN_VL_API_KEY 读取（或使用默认值 "test"）

功能：
- 支持本地图片文件
- 支持 base64 编码
- 兼容 OpenAI Vision API 格式

前提：
1. Qwen-VL-32B 服务运行在 http://10.248.60.233:5001
2. 设置环境变量（可选）: export QWEN_VL_API_KEY=test
"""

import base64
import os
import requests
from datetime import datetime, timedelta


def get_beijing_time(format="%Y-%m-%d %H:%M:%S"):
    """获取北京时间（东八区）"""
    return (datetime.utcnow() + timedelta(hours=8)).strftime(format)


def describe_image(image_path="animals.png", custom_prompt=None) -> str:
    """
    使用本地 Qwen-VL-32B 模型描述图片内容
    
    Args:
        image_path: str, 图片文件路径（支持相对路径和绝对路径）
        custom_prompt: str, 自定义提示词（默认："这张图片里有什么？"）
    
    Returns:
        str: 模型返回的图片描述
    
    Raises:
        FileNotFoundError: 图片文件不存在
        requests.exceptions.RequestException: API 调用失败
    """
    
    print("="*60)
    print(f"🖼️  Qwen-VL 图片描述服务")
    print("="*60)
    print(f"📅 时间: {get_beijing_time()} 北京时间")
    print(f"📁 图片: {image_path}")
    print()
    
    # ============================================================
    # 1. 检查图片文件是否存在
    # ============================================================
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    print(f"✅ 图片文件存在: {os.path.abspath(image_path)}")
    print(f"   文件大小: {os.path.getsize(image_path) / 1024:.2f} KB\n")
    
    # ============================================================
    # 2. 配置 API（从环境变量读取）
    # ============================================================
    # API Key（优先从环境变量读取，否则使用默认值）
    api_key = os.getenv("QWEN_VL_API_KEY", "test")
    
    # API 端点
    api_url = os.getenv(
        "QWEN_VL_API_URL", 
        "http://10.248.60.233:5001/v1/chat/completions"
    )
    
    # 模型名称
    model_name = os.getenv("QWEN_VL_MODEL", "qwen-vl-32b")
    
    print(f"🔧 API 配置:")
    print(f"   地址: {api_url}")
    print(f"   模型: {model_name}")
    print(f"   API Key: {'*' * (len(api_key) - 4)}{api_key[-4:] if len(api_key) > 4 else api_key}")
    print()
    
    # ============================================================
    # 3. 将图片编码为 base64
    # ============================================================
    def encode_image(image_path):
        """将图片文件编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    print("🔄 正在编码图片为 base64...")
    base64_image = encode_image(image_path)
    print(f"✅ 编码完成: {len(base64_image)} 字符\n")
    
    # ============================================================
    # 4. 构建请求
    # ============================================================
    # 默认提示词（中文）
    if custom_prompt is None:
        custom_prompt = "请详细描述这张图片的内容，包括主要对象、场景、颜色等细节。"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 请求体（兼容 OpenAI Vision API 格式）
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": custom_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500,      # 增加 token 限制（中文描述更长）
        "temperature": 0.7,     # 适中的创意度
    }
    
    print(f"📤 发送请求:")
    print(f"   提示词: {custom_prompt[:50]}...")
    print(f"   最大 Tokens: {payload['max_tokens']}")
    print()
    
    # ============================================================
    # 5. 调用 API
    # ============================================================
    try:
        print(f"⏳ 正在调用 Qwen-VL API...")
        start_time = datetime.utcnow()
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60  # 60 秒超时
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # 检查响应状态
        response.raise_for_status()
        
        print(f"✅ API 调用成功")
        print(f"   耗时: {duration:.2f} 秒")
        print(f"   状态码: {response.status_code}\n")
        
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 60 秒）")
        print(f"💡 提示: 检查网络连接或增加 timeout 值\n")
        raise
    
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print(f"💡 提示: 检查 API 服务是否运行在 {api_url}\n")
        raise
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP 错误: {e}")
        print(f"   响应内容: {response.text[:200]}\n")
        raise
    
    # ============================================================
    # 6. 解析响应
    # ============================================================
    try:
        response_json = response.json()
        
        # 提取描述内容
        description = response_json["choices"][0]["message"]["content"]
        
        # 显示完整响应（调试用）
        print(f"📊 响应详情:")
        if "usage" in response_json:
            usage = response_json["usage"]
            print(f"   输入 Tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   输出 Tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   总计 Tokens: {usage.get('total_tokens', 'N/A')}")
        print()
        
        print("="*60)
        print("📝 图片描述结果")
        print("="*60)
        print(description)
        print("="*60)
        print()
        
        return description
    
    except (KeyError, IndexError) as e:
        print(f"❌ 响应格式错误: {e}")
        print(f"   完整响应: {response.text}\n")
        raise


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 示例 1: 使用默认图片和提示词
    try:
        description = describe_image("animals.png")
        print(f"✅ 描述完成！\n")
    
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print(f"💡 提示: 请确保图片文件存在，或修改 image_path 参数\n")
    
    except Exception as e:
        print(f"❌ 发生错误: {e}\n")
    