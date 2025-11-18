# deepseek_proxy_server.py (修复版)
"""
DeepSeek V3.2-Exp Tool Calling 代理服务器 (修复版)
"""

import json
import re
import httpx
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="DeepSeek Tool Calling Proxy")

# 远程 DeepSeek API 配置
DEEPSEEK_API_URL = "http://10.248.60.236:5000/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-i7aFdS6UvglMyX2vWMrDccwsPeIK"


def build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """构建工具调用的系统提示词"""
    tool_descriptions = []
    
    for tool in tools:
        func = tool["function"]
        tool_descriptions.append(
            f"## {func['name']}\n"
            f"Description: {func['description']}\n"
            f"Parameters: {json.dumps(func['parameters'], indent=2)}"
        )
    
    tools_text = "\n\n".join(tool_descriptions)
    
    return f"""You are a helpful assistant with access to the following functions:

{tools_text}

CRITICAL RULES FOR CALLING FUNCTIONS:
1. When you need to call a function, output ONLY this XML format:
<function_call>
{{"name": "function_name", "arguments": {{"param": "value"}}}}
</function_call>

2. DO NOT add any text before or after the XML block
3. DO NOT explain what you're doing
4. DO NOT say "I will call" or "Let me use" - just output the XML directly
5. The JSON inside XML must be valid

EXAMPLES:
User: "What's the weather in Beijing?"
Your response:
<function_call>
{{"name": "get_weather", "arguments": {{"city": "Beijing"}}}}
</function_call>

User: "Call myecho with text HELLO"
Your response:
<function_call>
{{"name": "myecho", "arguments": {{"text": "HELLO"}}}}
</function_call>

IMPORTANT: After a function is called and you receive the result, provide a natural language response to the user based on the function result. DO NOT call the function again."""


def extract_xml_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """从响应中提取 XML 格式的工具调用"""
    if not content or "<function_call>" not in content:
        return None
    
    # 提取所有 <function_call>...</function_call> 块
    pattern = r'<function_call>\s*(\{.*?\})\s*</function_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None
    
    tool_calls = []
    for idx, match in enumerate(matches):
        try:
            call_data = json.loads(match)
            
            if "name" not in call_data or "arguments" not in call_data:
                print(f"⚠️  工具调用缺少必需字段: {call_data}")
                continue
            
            tool_calls.append({
                "id": f"call_{abs(hash(match)) % 100000}_{idx}",
                "type": "function",
                "function": {
                    "name": call_data["name"],
                    "arguments": json.dumps(call_data["arguments"])
                }
            })
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}\n   原始: {match}")
            continue
    
    return tool_calls if tool_calls else None


def filter_messages_for_deepseek(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤和转换消息，使其适合 DeepSeek 模型
    
    关键处理：
    1. 移除 tool_calls 字段（DeepSeek 不需要在历史中看到这个）
    2. 将 tool role 的消息转换为 user role（DeepSeek 更容易理解）
    """
    filtered = []
    
    for msg in messages:
        role = msg.get("role")
        
        if role == "system":
            # 保留系统消息（会被我们的工具提示词覆盖）
            filtered.append(msg)
        
        elif role == "user":
            # 保留用户消息
            filtered.append(msg)
        
        elif role == "assistant":
            # 移除 tool_calls，只保留文本内容
            content = msg.get("content", "")
            
            # 如果 assistant 消息有 tool_calls 但没有 content
            # 说明这是一个纯工具调用消息，我们跳过它
            # （因为 DeepSeek 会重新生成工具调用）
            if msg.get("tool_calls") and not content:
                print(f"   ⚠️  跳过空的 assistant 工具调用消息")
                continue
            
            filtered.append({
                "role": "assistant",
                "content": content
            })
        
        elif role == "tool":
            # 将 tool 消息转换为 user 消息
            # 格式：Function <name> returned: <content>
            tool_name = msg.get("name", "unknown")
            tool_content = msg.get("content", "")
            
            filtered.append({
                "role": "user",
                "content": f"Function {tool_name} returned: {tool_content}"
            })
            print(f"   🔄 转换 tool 消息为 user 消息: {tool_name}")
    
    return filtered


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """处理聊天补全请求"""
    try:
        # 解析请求体
        body = await request.json()
        
        messages = body.get("messages", [])
        tools = body.get("tools")
        
        print(f"\n{'='*60}")
        print(f"📨 收到请求:")
        print(f"   消息数: {len(messages)}")
        print(f"   工具数: {len(tools) if tools else 0}")
        
        # 调试：打印消息类型
        msg_types = [msg.get("role") for msg in messages]
        print(f"   消息类型: {msg_types}")
        
        # 过滤和转换消息
        filtered_messages = filter_messages_for_deepseek(messages)
        
        # 如果有工具，修改系统提示词
        if tools:
            tool_prompt = build_tool_prompt(tools)
            
            # 查找或添加系统消息
            system_found = False
            for i, msg in enumerate(filtered_messages):
                if msg.get("role") == "system":
                    filtered_messages[i] = {
                        "role": "system",
                        "content": tool_prompt
                    }
                    system_found = True
                    break
            
            if not system_found:
                filtered_messages.insert(0, {
                    "role": "system",
                    "content": tool_prompt
                })
            
            print(f"   ✅ 已注入工具调用提示词")
        
        # 构造发送给 DeepSeek 的请求（不传 tools）
        deepseek_body = {
            "model": body.get("model", "deepseek-chat"),
            "messages": filtered_messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 2000),
            "stream": False
        }
        
        # 调用远程 DeepSeek API
        print(f"   🔄 调用远程 API: {DEEPSEEK_API_URL}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                DEEPSEEK_API_URL,
                json=deepseek_body,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            response.raise_for_status()
            result = response.json()
        
        # ⭐ 验证响应格式
        if "choices" not in result:
            print(f"   ❌ 响应缺少 'choices' 字段")
            print(f"   完整响应: {json.dumps(result, indent=2)}")
            raise HTTPException(
                status_code=502,
                detail=f"Invalid response from DeepSeek API: missing 'choices' field"
            )
        
        if not result["choices"]:
            print(f"   ❌ 'choices' 数组为空")
            raise HTTPException(
                status_code=502,
                detail=f"Invalid response from DeepSeek API: empty 'choices' array"
            )
        
        # 提取响应内容
        choice = result["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        print(f"   📥 收到响应 ({len(content)} 字符)")
        
        # 解析 XML 工具调用
        tool_calls = None
        finish_reason = choice.get("finish_reason", "stop")
        
        if tools and content:
            tool_calls = extract_xml_tool_calls(content)
            
            if tool_calls:
                print(f"   ✅ 提取到 {len(tool_calls)} 个工具调用:")
                for tc in tool_calls:
                    print(f"      - {tc['function']['name']}({tc['function']['arguments']})")
                
                # 修改响应
                message["tool_calls"] = tool_calls
                message["content"] = ""  # 清空内容
                finish_reason = "tool_calls"
            else:
                print(f"   ⚠️  未检测到工具调用")
                if "<function_call>" in content:
                    print(f"   原始内容: {content[:200]}...")
        
        # 构造返回响应
        choice["finish_reason"] = finish_reason
        choice["message"] = message
        
        print(f"   ✅ 返回结果 (finish_reason: {finish_reason})")
        print(f"{'='*60}\n")
        
        return JSONResponse(content=result)
        
    except httpx.HTTPError as e:
        print(f"❌ HTTP 错误: {e}")
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(e)}")
    
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek"
            }
        ]
    }


@app.get("/health")
async def health():
    """健康检查"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://10.248.60.236:5000/v1/models")
            response.raise_for_status()
        return {"status": "healthy", "deepseek_api": "reachable"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    print("="*60)
    print("🚀 启动 DeepSeek Tool Calling 代理服务器 (修复版)")
    print("="*60)
    print(f"监听地址: http://localhost:8000")
    print(f"远程 API: {DEEPSEEK_API_URL}")
    print(f"健康检查: http://localhost:8000/health")
    print("="*60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)