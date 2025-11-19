# deepseek_proxy_server.py (支持流式版本)
"""
DeepSeek V3.2-Exp Tool Calling 代理服务器 (支持流式)
"""

import json
import os
import re
import httpx
from typing import Any, Dict, List, Optional, AsyncIterator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="DeepSeek Tool Calling Proxy")

# 远程 DeepSeek API 配置
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", 
    "http://10.248.60.236:5000/v1/chat/completions"
)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_OPENAI_API_KEY")

# 启动时验证配置
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "❌ 环境变量 DEEPSEEK_OPENAI_API_KEY 未设置！\n"
        "请运行: export DEEPSEEK_OPENAI_API_KEY='your-api-key'"
    )


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

CRITICAL WORKFLOW - FOLLOW EXACTLY:

**FIRST TIME USER ASKS:**
1. Output ONLY the XML function call:
<function_call>
{{"name": "function_name", "arguments": {{"param": "value"}}}}
</function_call>

**AFTER RECEIVING FUNCTION RESULT:**
2. Respond in natural language with the result
3. DO NOT output any XML
4. DO NOT call the function again
5. STOP after providing the result

EXAMPLE - CORRECT BEHAVIOR:
User: "call myecho with text HELLO"

Your 1st response:
<function_call>
{{"name": "myecho", "arguments": {{"text": "HELLO"}}}}
</function_call>

System: Function myecho returned: <<ECHO: HELLO>>

Your 2nd response:
The function has been called and returned: <<ECHO: HELLO>>
[STOP HERE - Do not call myecho again]

REMEMBER: Call function ONCE, then provide natural language response."""


def extract_xml_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """从响应中提取 XML 格式的工具调用"""
    if not content or "<function_call>" not in content:
        return None
    
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
    """过滤和转换消息，使其适合 DeepSeek 模型"""
    filtered = []
    
    for msg in messages:
        role = msg.get("role")
        
        if role == "system":
            filtered.append(msg)
        
        elif role == "user":
            filtered.append(msg)
        
        elif role == "assistant":
            content = msg.get("content", "")
            
            if msg.get("tool_calls") and not content:
                print(f"   ⚠️  跳过空的 assistant 工具调用消息")
                continue
            
            filtered.append({
                "role": "assistant",
                "content": content
            })
        
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            tool_content = msg.get("content", "")
            
            filtered.append({
                "role": "user",
                "content": f"Function {tool_name} returned: {tool_content}"
            })
            print(f"   🔄 转换 tool 消息为 user 消息: {tool_name}")
    
    return filtered


def should_use_streaming(messages: List[Dict[str, Any]], tools: Optional[List]) -> bool:
    """
    判断是否应该使用流式响应
    
    规则：
    1. 如果没有工具定义 → 可以流式
    2. 如果有工具，但历史消息中已有 tool role → 不需要流式（这是第二轮回复）
    3. 如果有工具，且是首次请求 → 不能流式（需要解析 XML）
    """
    # 没有工具定义，直接流式
    if not tools:
        return True
    
    # 检查消息中是否有 tool role
    has_tool_result = any(msg.get("role") == "tool" for msg in messages)
    
    # 如果有工具结果，说明这是第二轮回复，可以流式
    if has_tool_result:
        print(f"   ℹ️  检测到工具结果，第二轮回复可以使用流式")
        return True
    
    # 否则，这是首次工具调用请求，必须非流式
    print(f"   ℹ️  首次工具调用请求，使用非流式")
    return False


async def stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any]
) -> AsyncIterator[str]:
    """流式转发响应"""
    async with client.stream("POST", url, json=body, headers=headers, timeout=60.0) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """处理聊天补全请求（支持流式和非流式）"""
    try:
        body = await request.json()
        
        messages = body.get("messages", [])
        tools = body.get("tools")
        is_stream = body.get("stream", False)  # ⭐ 获取客户端的流式请求
        
        print(f"\n{'='*60}")
        print(f"📨 收到请求:{messages}")
        print(f"   消息数: {len(messages)}")
        print(f"   工具数: {len(tools) if tools else 0}")
        print(f"   客户端请求流式: {is_stream}")
        
        msg_types = [msg.get("role") for msg in messages]
        print(f"   消息类型: {msg_types}")
        
        # 过滤消息
        filtered_messages = filter_messages_for_deepseek(messages)
        
        # 如果有工具，修改系统提示词
        if tools:
            tool_prompt = build_tool_prompt(tools)
            
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
        
        # ⭐ 智能判断是否使用流式
        use_streaming = is_stream and should_use_streaming(messages, tools)
        
        # 构造请求
        deepseek_body = {
            "model": body.get("model", "deepseek-chat"),
            "messages": filtered_messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 2000),
            "stream": use_streaming  # ⭐ 根据判断决定是否流式
        }
        
        print(f"   🔄 调用远程 API: {DEEPSEEK_API_URL}")
        print(f"   📡 使用{'流式' if use_streaming else '非流式'}传输")
        
        # ⭐ 流式响应
        if use_streaming:
            async def generate():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async for chunk in stream_response(
                        client,
                        DEEPSEEK_API_URL,
                        {
                            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        deepseek_body
                    ):
                        yield chunk
            
            print(f"   ✅ 返回流式响应")
            print(f"{'='*60}\n")
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # ⭐ 非流式响应（工具调用）
        else:
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
            
            # 验证响应
            if "choices" not in result or not result["choices"]:
                raise HTTPException(
                    status_code=502,
                    detail="Invalid response from DeepSeek API"
                )
            
            # 提取响应
            choice = result["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            print(f"   📥 收到响应 ({len(content)} 字符)")
            
            # 解析工具调用
            tool_calls = None
            finish_reason = choice.get("finish_reason", "stop")
            
            if tools and content:
                tool_calls = extract_xml_tool_calls(content)
                
                if tool_calls:
                    print(f"   ✅ 提取到 {len(tool_calls)} 个工具调用:")
                    for tc in tool_calls:
                        print(f"      - {tc['function']['name']}({tc['function']['arguments']})")
                    
                    message["tool_calls"] = tool_calls
                    message["content"] = ""
                    finish_reason = "tool_calls"
            
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
        return {
            "status": "healthy",
            "deepseek_api": "reachable",
            "streaming_support": True
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    """根路径信息"""
    return {
        "name": "DeepSeek Tool Calling Proxy",
        "version": "2.0.0",
        "features": [
            "Tool calling via XML parsing",
            "Streaming support for non-tool requests",
            "Intelligent stream/non-stream switching"
        ],
        "config": {
            "api_url": DEEPSEEK_API_URL,
            "api_key_set": bool(DEEPSEEK_API_KEY)
        }
    }


if __name__ == "__main__":
    print("="*60)
    print("🚀 DeepSeek Tool Calling 代理服务器 v2.0")
    print("="*60)
    print(f"📅 当前时间: 2025-11-19 02:06:03 UTC")
    print(f"👤 当前用户: greatabel")
    print(f"🔗 监听地址: http://localhost:8000")
    print(f"🌐 远程 API: {DEEPSEEK_API_URL}")
    print(f"🔑 API Key: {'✅ 已设置' if DEEPSEEK_API_KEY else '❌ 未设置'}")
    print(f"📡 流式支持: ✅ 已启用")
    print(f"💊 健康检查: http://localhost:8000/health")
    print("="*60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)