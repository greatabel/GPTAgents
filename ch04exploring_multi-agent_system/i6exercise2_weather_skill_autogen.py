# weather_skill_autogen.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import requests


# ========= 1. 读取本地 LLM / DeepSeek 配置 =========
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")


# ========= 2. 定义一个“技能函数”：查天气 =========
def get_weather(city: str) -> str:
    """
    使用 Open-Meteo 的 Geocoding + Weather API 查询某个城市的当前天气。
    返回一段适合直接展示给用户的中文描述。
    """
    if not city:
        return "没有提供城市名称。"

    # 1）城市名 -> 经纬度（Geocoding API）
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json",
        },
        timeout=10,
    )
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"没有找到与「{city}」匹配的城市，请尝试更精确的名称（例如：上海市、Beijing）。"

    top = geo_data["results"][0]
    lat = top["latitude"]
    lon = top["longitude"]
    name = top.get("name", city)
    country = top.get("country", "")

    # 2）经纬度 -> 当前天气（Forecast API）
    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    )
    weather = weather_resp.json().get("current", {})

    temp = weather.get("temperature_2m")
    wind = weather.get("wind_speed_10m")
    code = weather.get("weather_code")

    # 简单把 weather_code 映射成一个大致中文描述（只做了少量常见值）
    code_map = {
        0: "晴朗",
        1: "大致晴朗",
        2: "部分多云",
        3: "阴",
        45: "有雾",
        48: "雾且结霜",
        51: "小毛毛雨",
        53: "中毛毛雨",
        55: "大毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        95: "雷阵雨",
    }
    weather_text = code_map.get(code, f"天气代码 {code}")

    return (
        f"城市：{name}（{country}）\n"
        f"当前气温：{temp} ℃\n"
        f"天气情况：{weather_text}\n"
        f"风速约：{wind} m/s\n"
    )


# ========= 3. 创建带“技能”的 WeatherBot 助手 =========
weather_bot = AssistantAgent(
    name="WeatherBot",
    llm_config={
        "config_list": config_list,
        "temperature": 0.1,
    },
    system_message="""
你是一个天气助手，可以通过调用工具函数 get_weather(city: str) 来获取实时天气。
使用说明：
1. 当用户想查天气，但没有说明城市时，要先主动询问：“你想查询哪个城市的天气？”
2. 当用户提供城市后，调用 get_weather 工具获取数据，并用简洁的中文输出结果。
3. 如果工具返回了错误信息（例如找不到城市），要把错误原因转述给用户，并提示用户重新输入城市名。
    """,
    # 关键点：把 Python 函数挂到 agent 上，作为“技能”
    function_map={
        "get_weather": get_weather,
    },
)


# ========= 4. UserProxyAgent：负责执行 tool / 代码 =========
user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
    },
    human_input_mode="ALWAYS",  # 为了让它可以问你城市名并等待你输入
)


if __name__ == "__main__":
    # 启动一个简单 Demo：你可以直接跟 WeatherBot 聊天
    task = "我想查一下天气，可以帮我吗？"  # 故意不写城市，让 agent 主动问你

    user_proxy.initiate_chat(
        recipient=weather_bot,
        message=task,
        max_turns=10,      # 允许多轮对话（先问城市，再查天气，再回复）
        summary_method="last_msg",
    )
