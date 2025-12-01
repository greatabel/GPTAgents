from promptflow.core import tool
from typing import List, Dict, Any
import json


@tool
def grounding(inputs: Any) -> List[Dict]:
    """
    inputs:
      - 期望是一个 list[dict]，例如：
        [
          {"title": "Edge of Tomorrow", "subject": "5", "format": "5", "genre": "5"},
          ...
        ]
      - 但在实际运行中，有可能是 JSON 字符串，所以这里做一层兼容。
    """

    # 如果传进来的是字符串（通常是 JSON），先反序列化
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except Exception:
            # 如果解析失败，直接当成空列表处理
            inputs = []

    output: List[Dict] = []

    for data_dict in inputs:
        # 有时候里面的元素也可能是字符串，这里也兼容一下
        if isinstance(data_dict, str):
            try:
                data_dict = json.loads(data_dict)
            except Exception:
                # 实在不行就跳过这一条
                continue

        # 确保是 dict 才处理
        if not isinstance(data_dict, dict):
            continue

        total_score = 0.0
        score_count = 0

        for key, value in list(data_dict.items()):
            if key.lower() == "title":
                # title 不参与评分
                continue

            try:
                score = float(value)
            except (TypeError, ValueError):
                # 非数字，直接跳过
                continue

            total_score += score
            score_count += 1

            # 顺便把原来的字符串 "5" 也变成 5.0
            data_dict[key] = score

        avg_score = total_score / score_count if score_count > 0 else 0.0
        data_dict["avg_score"] = round(avg_score, 2)

        output.append(data_dict)

    return output
