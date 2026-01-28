import json
import re


def string_to_json(input_string):
    if not input_string or not input_string.strip():
        raise ValueError("输入字符串不能为空")

    # 清理输入字符串
    text = input_string.strip()

    # 检查是否是Markdown代码块格式
    code_block_pattern = r'^```(?:json)?\s*(.*?)\s*```$'
    match = re.search(code_block_pattern, text, re.DOTALL)

    if match:
        # 提取代码块内容
        json_content = match.group(1).strip()
    else:
        # 如果不是代码块格式，直接使用原内容
        json_content = text

    # 移除JSON注释（单行注释和多行注释）
    json_content = re.sub(r'//.*?$', '', json_content, flags=re.MULTILINE)  # 移除单行注释
    json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)  # 移除多行注释

    # 清理多余的空行和空格
    json_content = re.sub(r'\n\s*\n', '\n', json_content)  # 移除空行
    json_content = json_content.strip()

    try:
        # 尝试解析为JSON
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        # 提供更详细的错误信息
        raise json.JSONDecodeError(
            f"无法解析JSON字符串: {e.msg}",
            e.doc,
            e.pos
        )


# 测试函数
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 纯JSON
        '{"name": "John", "age": 30}',

        # Markdown代码块
        '```json\n{"name": "John", "age": 30}\n```',

        # 没有语言标识的代码块
        '```\n{"name": "John", "age": 30}\n```',

        # 数组格式的JSON
        '[1, 2, 3, 4, 5]',

        # 代码块中的数组
        '```json\n[1, 2, 3, 4, 5]\n```'
    ]

    for i, test in enumerate(test_cases, 1):
        try:
            result = string_to_json(test)
            print(f"测试用例 {i}: 成功")
            print(f"结果: {result}")
            print(f"类型: {type(result)}\n")
        except Exception as e:
            print(f"测试用例 {i}: 失败 - {e}\n")