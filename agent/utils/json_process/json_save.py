import json
import os
import random
import string


def save_json_to_file(data, folder="./json"):
    # 确保文件夹存在
    os.makedirs(folder, exist_ok=True)

    # 生成8位随机文件名（字母+数字）
    filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    filepath = os.path.join(folder, f"{filename}.json")

    # 保存JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath