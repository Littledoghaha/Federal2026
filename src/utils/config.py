# 保存配置文件

import json
import os
from datetime import datetime


def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_result_dir(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{experiment_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_config_copy(config, result_dir):
    config_save_path = os.path.join(result_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)