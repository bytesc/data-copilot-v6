from typing import Dict, Any, Union
import json
from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

import pandas as pd
from opensearchpy import OpenSearch, helpers
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import uuid
import numpy as np
import time
import re

# 禁用SSL警告
urllib3.disable_warnings(InsecureRequestWarning)

# 1. 连接OpenSearch
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'S202512sss'),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=30
)

def search_by_dsl(dsl: Union[str, Dict[str, Any]],
                  index: str = None,
                  return_whole_response: bool = False,
                  raise_on_error: bool = True) -> Dict[str, Any]:
    """
       向 OpenSearch 发送 DSL 查询并返回 JSON 结果。
    """
    try:
        # 如果是字符串，先解析成 dict
        if isinstance(dsl, str):
            dsl = json.loads(dsl)

        # 发送查询
        resp = client.search(index=index, body=dsl)

        # 按需返回
        if return_whole_response:
            return resp
        return resp.get("hits", {})

    except json.JSONDecodeError as e:
        err_info = {"error": "DSL JSON 解析失败", "details": str(e)}
    except OpenSearchException as e:
        err_info = {"error": "OpenSearch 查询失败", "details": str(e)}
    except Exception as e:
        err_info = {"error": "未知异常", "details": str(e)}

    if raise_on_error:
        raise RuntimeError(err_info)
    return err_info


def get_index_mapping(index: str,
                      raise_on_error: bool = True) -> Dict[str, Any]:
    try:
        # 使用OpenSearch客户端的indices.get_mapping方法
        mapping = client.indices.get_mapping(index=index)
        return mapping

    except OpenSearchException as e:
        err_info = {"error": f"获取索引 {index} 的mapping失败", "details": str(e)}
        if raise_on_error:
            raise RuntimeError(err_info)
        return err_info
    except Exception as e:
        err_info = {"error": "未知异常", "details": str(e)}
        if raise_on_error:
            raise RuntimeError(err_info)
        return err_info

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # # 例1：直接写字典
    # dsl_dict = {
    #     "size": 5,
    #     "query": {"match_all": {}},
    #     "_source": False
    # }
    # print(search_by_dsl(dsl_dict, index="my_index"))

    # 例2：写 JSON 字符串
#     dsl_json = '''
# {
#   "query": {
#     "bool": {
#       "must": [
#         { "term": { "diabetic_retinopathy": 1 } },
#         { "range": { "patient_age": { "gte": 40 } } }
#       ]
#     }
#   }
# }
#     '''

    dsl_json = '''
{
  "_source": ["image_id", "patient_id", "diabetes_time_y"],
  "query": {
    "bool": {
      "must": [
        { "term": { "nationality.keyword": "Brazil" } },
        { "range": { "diabetes_time_y": { "gt": 10 } } }
      ]
    }
  },
  "sort": [{ "diabetes_time_y": { "order": "desc" } }]
}
        '''

#     dsl_json = '''
# {
#   "size": 0,
#   "aggs": {
#     "by_camera": {
#       "terms": { "field": "camera.keyword", "size": 10 },
#       "aggs": {
#         "hemorrhage_count": {
#           "filter": { "term": { "hemorrhage": 1 } }
#         }
#       }
#     }
#   }
# }
#             '''
    print(search_by_dsl(dsl_json, index="brset", return_whole_response=True))

    mapping = get_index_mapping("brset")
    print("brset索引的mapping信息：")
    print(json.dumps(mapping, indent=2, ensure_ascii=False))

