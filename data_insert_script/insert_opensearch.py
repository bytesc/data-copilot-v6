import pandas as pd
from opensearchpy import OpenSearch, helpers
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import uuid
import numpy as np
import time
import re
import json
from typing import Dict, List, Tuple, Optional, Any

# 禁用SSL警告
urllib3.disable_warnings(InsecureRequestWarning)


class OpenSearchDataImporter:
    """
    OpenSearch数据导入器
    功能：将CSV数据自动导入到OpenSearch索引中
    """

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 9200,
                 username: str = 'admin',
                 password: str = 'S202512sss',
                 use_ssl: bool = True,
                 verify_certs: bool = False,
                 timeout: int = 30):
        """
        初始化OpenSearch连接

        Args:
            host: OpenSearch主机地址
            port: OpenSearch端口
            username: 用户名
            password: 密码
            use_ssl: 是否使用SSL
            verify_certs: 是否验证证书
            timeout: 连接超时时间
        """
        self.host = host
        self.port = port
        self.client = None
        self.connected = False

        # 初始化连接
        self._init_client(username, password, use_ssl, verify_certs, timeout)

    def _init_client(self, username, password, use_ssl, verify_certs, timeout):
        """初始化OpenSearch客户端"""
        try:
            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_auth=(username, password),
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_show_warn=False,
                timeout=timeout
            )
            # 测试连接
            if self.client.ping():
                self.connected = True
                print(f"✓ 成功连接到 OpenSearch ({self.host}:{self.port})")
            else:
                print(f"✗ 无法连接到 OpenSearch ({self.host}:{self.port})")
        except Exception as e:
            print(f"✗ 连接OpenSearch失败: {e}")
            self.connected = False

    def _infer_field_types_from_mapping(self, mapping: Dict) -> Tuple[List, List, List]:
        """
        从mapping定义中推断字段类型

        Args:
            mapping: OpenSearch mapping配置

        Returns:
            (integer_fields, float_fields, keyword_fields)
        """
        integer_fields = []
        float_fields = []
        keyword_fields = []

        # 兼容不同的mapping结构
        properties = mapping.get("mappings", {})
        if isinstance(properties, dict) and "properties" in properties:
            properties = properties.get("properties", {})
        elif isinstance(properties, dict):
            properties = properties.get("properties", {})
        else:
            properties = {}

        for field_name, field_config in properties.items():
            field_type = field_config.get("type", "")

            if field_type == "integer":
                integer_fields.append(field_name)
            elif field_type == "float":
                float_fields.append(field_name)
            elif field_type == "keyword":
                keyword_fields.append(field_name)
            elif field_type == "text":
                keyword_fields.append(field_name)  # 文本字段也按字符串处理
            # 可以继续添加其他类型的处理

        return integer_fields, float_fields, keyword_fields

    def _clean_data(self, df: pd.DataFrame, integer_fields: List, float_fields: List) -> pd.DataFrame:
        """
        根据字段类型进行数据清洗

        Args:
            df: 原始数据框
            integer_fields: 整数字段列表
            float_fields: 浮点数字段列表

        Returns:
            清洗后的数据框
        """
        print("开始数据清洗...")
        print(f"原始数据形状: {df.shape}")

        # 创建副本避免修改原数据
        df_cleaned = df.copy()

        # 1. 替换NaN为None
        df_cleaned = df_cleaned.where(pd.notnull(df_cleaned), None)
        print("✓ 替换NaN为None完成")

        # 2. 清理整数字段
        print("处理整数字段...")
        for col in integer_fields:
            if col in df_cleaned.columns:
                before_count = df_cleaned[col].notnull().sum()
                try:
                    # 先尝试转换为数值类型
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    # 转换为可为空的整数类型
                    df_cleaned[col] = df_cleaned[col].astype('Int64')
                    df_cleaned[col] = df_cleaned[col].where(pd.notnull(df_cleaned[col]), None)
                    after_count = df_cleaned[col].notnull().sum()
                    print(f"  ✓ {col}: 整数字段处理完成，非空值 {before_count} -> {after_count}")
                except Exception as e:
                    print(f"  ✗ {col}: 整数字段处理失败 - {e}")

        # 3. 清理浮点数字段
        print("处理浮点数字段...")
        for col in float_fields:
            if col in df_cleaned.columns:
                before_count = df_cleaned[col].notnull().sum()
                try:
                    # 统一处理各种格式的浮点数
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '.', regex=False)
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

                    # 替换特殊浮点数值
                    df_cleaned[col] = df_cleaned[col].apply(
                        lambda x: None if pd.isna(x) or x in [float('inf'), float('-inf'), float('nan')] else float(x)
                    )

                    after_count = df_cleaned[col].notnull().sum()
                    print(f"  ✓ {col}: 浮点数字段处理完成，非空值 {before_count} -> {after_count}")
                except Exception as e:
                    print(f"  ✗ {col}: 浮点数字段处理失败 - {e}")

        # 4. 清理字符串字段
        print("清理字符串字段...")
        # 获取所有非数值字段
        all_fields = set(df_cleaned.columns)
        numeric_fields = set(integer_fields + float_fields)
        string_fields = all_fields - numeric_fields

        for col in string_fields:
            if col in df_cleaned.columns:
                try:
                    # 转换为字符串
                    df_cleaned[col] = df_cleaned[col].astype(str)

                    # 清理字符串
                    def clean_string(value):
                        if not isinstance(value, str):
                            return value

                        # 移除控制字符，但保留基本空白字符
                        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)

                        # 去除首尾空格
                        cleaned = cleaned.strip()

                        # 处理空字符串
                        if cleaned == '' or cleaned.lower() in ['nan', 'none', 'null', '']:
                            return None

                        return cleaned

                    df_cleaned[col] = df_cleaned[col].apply(clean_string)
                    print(f"  ✓ {col}: 字符串字段处理完成")
                except Exception as e:
                    print(f"  ✗ {col}: 字符串字段处理失败 - {e}")

        print("数据清洗完成")
        print(f"清洗后数据形状: {df_cleaned.shape}")
        return df_cleaned

    def _prepare_documents(self, df_batch: pd.DataFrame, integer_fields: List, float_fields: List) -> List[Dict]:
        """
        准备批量插入的文档

        Args:
            df_batch: 批次数据框
            integer_fields: 整数字段列表
            float_fields: 浮点数字段列表

        Returns:
            文档列表
        """
        documents = []

        for _, row in df_batch.iterrows():
            # 转换为字典
            doc_data = {}

            for key, value in row.items():
                if value is None or pd.isna(value):
                    doc_data[key] = None
                elif key in integer_fields and isinstance(value, (int, np.integer)):
                    # 整数字段处理
                    try:
                        doc_data[key] = int(value)
                    except (ValueError, TypeError):
                        doc_data[key] = None
                elif key in float_fields and isinstance(value, (float, np.floating)):
                    # 浮点数字段处理
                    if pd.isna(value) or value in [float('inf'), float('-inf'), float('nan')]:
                        doc_data[key] = None
                    else:
                        try:
                            doc_data[key] = float(value)
                        except (ValueError, TypeError):
                            doc_data[key] = None
                elif isinstance(value, (int, np.integer)):
                    # 其他整数值
                    try:
                        doc_data[key] = int(value)
                    except (ValueError, TypeError):
                        doc_data[key] = None
                elif isinstance(value, (float, np.floating)):
                    # 其他浮点值
                    if pd.isna(value) or value in [float('inf'), float('-inf'), float('nan')]:
                        doc_data[key] = None
                    else:
                        try:
                            doc_data[key] = float(value)
                        except (ValueError, TypeError):
                            doc_data[key] = None
                elif isinstance(value, str):
                    # 字符串处理
                    if value.strip() == '' or value.lower() in ['nan', 'none', 'null']:
                        doc_data[key] = None
                    else:
                        # 检查是否为meta字段并限制长度
                        if key.endswith('_meta') or 'meta' in key.lower():
                            # 对meta字段应用50字符限制
                            if len(value) > 50:
                                doc_data[key] = value[:47] + "..."
                                # 可选：记录截断日志
                                # print(f"警告: 字段'{key}'的值超过50字符，已被截断: {value[:50]}...")
                            else:
                                doc_data[key] = value
                        else:
                            doc_data[key] = value
                else:
                    # 其他类型
                    try:
                        doc_data[key] = str(value) if value is not None else None
                    except:
                        doc_data[key] = None

            # 使用UUID作为独立ID
            doc_id = str(uuid.uuid4())

            documents.append({
                "_index": self.current_index,
                "_id": doc_id,
                "_source": doc_data
            })

        return documents

    def _batch_insert(self, df: pd.DataFrame, integer_fields: List, float_fields: List,
                      batch_size: int = 500) -> Tuple[int, int, List]:
        """
        分批次插入数据

        Args:
            df: 数据框
            integer_fields: 整数字段列表
            float_fields: 浮点数字段列表
            batch_size: 批次大小

        Returns:
            (total_success, total_failed, failed_docs)
        """
        print(f"\n" + "=" * 50)
        print(f"步骤4: 分批次插入数据 (批次大小: {batch_size})")
        print("=" * 50)

        total_batches = (len(df) + batch_size - 1) // batch_size
        total_success = 0
        total_failed = 0
        failed_docs = []

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            print(f"\n处理批次 {batch_num + 1}/{total_batches} (行 {start_idx} 到 {end_idx - 1})")

            # 准备本批次文档
            batch_docs = self._prepare_documents(batch_df, integer_fields, float_fields)

            if not batch_docs:
                print(f"  警告: 批次 {batch_num + 1} 没有文档可插入")
                continue

            try:
                # 执行批量插入
                success, failed = helpers.bulk(
                    self.client,
                    batch_docs,
                    raise_on_error=False,
                    max_retries=3,
                    request_timeout=60,
                    stats_only=False
                )

                total_success += success

                if failed:
                    total_failed += len(failed)
                    failed_docs.extend(failed)
                    print(f"  ✗ 本批次失败: {len(failed)} 个文档")

                    # 显示本批次第一个错误
                    if len(failed) > 0:
                        fail = failed[0]
                        if 'index' in fail and 'error' in fail['index']:
                            error_info = fail['index']['error']
                            error_reason = error_info.get('reason', '未知错误')
                            print(f"    示例错误: {error_reason[:100]}...")
                else:
                    print(f"  ✓ 本批次成功: {len(batch_docs)} 个文档")

                # 显示进度
                progress = ((batch_num + 1) / total_batches) * 100
                print(f"  总进度: {progress:.1f}% ({total_success} 成功, {total_failed} 失败)")

            except Exception as e:
                print(f"  ✗ 批次 {batch_num + 1} 插入异常: {e}")
                import traceback
                traceback.print_exc()
                total_failed += len(batch_df)

        return total_success, total_failed, failed_docs

    def import_data(self, mapping: Dict, csv_path: str, index_name: Optional[str] = None,
                    batch_size: int = 500, recreate_index: bool = True) -> Dict[str, Any]:
        """
        主函数：导入数据到OpenSearch

        Args:
            mapping: OpenSearch mapping配置
            csv_path: CSV文件路径
            index_name: 索引名称（如果为None，则从mapping的settings中获取或使用默认值）
            batch_size: 批次大小
            recreate_index: 是否重新创建索引（删除已存在的）

        Returns:
            导入结果统计信息
        """
        if not self.connected:
            return {"success": False, "error": "OpenSearch连接未建立"}

        # 设置当前索引名称
        if index_name:
            self.current_index = index_name
        else:
            # 尝试从mapping中获取索引名称
            if "index" in mapping:
                self.current_index = mapping["index"]
            else:
                # 使用CSV文件名作为索引名
                import os
                self.current_index = os.path.splitext(os.path.basename(csv_path))[0]

        print("=" * 50)
        print(f"开始导入数据到索引: {self.current_index}")
        print("=" * 50)

        # 1. 从mapping自动推断字段类型
        print("\n步骤1: 分析mapping结构")
        print("=" * 50)
        integer_fields, float_fields, keyword_fields = self._infer_field_types_from_mapping(mapping)
        print("从mapping推断的字段类型:")
        print(f"  整数字段: {len(integer_fields)} 个: {integer_fields}")
        print(f"  浮点数字段: {len(float_fields)} 个: {float_fields}")
        print(f"  关键词字段: {len(keyword_fields)} 个")

        # 2. 删除已存在的索引（如果需要）
        if recreate_index:
            print("\n" + "=" * 50)
            print(f"步骤2: 检查并删除现有索引 {self.current_index}")
            print("=" * 50)
            try:
                if self.client.indices.exists(index=self.current_index):
                    print(f"删除已存在的索引 {self.current_index}...")
                    response = self.client.indices.delete(index=self.current_index)
                    if response.get('acknowledged'):
                        print("✓ 索引删除成功")
                    time.sleep(1)
                else:
                    print(f"索引 {self.current_index} 不存在，无需删除")
            except Exception as e:
                print(f"✗ 删除索引时出错: {e}")
                return {"success": False, "error": f"删除索引失败: {e}"}

        # 3. 创建索引
        print("\n" + "=" * 50)
        print("步骤3: 创建索引")
        print("=" * 50)
        try:
            response = self.client.indices.create(index=self.current_index, body=mapping)
            if response.get('acknowledged'):
                print(f"✓ 索引 {self.current_index} 创建成功")

            time.sleep(1)
            if self.client.indices.exists(index=self.current_index):
                print("✓ 索引验证成功")
            else:
                print("✗ 索引创建失败")
                return {"success": False, "error": "索引创建失败"}

        except Exception as e:
            print(f"✗ 创建索引时出错: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"创建索引失败: {e}"}

        # 4. 读取CSV并处理数据
        print("\n" + "=" * 50)
        print("步骤4: 读取和处理CSV数据")
        print("=" * 50)
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ CSV文件读取成功: {len(df)} 行, {len(df.columns)} 列")
            print(f"列名: {list(df.columns)}")

            # 检查CSV中的列是否与mapping匹配
            csv_columns = set(df.columns)
            mapping_columns = set()

            # 获取mapping中的所有字段
            properties = mapping.get("mappings", {})
            if isinstance(properties, dict) and "properties" in properties:
                mapping_columns = set(properties.get("properties", {}).keys())
            elif isinstance(properties, dict):
                mapping_columns = set(properties.get("properties", {}).keys())

            missing_in_csv = mapping_columns - csv_columns
            missing_in_mapping = csv_columns - mapping_columns

            if missing_in_csv:
                print(f"⚠️ 警告: 以下mapping字段在CSV中不存在: {list(missing_in_csv)}")

            if missing_in_mapping:
                print(f"⚠️ 警告: 以下CSV字段在mapping中未定义: {list(missing_in_mapping)}")

        except Exception as e:
            print(f"✗ 读取CSV文件失败: {e}")
            return {"success": False, "error": f"读取CSV文件失败: {e}"}

        # 5. 数据清洗
        print("\n" + "=" * 50)
        print("步骤5: 数据清洗")
        print("=" * 50)
        df = self._clean_data(df, integer_fields, float_fields)
        print(f"✓ 数据清洗完成，共 {len(df)} 行")

        # 6. 执行分批次插入
        print("\n" + "=" * 50)
        print("步骤6: 数据导入")
        print("=" * 50)
        total_success, total_failed, failed_docs = self._batch_insert(df, integer_fields, float_fields, batch_size)

        # 7. 结果统计
        result = {
            "success": total_failed == 0,
            "total_docs": len(df),
            "inserted": total_success,
            "failed": total_failed,
            "index_name": self.current_index,
            "failed_details": failed_docs if failed_docs else []
        }

        if len(df) > 0:
            result["success_rate"] = total_success / len(df) * 100

        print("\n" + "=" * 50)
        print("导入结果统计")
        print("=" * 50)
        print(f"索引名称: {self.current_index}")
        print(f"总文档数: {len(df)}")
        print(f"成功插入: {total_success}")
        print(f"插入失败: {total_failed}")
        if len(df) > 0:
            print(f"成功率: {total_success / len(df) * 100:.2f}%")
        else:
            print("成功率: N/A")

        if total_failed > 0:
            print(f"\n失败文档统计:")

            # 按错误类型统计
            error_stats = {}
            for fail in failed_docs:
                error_type = fail.get('index', {}).get('error', {}).get('type', 'unknown')
                error_stats[error_type] = error_stats.get(error_type, 0) + 1

            if error_stats:
                print(f"错误类型分布:")
                for error_type, count in error_stats.items():
                    print(f"  {error_type}: {count} 个")

            # 显示前3个详细错误
            if len(failed_docs) > 0:
                print(f"\n前3个详细错误:")
                for i, fail in enumerate(failed_docs[:3]):
                    print(f"\n错误 {i + 1}:")
                    if 'index' in fail and 'error' in fail['index']:
                        error = fail['index']['error']
                        print(f"  错误类型: {error.get('type', 'unknown')}")
                        print(f"  错误原因: {error.get('reason', 'unknown')}")

                        if 'caused_by' in error:
                            caused_by = error['caused_by']
                            print(f"  具体原因: {caused_by.get('reason', 'unknown')}")

        print("\n" + "=" * 50)
        print("数据导入完成")
        print("=" * 50)

        return result

    def close(self):
        """关闭连接"""
        if self.client:
            self.client = None
            self.connected = False
            print("OpenSearch连接已关闭")


# 使用示例
if __name__ == "__main__":
    # 定义mapping
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "_meta": {
                "description": "BRSET Dataset. Retinal fundus images with demographics, quality, and pathology labels."
            },
            "properties": {
                "image_id": {
                    "type": "keyword",
                    "meta": {
                        "description": "Unique identifier for the fundus image."
                    }
                },
                "patient_id": {
                    "type": "keyword",
                    "meta": {
                        "description": "Unique identifier for the patient."
                    }
                },
                "camera": {
                    "type": "keyword",
                    "meta": {
                        "description": "Retinal camera model."
                    }
                },
                "patient_age": {
                    "type": "integer",
                    "meta": {
                        "description": "Age of the patient in years. Range: 5-97."
                    }
                },
                "comorbidities": {
                    "type": "keyword",
                    "meta": {
                        "description": "Patient's comorbidities as keywords."
                    }
                },
                "diabetes_time_y": {
                    "type": "float",
                    "meta": {
                        "description": "Diabetes duration in years."
                    }
                },
                "insuline": {
                    "type": "keyword",
                    "meta": {
                        "description": "Insulin usage. e.g., yes/no."
                    }
                },
                "patient_sex": {
                    "type": "keyword",
                    "meta": {
                        "description": "Patient sex. 0: Male, 1: Female."
                    }
                },
                "exam_eye": {
                    "type": "keyword",
                    "meta": {
                        "description": "Eye examined. e.g., OD (right), OS (left)."
                    }
                },
                "diabetes": {
                    "type": "keyword",
                    "meta": {
                        "description": "Diabetes status. e.g., Yes/No."
                    }
                },
                "nationality": {
                    "type": "keyword",
                    "meta": {
                        "description": "Patient nationality, e.g., Brazilian."
                    }
                },
                "optic_disc": {
                    "type": "keyword",
                    "meta": {
                        "description": "Optic disc anatomy. 1: Normal, 2: Abnormal."
                    }
                },
                "vessels": {
                    "type": "keyword",
                    "meta": {
                        "description": "Retinal vessels anatomy. 1: Normal, 2: Abnormal."
                    }
                },
                "macula": {
                    "type": "keyword",
                    "meta": {
                        "description": "Macula anatomy. 1: Normal, 2: Abnormal."
                    }
                },
                "DR_SDRG": {
                    "type": "integer",
                    "meta": {
                        "description": "DR grade (SDRG). 0-4. 0: No retinopathy."
                    }
                },
                "DR_ICDR": {
                    "type": "integer",
                    "meta": {
                        "description": "DR grade (ICDR). 0-4. 0: No retinopathy."
                    }
                },
                "focus": {
                    "type": "keyword",
                    "meta": {
                        "description": "Image focus quality. 1: Adequate, 2: Inadequate."
                    }
                },
                "Illuminaton": {
                    "type": "keyword",
                    "meta": {
                        "description": "Image illumination. 1: Adequate, 2: Inadequate."
                    }
                },
                "image_field": {
                    "type": "keyword",
                    "meta": {
                        "description": "Image field coverage. 1: Adequate, 2: Inadequate."
                    }
                },
                "artifacts": {
                    "type": "keyword",
                    "meta": {
                        "description": "Image artifacts. 1: Adequate, 2: Inadequate."
                    }
                },
                "diabetic_retinopathy": {
                    "type": "integer",
                    "meta": {
                        "description": "Diabetic retinopathy. 1: Present, 0: Absent."
                    }
                },
                "macular_edema": {
                    "type": "integer",
                    "meta": {
                        "description": "Macular edema. 1: Present, 0: Absent."
                    }
                },
                "scar": {
                    "type": "integer",
                    "meta": {
                        "description": "Scar (e.g., Toxoplasmosis). 1: Present, 0: Absent."
                    }
                },
                "nevus": {
                    "type": "integer",
                    "meta": {
                        "description": "Nevus. 1: Present, 0: Absent."
                    }
                },
                "amd": {
                    "type": "integer",
                    "meta": {
                        "description": "Age macular degeneration 1: Present, 0: Absent"
                    }
                },
                "vascular_occlusion": {
                    "type": "integer",
                    "meta": {
                        "description": "Vascular occlusion. 1: Present, 0: Absent."
                    }
                },
                "hypertensive_retinopathy": {
                    "type": "integer",
                    "meta": {
                        "description": "Hypertensive retinopathy. 1: Present, 0: Absent."
                    }
                },
                "drusens": {
                    "type": "integer",
                    "meta": {
                        "description": "Drusens. 1: Present, 0: Absent."
                    }
                },
                "hemorrhage": {
                    "type": "integer",
                    "meta": {
                        "description": "retinal hemorrhage 1: Present, 0: Absent"
                    }
                },
                "retinal_detachment": {
                    "type": "integer",
                    "meta": {
                        "description": "Retinal detachment. 1: Present, 0: Absent."
                    }
                },
                "myopic_fundus": {
                    "type": "integer",
                    "meta": {
                        "description": "Myopic fundus. 1: Present, 0: Absent."
                    }
                },
                "increased_cup_disc": {
                    "type": "integer",
                    "meta": {
                        "description": "Increased cup-disc ratio. 1: Present, 0: Absent."
                    }
                },
                "other": {
                    "type": "integer",
                    "meta": {
                        "description": "Other pathologies. 1: Present, 0: Absent."
                    }
                },
                "quality": {
                    "type": "keyword",
                    "meta": {
                        "description": "Overall image quality grade/flag."
                    }
                }
            }
        }
    }
    # CSV文件路径
    csv_path = r'D:\IDLE\projects\med-data\source-data\labels_brset.csv'

    # 创建导入器实例
    importer = OpenSearchDataImporter(
        host='localhost',
        port=9200,
        username='admin',
        password='S202512sss',
        use_ssl=True,
        verify_certs=False,
        timeout=30
    )

    # 执行数据导入
    result = importer.import_data(
        mapping=mapping,
        csv_path=csv_path,
        index_name="brset",  # 可以指定索引名，不指定则使用CSV文件名
        batch_size=500,
        recreate_index=True
    )

    # 打印结果
    if result["success"]:
        print(f"\n✅ 数据导入成功！共导入 {result['inserted']} 条记录到索引 {result['index_name']}")
    else:
        print(f"\n❌ 数据导入失败！失败 {result['failed']} 条记录")

    # 关闭连接
    importer.close()
