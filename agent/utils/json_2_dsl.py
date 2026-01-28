import json
from typing import Dict, List, Any, Optional



"""
统计查询（Stats Query）

用户请求 → translate() → _build_stats_query()
                
OpenSearch返回结果 → process_result() → _process_stats_result()
"""


"""
分布查询（Distribution Query）

用户请求 → translate() → _build_distribution_query() → _build_distribution_aggregations() → _add_metrics_aggregations()
                
OpenSearch返回结果 → process_result() → _process_distribution_result() → _process_distribution_aggregations() → _create_bucket_result() → _calculate_metrics()
"""


class OpenSearchJsonTranslator:
    """OpenSearch统计分析JSON翻译器"""

    # 支持的统计指标映射
    STATS_METRICS_MAP = {
        'count': 'value_count',
        'min': 'min',
        'max': 'max',
        'avg': 'avg',
        'sum': 'sum',
        'median': 'percentiles',
        'q1': 'percentiles',
        'q3': 'percentiles',
        'q5': 'percentiles',
        'std_deviation': 'extended_stats',
        'variance': 'extended_stats',
        'mode': 'terms',
        'cardinality': 'cardinality'
    }

    def __init__(self):
        self.query = {}

    def translate(self, input_json: Dict) -> Dict:
        """主翻译方法：JSON配置转OpenSearch DSL"""
        try:
            query_type = input_json['query']['type']
            config = input_json['query']['config']

            self._build_base_query(config.get('filters', []))

            if query_type == 'stats':
                return self._build_stats_query(config)
            elif query_type == 'distribution':
                return self._build_distribution_query(config)
            else:
                raise ValueError(f"不支持的查询类型: {query_type}")

        except Exception as e:
            return {'error': str(e)}

    def _build_base_query(self, filters: List[Dict]) -> None:
        """构建基础查询条件"""
        if not filters:
            self.query = {'match_all': {}}
            return

        bool_query = {'bool': {'must': []}}

        for filter_cond in filters:
            field = filter_cond['field']
            operator = filter_cond['operator']
            value = filter_cond.get('value')

            condition = self._build_filter_condition(field, operator, value)
            if condition:
                bool_query['bool']['must'].append(condition)

        self.query = bool_query

    def _build_filter_condition(self, field: str, operator: str, value: Any) -> Optional[Dict]:
        """构建单个过滤条件"""
        if operator == 'eq':
            return {'term': {field: value}}
        elif operator == 'neq':
            return {'bool': {'must_not': [{'term': {field: value}}]}}
        elif operator == 'gt':
            return {'range': {field: {'gt': value}}}
        elif operator == 'gte':
            return {'range': {field: {'gte': value}}}
        elif operator == 'lt':
            return {'range': {field: {'lt': value}}}
        elif operator == 'lte':
            return {'range': {field: {'lte': value}}}
        elif operator == 'in':
            return {'terms': {field: value}}
        elif operator == 'range':
            return {'range': {field: value}}
        elif operator == 'exists':
            return {'exists': {'field': field}}
        elif operator == 'missing':
            return {'bool': {'must_not': [{'exists': {'field': field}}]}}
        else:
            return None

    def _build_stats_query(self, config: Dict) -> Dict:
        """构建统计计算查询"""
        fields = config.get('fields', [])
        metrics = config.get('metrics', ['min', 'max', 'avg', 'count'])

        aggs = {}

        # 为每个字段创建独立的聚合
        for field in fields:
            if (set(metrics) == {'min', 'max', 'avg', 'count'} or
                    set(metrics) == {'min', 'max', 'avg', 'count', 'sum'}):

                aggs[f'{field}_stats'] = {'stats': {'field': field}}

            else:
                # 为每个指标创建单独的聚合
                for metric in metrics:
                    if metric == 'count':
                        aggs[f'{field}_count'] = {'value_count': {'field': field}}
                    elif metric == 'min':
                        aggs[f'{field}_min'] = {'min': {'field': field}}
                    elif metric == 'max':
                        aggs[f'{field}_max'] = {'max': {'field': field}}
                    elif metric == 'avg':
                        aggs[f'{field}_avg'] = {'avg': {'field': field}}
                    elif metric == 'sum':
                        aggs[f'{field}_sum'] = {'sum': {'field': field}}
                    elif metric in ['median', 'q1', 'q3', 'q5']:
                        # 百分位数聚合
                        percents = []
                        if 'median' in metrics or 'q1' in metrics or 'q3' in metrics or 'q5' in metrics:
                            if 'median' in metrics:
                                percents.append(50.0)
                            if 'q1' in metrics:
                                percents.append(25.0)
                            if 'q3' in metrics:
                                percents.append(75.0)
                            if 'q5' in metrics:
                                percents.append(5.0)
                        aggs[f'{field}_percentiles'] = {
                            'percentiles': {'field': field, 'percents': percents}
                        }
                    elif metric in ['std_deviation', 'variance']:
                        # 扩展统计
                        aggs[f'{field}_extended_stats'] = {'extended_stats': {'field': field}}
                    elif metric == 'mode':
                        # 众数（出现最多的值）
                        aggs[f'{field}_mode'] = {'terms': {'field': field, 'size': 1}}

        return {
            'size': 0,
            'query': self.query,
            'aggs': aggs
        }

    def _build_distribution_query(self, config: Dict) -> Dict:
        """构建分布分析查询，支持百分比计算"""
        dimensions = config.get('dimensions', [])
        groups = config.get('groups', [])
        buckets = config.get('buckets', [])
        metrics = config.get('metrics', ['count', 'percentage'])
        metrics_field = config.get('metrics_field')

        # 构建聚合结构
        aggs = self._build_distribution_aggregations(
            dimensions, groups, buckets, metrics, metrics_field
        )

        return {
            'size': 0,
            'query': self.query,
            'aggs': aggs
        }

    def _build_distribution_aggregations(self, dimensions: List[str], groups: List[str],
                                         buckets: List[Dict], metrics: List[str],
                                         metrics_field: str) -> Dict:
        """构建分布分析的聚合结构"""
        aggs = {}
        current_level = aggs

        # 添加总计数用于百分比计算
        if 'percentage' in metrics:
            current_level['_total_count'] = {'value_count': {'field': '_index'}}

        # 构建分组层级
        for group_field in groups:
            current_level[group_field] = {
                'terms': {'field': group_field, 'size': 1000000},
                'aggs': {
                    '_group_count': {'value_count': {'field': '_index'}}  # 分组级别计数
                }
            }
            current_level = current_level[group_field]['aggs']

        # 构建桶聚合
        for bucket in buckets:
            bucket_type = bucket['type']
            bucket_field = bucket['field']

            if bucket_type == 'terms':
                current_level[bucket_field] = {
                    'terms': {'field': bucket_field, 'size': bucket.get('size', 1000000)},
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}  # 桶级别计数
                    }
                }
                current_level = current_level[bucket_field]['aggs']

            elif bucket_type == 'range':
                range_ranges = []
                for range_def in bucket['ranges']:
                    range_spec = {}
                    if 'from' in range_def:
                        range_spec['from'] = range_def['from']
                    if 'to' in range_def:
                        range_spec['to'] = range_def['to']
                    if 'key' in range_def:
                        range_spec['key'] = range_def['key']
                    range_ranges.append(range_spec)

                current_level[bucket_field] = {
                    'range': {'field': bucket_field, 'ranges': range_ranges},
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}
                    }
                }
                current_level = current_level[bucket_field]['aggs']

            elif bucket_type == 'date_histogram':
                current_level[bucket_field] = {
                    'date_histogram': {
                        'field': bucket_field,
                        'interval': bucket['interval'],
                        'format': bucket.get('format', 'yyyy-MM')
                    },
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}
                    }
                }
                current_level = current_level[bucket_field]['aggs']

        # 构建维度聚合
        for dim_field in dimensions:
            current_level[dim_field] = {
                'terms': {'field': dim_field, 'size': 100},
                'aggs': {
                    '_dimension_count': {'value_count': {'field': '_index'}}
                }
            }
            current_level = current_level[dim_field]['aggs']

        # 添加指标计算
        self._add_metrics_aggregations(current_level, metrics, metrics_field)

        return aggs

    def _add_metrics_aggregations(self, aggs: Dict, metrics: List[str], metrics_field: str) -> None:
        """添加指标计算聚合"""
        for metric in metrics:
            if metric == 'count':
                aggs['count'] = {'value_count': {'field': '_index'}}
            elif metric in ['avg', 'sum', 'min', 'max'] and metrics_field:
                es_metric = self.STATS_METRICS_MAP.get(metric, metric)
                aggs[metric] = {es_metric: {'field': metrics_field}}

    def process_result(self, es_result: Dict, original_config: Dict) -> Dict:
        """
        统一的结果处理方法：自动识别查询类型并调用相应的处理逻辑
        """
        try:
            query_type = original_config['query']['type']

            if query_type == 'stats':
                return self._process_stats_result(es_result, original_config)
            elif query_type == 'distribution':
                return self._process_distribution_result(es_result, original_config)
            else:
                return {'error': f'不支持的查询类型: {query_type}'}

        except Exception as e:
            return {'error': f'结果处理错误: {str(e)}'}

    def _process_stats_result(self, es_result: Dict, original_config: Dict) -> Dict:
        """处理统计计算结果"""
        result = {}
        config = original_config['query']['config']
        fields = config.get('fields', [])
        metrics = config.get('metrics', ['min', 'max', 'avg', 'count'])

        aggregations = es_result.get('aggregations', {})

        for field in fields:
            field_result = {}

            # 检查是否有stats聚合（基本统计）
            stats_key = f'{field}_stats'
            if stats_key in aggregations:
                stats_data = aggregations[stats_key]
                if 'min' in metrics:
                    field_result['min'] = stats_data.get('min')
                if 'max' in metrics:
                    field_result['max'] = stats_data.get('max')
                if 'avg' in metrics:
                    field_result['avg'] = stats_data.get('avg')
                if 'count' in metrics:
                    field_result['count'] = stats_data.get('count')
                if 'sum' in metrics:
                    field_result['sum'] = stats_data.get('sum')
            else:
                # 处理单独的指标聚合
                for metric in metrics:
                    agg_key = f'{field}_{metric}'

                    if metric == 'count' and agg_key in aggregations:
                        field_result['count'] = aggregations[agg_key].get('value', 0)
                    elif metric in ['min', 'max', 'avg', 'sum'] and agg_key in aggregations:
                        field_result[metric] = aggregations[agg_key].get('value')
                    elif metric in ['std_deviation', 'variance'] and f'{field}_extended_stats' in aggregations:
                        ext_stats = aggregations[f'{field}_extended_stats']
                        if metric == 'std_deviation':
                            field_result['std_deviation'] = ext_stats.get('std_deviation')
                        else:
                            field_result['variance'] = ext_stats.get('variance')
                    elif metric in ['median', 'q1', 'q3', 'q5'] and f'{field}_percentiles' in aggregations:
                        percentiles = aggregations[f'{field}_percentiles'].get('values', {})
                        key_map = {'median': '50.0', 'q1': '25.0', 'q3': '75.0', 'q5': '5.0'}
                        key = key_map.get(metric)
                        if key in percentiles:
                            field_result[metric] = percentiles[key]
                    elif metric == 'mode' and f'{field}_mode' in aggregations:
                        buckets = aggregations[f'{field}_mode'].get('buckets', [])
                        if buckets:
                            field_result['mode'] = buckets[0].get('key')
                            field_result['mode_count'] = buckets[0].get('doc_count')

            result[field] = field_result

        return result

    def _process_distribution_result(self, es_result: Dict, original_config: Dict) -> Dict:
        """处理分布分析结果，支持百分比计算"""
        aggregations = es_result.get('aggregations', {})
        config = original_config['query']['config']

        # 获取总计数用于百分比计算
        total_count = aggregations.get('_total_count', {}).get('value', 0)

        return self._process_distribution_aggregations(
            aggregations, config, total_count, level=0
        )

    def _process_distribution_aggregations(self, aggs: Dict, config: Dict,
                                           parent_total: int, level: int = 0) -> Dict:
        """递归处理分布分析聚合结果"""
        result = {'buckets': []}

        # 获取当前层级的聚合键
        aggregation_keys = [k for k in aggs.keys() if not k.startswith('_')]

        for agg_key in aggregation_keys:
            agg_data = aggs[agg_key]

            if 'buckets' in agg_data:
                # 处理桶聚合
                buckets = agg_data['buckets']
                current_level_total = sum(bucket.get('doc_count', 0) for bucket in buckets)

                for bucket in buckets:
                    # 处理当前桶
                    bucket_result = self._create_bucket_result(bucket, config, parent_total, current_level_total)

                    # 递归处理子聚合
                    sub_aggs = {k: v for k, v in bucket.items()
                                if k not in ['key', 'from', 'to', 'doc_count', 'key_as_string']}

                    if sub_aggs and level < 5:  # 防止无限递归
                        sub_result = self._process_distribution_aggregations(
                            sub_aggs, config, bucket.get('doc_count', 0), level + 1
                        )
                        if sub_result.get('buckets'):
                            bucket_result['sub_aggregations'] = sub_result

                    result['buckets'].append(bucket_result)

        return result

    def _create_bucket_result(self, bucket: Dict, config: Dict,
                              parent_total: int, current_level_total: int) -> Dict:
        """创建桶结果"""
        bucket_result = {
            'key': bucket.get('key'),
            'doc_count': bucket.get('doc_count', 0)
        }

        # 只有 range 聚合才设置 from 和 to
        if bucket.get('from') is not None or bucket.get('to') is not None:
            bucket_result['from'] = bucket.get('from')
            bucket_result['to'] = bucket.get('to')

        # 只有需要格式化的聚合才设置 key_as_string
        if 'key_as_string' in bucket:
            bucket_result['key_as_string'] = bucket.get('key_as_string')

        # 计算指标
        metrics_result = self._calculate_metrics(bucket, config, parent_total)
        if metrics_result:
            bucket_result['metrics'] = metrics_result

        return bucket_result

    def _calculate_metrics(self, bucket: Dict, config: Dict, parent_total: int) -> Dict:
        """计算指标"""
        metrics = config.get('metrics', ['count', 'percentage'])
        metrics_field = config.get('metrics_field')
        metrics_result = {}

        for metric in metrics:
            if metric == 'count':
                metrics_result['count'] = bucket.get('doc_count', 0)

            elif metric == 'percentage':
                if parent_total > 0:
                    percentage = (bucket.get('doc_count', 0) / parent_total) * 100
                    metrics_result['percentage'] = round(percentage, 2)
                else:
                    metrics_result['percentage'] = 0.0

            elif metric in ['avg', 'sum', 'min', 'max'] and metrics_field:
                metric_value = bucket.get(metric, {}).get('value')
                if metric_value is not None:
                    metrics_result[metric] = metric_value

        return metrics_result

    def flatten_result(self, processed_result: Dict, original_config: Dict) -> List[Dict]:
        """
        扁平化处理结果，将嵌套结构转换为表格行格式，便于展示和分析

        对于 stats 查询：将字段级统计转换为行列表，每行包含字段名和对应指标
        对于 distribution 查询：将嵌套桶结构转换为交叉表格式，每行包含完整的维度路径和指标

        Args:
            processed_result: process_result 方法的输出结果
            original_config: 原始输入配置，用于确定维度层级

        Returns:
            List[Dict]: 扁平化的行列表，每行是一个字典，包含所有维度和指标
        """
        try:
            query_type = original_config['query']['type']

            if query_type == 'stats':
                return self._flatten_stats_result(processed_result)
            elif query_type == 'distribution':
                return self._flatten_distribution_result(processed_result, original_config)
            else:
                return processed_result

        except Exception as e:
            return [{'error': f'扁平化处理错误: {str(e)}'}]

    def _flatten_stats_result(self, result: Dict) -> List[Dict]:
        """
        扁平化 stats 结果
        输入: {"age": {"min": 20, "max": 65}, "salary": {"min": 3000, ...}}
        输出: [{"field": "age", "min": 20, "max": 65}, {"field": "salary", ...}]
        """
        flattened = []

        for field_name, metrics in result.items():
            row = {'field': field_name}

            # 添加所有指标值
            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                row['value'] = metrics

            flattened.append(row)

        return flattened

    def _flatten_distribution_result(self, result: Dict, original_config: Dict) -> List[Dict]:
        """
        扁平化 distribution 结果，将嵌套桶结构转换为二维表
        输入: {"buckets": [{"key": "male", "sub_aggregations": {"buckets": [...]}}]}
        输出: [{"gender": "male", "age": "20-30", "count": 100, "percentage": 10}, ...]
        """
        config = original_config['query']['config']

        # 构建维度层级顺序（从外层到内层）
        hierarchy = []

        # 1. 分组维度 (groups)
        if 'groups' in config:
            hierarchy.extend(config['groups'])

        # 2. 分桶维度 (buckets)
        if 'buckets' in config:
            for bucket in config['buckets']:
                hierarchy.append(bucket['field'])

        # 3. 分析维度 (dimensions)
        if 'dimensions' in config:
            hierarchy.extend(config['dimensions'])

        rows = []

        def recurse(current_data: Dict, current_values: Dict, depth: int):
            """
            递归遍历嵌套桶结构
            current_data: 当前层级的数据（包含 buckets 的字典）
            current_values: 当前路径累积的维度值
            depth: 当前层级深度
            """
            if not isinstance(current_data, dict) or 'buckets' not in current_data:
                return

            for bucket in current_data['buckets']:
                # 复制当前路径值，避免引用问题
                new_values = current_values.copy()

                # 记录当前层级的维度值（如果有定义层级名称）
                if depth < len(hierarchy):
                    dim_name = hierarchy[depth]
                    new_values[dim_name] = bucket.get('key')
                else:
                    # 如果超出预定义层级，使用通用名称
                    new_values[f'level_{depth}'] = bucket.get('key')

                # 检查是否有子聚合
                if 'sub_aggregations' in bucket and bucket['sub_aggregations']:
                    recurse(bucket['sub_aggregations'], new_values, depth + 1)
                else:
                    # 叶子节点：构建完整数据行
                    row = new_values.copy()

                    # 添加指标数据（优先使用 metrics，否则使用 doc_count）
                    if 'metrics' in bucket:
                        row.update(bucket['metrics'])
                    else:
                        row['doc_count'] = bucket.get('doc_count', 0)

                    # 保留 range 类型的范围信息
                    if 'from' in bucket:
                        row['range_from'] = bucket['from']
                    if 'to' in bucket:
                        row['range_to'] = bucket['to']
                    if 'key_as_string' in bucket:
                        row['key_string'] = bucket['key_as_string']

                    rows.append(row)

        # 开始递归处理
        recurse(result, {}, 0)
        return rows


def test_opensearch_stats_translator():
    """测试OpenSearch统计翻译器"""

    # 创建翻译器实例
    translator = OpenSearchJsonTranslator()

    def test_basic_stats_query():
        """测试基础统计查询"""
        print("=== 测试基础统计查询 ===")

        # 模拟输入
        input_json = {
            "query": {
                "type": "stats",
                "config": {
                    "fields": ["age", "salary"],
                    "metrics": ["min", "max", "avg", "count", "median"],
                    "filters": [
                        {
                            "field": "department",
                            "operator": "eq",
                            "value": "engineering"
                        }
                    ]
                }
            }
        }

        # 翻译查询
        result = translator.translate(input_json)
        print("生成的DSL:", json.dumps(result, indent=2, ensure_ascii=False))

        # 模拟OpenSearch返回结果
        mock_es_result = {
            "aggregations": {
                "age": {
                    "min": {"value": 20},
                    "max": {"value": 65},
                    "avg": {"value": 35.5},
                    "count": {"value": 1000},
                    "percentiles": {"values": {"50.0": 35.0}}
                },
                "salary": {
                    "min": {"value": 3000},
                    "max": {"value": 50000},
                    "avg": {"value": 15000.0},
                    "count": {"value": 1000},
                    "percentiles": {"values": {"50.0": 12000.0}}
                }
            }
        }

        # 处理结果
        processed_result = translator.process_result(mock_es_result, input_json)
        print("处理后的结果:", json.dumps(processed_result, indent=2, ensure_ascii=False))

        f_result = translator.flatten_result(processed_result, input_json)
        print("扁平后的结果:", json.dumps(f_result, indent=2, ensure_ascii=False))
        return True

    def test_age_disease_distribution():
        """测试不同年龄段是否患病比例分布"""
        print("\n=== 测试年龄段患病分布 ===")

        # 模拟输入：分析不同年龄段是否患病的分布
        input_json = {
            "query": {
                "type": "distribution",
                "config": {
                    "dimensions": ["has_disease"],
                    "buckets": [
                        {
                            "type": "range",
                            "field": "age",
                            "ranges": [
                                {"key": "青年", "from": 0, "to": 30},
                                {"key": "中年", "from": 30, "to": 60},
                                {"key": "老年", "from": 60}
                            ]
                        }
                    ],
                    "metrics": ["count", "percentage"],
                    "filters": [
                        {
                            "field": "data_source",
                            "operator": "eq",
                            "value": "medical_survey"
                        }
                    ]
                }
            }
        }

        # 翻译查询
        result = translator.translate(input_json)
        print("生成的分布分析DSL:", json.dumps(result, indent=2, ensure_ascii=False))

        # 模拟OpenSearch返回的分布分析结果
        mock_es_result = {
            "aggregations": {
                "_total_count": {"value": 3000},
                "age": {
                    "buckets": [
                        {
                            "key": "青年",
                            "from": 0,
                            "to": 30,
                            "doc_count": 1000,
                            "has_disease": {
                                "buckets": [
                                    {"key": "true", "doc_count": 100},
                                    {"key": "false", "doc_count": 900}
                                ]
                            }
                        },
                        {
                            "key": "中年",
                            "from": 30,
                            "to": 60,
                            "doc_count": 1500,
                            "has_disease": {
                                "buckets": [
                                    {"key": "true", "doc_count": 300},
                                    {"key": "false", "doc_count": 1200}
                                ]
                            }
                        },
                        {
                            "key": "老年",
                            "from": 60,
                            "doc_count": 500,
                            "has_disease": {
                                "buckets": [
                                    {"key": "true", "doc_count": 200},
                                    {"key": "false", "doc_count": 300}
                                ]
                            }
                        }
                    ]
                }
            }
        }

        # 处理分布分析结果
        processed_result = translator.process_result(mock_es_result, input_json)
        print("患病分布结果:", json.dumps(processed_result, indent=2, ensure_ascii=False))

        f_result = translator.flatten_result(processed_result, input_json)
        print("扁平后的结果:", json.dumps(f_result, indent=2, ensure_ascii=False))
        return True

    def test_complex_distribution():
        """测试复杂分布分析：年龄段+性别+患病状态的交叉分析"""
        print("\n=== 测试复杂分布分析 ===")

        input_json = {
            "query": {
                "type": "distribution",
                "config": {
                    "dimensions": ["has_disease", "disease_type"],
                    "groups": ["gender"],
                    "buckets": [
                        {
                            "type": "range",
                            "field": "age",
                            "ranges": [
                                {"key": "20-30", "from": 20, "to": 30},
                                {"key": "30-40", "from": 30, "to": 40},
                                {"key": "40-50", "from": 40, "to": 50}
                            ]
                        }
                    ],
                    "metrics": ["count", "percentage"],
                    "filters": [
                        {
                            "field": "survey_year",
                            "operator": "gte",
                            "value": 2020
                        }
                    ]
                }
            }
        }

        # 翻译查询
        result = translator.translate(input_json)
        print("复杂分布分析DSL:", json.dumps(result, indent=2, ensure_ascii=False))

        # 模拟复杂返回结果
        mock_es_result = {
            "aggregations": {
                "_total_count": {"value": 5000},
                "age": {
                    "buckets": [
                        {
                            "key": "20-30",
                            "from": 20,
                            "to": 30,
                            "doc_count": 2000,
                            "gender": {
                                "buckets": [
                                    {
                                        "key": "male",
                                        "doc_count": 1000,
                                        "has_disease": {
                                            "buckets": [
                                                {"key": "true", "doc_count": 50},
                                                {"key": "false", "doc_count": 950}
                                            ]
                                        }
                                    },
                                    {
                                        "key": "female",
                                        "doc_count": 1000,
                                        "has_disease": {
                                            "buckets": [
                                                {"key": "true", "doc_count": 30},
                                                {"key": "false", "doc_count": 970}
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }

        # 处理结果
        processed_result = translator.process_result(mock_es_result, input_json)
        print("复杂分布结果:", json.dumps(processed_result, indent=2, ensure_ascii=False))

        f_result = translator.flatten_result(processed_result, input_json)
        print("扁平后的结果:", json.dumps(f_result, indent=2, ensure_ascii=False))

        return True

    def test_error_handling():
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")

        # 测试无效查询类型
        invalid_input = {
            "query": {
                "type": "invalid_type",
                "config": {}
            }
        }

        result = translator.translate(invalid_input)
        print("错误查询结果:", result)

        # 测试无效操作符
        invalid_filter_input = {
            "query": {
                "type": "stats",
                "config": {
                    "fields": ["age"],
                    "filters": [
                        {
                            "field": "department",
                            "operator": "invalid_operator",
                            "value": "test"
                        }
                    ]
                }
            }
        }

        result = translator.translate(invalid_filter_input)
        print("无效操作符结果:", json.dumps(result, indent=2, ensure_ascii=False))

        return True

    # 执行所有测试
    try:
        test_basic_stats_query()
        test_age_disease_distribution()
        test_complex_distribution()
        test_error_handling()
        print("\n=== 所有测试执行完成 ===")
        return True
    except Exception as e:
        print(f"测试执行失败: {e}")
        return False


# 运行测试
if __name__ == "__main__":
    test_opensearch_stats_translator()
