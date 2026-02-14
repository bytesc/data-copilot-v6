import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import string
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_random_filename(length: int = 8) -> str:
    """Generate random lowercase filename with .png extension"""
    return ''.join(random.choices(string.ascii_lowercase, k=length)) + '.png'


def ensure_directory(directory: str):
    """Ensure target directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def extract_chart_metadata(query_json: Dict[str, Any], result_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract title, xlabel, ylabel from query and result JSON based on interface spec.
    """
    query = query_json.get('query', {})
    config = query.get('config', {})
    query_type = query.get('type', '')
    chart_type = query.get('chart_type', '')

    metadata = {
        'title': '',
        'xlabel': '',
        'ylabel': ''
    }

    # Build title components
    title_parts = []

    if query_type == 'stats':
        fields = config.get('fields', [])
        metrics = config.get('metrics', ['min', 'max', 'avg', 'count'])

        if fields:
            title_parts.append(f"Field Statistics: {', '.join(fields)}")
        if chart_type:
            title_parts.append(f"Chart: {chart_type.capitalize()}")

        # Axis labels for stats
        if chart_type == 'histogram':
            metadata['xlabel'] = 'Statistical Metrics'
            metadata['ylabel'] = 'Values'
        elif chart_type == 'bar':
            metadata['xlabel'] = 'Fields'
            metadata['ylabel'] = metrics[0].capitalize() if metrics else 'Value'

    elif query_type == 'distribution':
        dimensions = config.get('dimensions', [])
        groups = config.get('groups', [])
        metrics = config.get('metrics', ['count', 'percentage'])
        metrics_field = config.get('metrics_field', '')

        if dimensions:
            title_parts.append(f"Distribution Analysis: {', '.join(dimensions)}")
        if groups:
            title_parts.append(f"Grouped by: {', '.join(groups)}")
        if metrics_field:
            title_parts.append(f"Metric Field: {metrics_field}")

        # Axis labels for distribution
        if chart_type in ['bar', 'histogram']:
            metadata['xlabel'] = dimensions[0] if dimensions else 'Categories'
            metric_name = metrics[0] if metrics else 'count'
            metadata['ylabel'] = f"{metric_name.capitalize()}{f' of {metrics_field}' if metrics_field else ''}"
        elif chart_type == 'pie':
            # Pie charts typically don't need axis labels
            metadata['title'] = f"Distribution of {dimensions[0] if dimensions else 'Categories'}"
        elif chart_type == 'heatmap':
            # 热力图：根据数据结构确定横纵坐标
            # 外层桶（buckets字段）对应X轴，内层桶（sub_aggregations/groups）对应Y轴
            buckets = config.get('buckets', [])

            # X轴标签：优先使用buckets的field字段，其次是dimensions的第一个字段
            if buckets and len(buckets) > 0:
                xlabel_source = buckets[0].get('field', '')
                if xlabel_source:
                    metadata['xlabel'] = xlabel_source
                elif len(dimensions) > 0:
                    metadata['xlabel'] = dimensions[0]
            elif len(dimensions) > 0:
                metadata['xlabel'] = dimensions[0]

            # Y轴标签：优先使用dimensions字段（分析目标），其次是metrics_field
            if len(dimensions) > 0:
                # 如果dimensions只有一个，那就是分析目标；如果有多个，第二个可能是分组
                metadata['ylabel'] = dimensions[0]
            elif metrics_field:
                metadata['ylabel'] = metrics_field

            # 热力图标题
            title_components = []
            if len(dimensions) > 0:
                title_components.append(dimensions[0])
            if buckets and len(buckets) > 0 and buckets[0].get('field'):
                title_components.append(buckets[0].get('field'))
            if metrics_field:
                title_components.append(f"by {metrics_field}")

            if title_components:
                metadata['title'] = f"Heatmap: {' vs '.join(title_components)}"
            else:
                metadata['title'] = 'Heatmap Analysis'

    metadata['title'] = ' | '.join(title_parts) if title_parts and not metadata.get('title') else (
                metadata.get('title') or 'OpenSearch Analysis Visualization')

    return metadata

def parse_distribution_buckets(result_json: Dict[str, Any], max_depth: int = 10) -> pd.DataFrame:
    """
    Parse distribution result buckets into DataFrame for easier plotting.
    Handles nested bucket structures including sub_aggregations format.
    """
    buckets = result_json.get('buckets', [])
    if not buckets:
        return pd.DataFrame()

    records = []

    def extract_bucket_data(bucket: Dict, level: int = 0, parent_keys: Dict = None):
        """Recursively extract nested bucket data"""
        if level > max_depth:
            return

        if parent_keys is None:
            parent_keys = {}

        record = {
            'key': bucket.get('key', ''),
            'doc_count': bucket.get('doc_count', 0),
            'level': level
        }

        # Add metrics
        metrics = bucket.get('metrics', {})
        for metric_key, metric_val in metrics.items():
            record[metric_key] = metric_val

        # Add parent context
        for pk, pv in parent_keys.items():
            record[pk] = pv

        records.append(record)

        # Handle sub_aggregations (translated result format)
        sub_ags = bucket.get('sub_aggregations', {})
        if sub_ags and 'buckets' in sub_ags:
            sub_buckets = sub_ags['buckets']
            for sub_bucket in sub_buckets:
                new_parent_keys = parent_keys.copy()
                new_parent_keys[f'level_{level}_group'] = bucket.get('key')
                extract_bucket_data(sub_bucket, level + 1, new_parent_keys)

        # Handle nested dimensions (native opensearch format)
        dimensions = bucket.get('dimensions', {})
        for dim_name, dim_data in dimensions.items():
            sub_buckets = dim_data.get('buckets', [])
            for sub_bucket in sub_buckets:
                new_parent_keys = parent_keys.copy()
                new_parent_keys[f'level_{level}_key'] = bucket.get('key')
                new_parent_keys[f'dimension_{level}'] = dim_name
                extract_bucket_data(sub_bucket, level + 1, new_parent_keys)

        # Handle nested groups (native opensearch format)
        groups = bucket.get('groups', {})
        for group_name, group_data in groups.items():
            sub_buckets = group_data.get('buckets', [])
            for sub_bucket in sub_buckets:
                new_parent_keys = parent_keys.copy()
                new_parent_keys[f'group_{group_name}'] = bucket.get('key')
                extract_bucket_data(sub_bucket, level + 1, new_parent_keys)

    for bucket in buckets:
        extract_bucket_data(bucket)

    return pd.DataFrame(records)


def is_range_bucket(bucket: Dict) -> bool:
    """Check if bucket represents a range (has 'from' or 'to')"""
    return 'from' in bucket or 'to' in bucket


def determine_chart_strategy(df: pd.DataFrame, query_config: Dict[str, Any], chart_type: str = '') -> str:
    """
    Determine the best chart strategy based on data structure.
    """
    # 如果明确指定了热力图，优先返回heatmap策略
    if chart_type == 'heatmap':
        return 'heatmap'

    # 新增：如果明确指定了histogram且有嵌套结构，使用叠加直方图
    if chart_type == 'histogram':
        # 检查是否有嵌套结构（用于叠加显示）
        parent_cols = [c for c in df.columns if c.startswith('level_') or c.startswith('group_')]
        if parent_cols or 'level_0_group' in df.columns:
            return 'stacked_histogram'

    if df.empty:
        return 'single_bar'

    # Check for histogram/range data
    if 'from' in df.columns or 'to' in df.columns:
        return 'histogram'

    # Check if single layer (no parent keys)
    parent_cols = [c for c in df.columns if c.startswith('level_') or c.startswith('group_')]
    if not parent_cols:
        # Check if has percentage for pie combo
        if 'percentage' in df.columns or 'metrics' in str(df.to_dict()):
            return 'bar_with_pie'
        return 'single_bar'

    # Check outer buckets count (level_0_group indicates outer grouping)
    if 'level_0_group' in df.columns:
        outer_groups = df['level_0_group'].nunique()
        if outer_groups > 1:
            return 'grouped_bar'
        else:
            # Only one outer group (likely due to filter), flatten to single bar
            return 'single_bar'

    return 'single_bar'

def plot_single_bar(ax, df: pd.DataFrame, metadata: Dict[str, str], primary_metric: str = 'doc_count',
                    show_pie: bool = False):
    """Plot simple bar chart with axis labels from metadata"""
    if primary_metric not in df.columns:
        primary_metric = 'doc_count'

    values = df[primary_metric].values
    keys = df['key'].astype(str).values

    # Limit categories if too many
    if len(keys) > 15:
        sorted_idx = np.argsort(values)[::-1]
        top_n = 14
        top_idx = sorted_idx[:top_n]
        other_sum = values[sorted_idx[top_n:]].sum()

        keys = list(keys[top_idx]) + ['Others']
        values = list(values[top_idx]) + [other_sum]

    colors = sns.color_palette("husl", len(keys))

    if show_pie and len(keys) <= 10:
        # Create two subplots side by side
        ax.set_visible(False)
        ax1 = ax.figure.add_subplot(121)
        ax2 = ax.figure.add_subplot(122)

        # Bar chart with metadata labels
        bars = ax1.bar(keys, values, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels(keys, rotation=45, ha='right')

        # 使用 metadata 中的标签
        if metadata.get('xlabel'):
            ax1.set_xlabel(metadata['xlabel'], fontsize=11)
        if metadata.get('ylabel'):
            ax1.set_ylabel(metadata['ylabel'], fontsize=11)
        else:
            ax1.set_ylabel(primary_metric.capitalize(), fontsize=11)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # Pie chart
        wedges, texts, autotexts = ax2.pie(values, labels=keys, autopct='%1.1f%%',
                                           startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title(f'{metadata.get("title", "Distribution")} (Pie View)', fontsize=11)

    else:
        # Simple bar chart with metadata labels
        bars = ax.bar(keys, values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')

        # 设置横纵坐标标签
        if metadata.get('xlabel'):
            ax.set_xlabel(metadata['xlabel'], fontsize=12)
        if metadata.get('ylabel'):
            ax.set_ylabel(metadata['ylabel'], fontsize=12)
        else:
            ax.set_ylabel(primary_metric.capitalize(), fontsize=12)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)


def plot_grouped_bar(result_json: Dict[str, Any], query_config: Dict[str, Any],
                     metadata: Dict[str, str], save_dir: str) -> str:
    """Plot grouped bar charts with axis labels from metadata"""
    buckets = result_json.get('buckets', [])
    if not buckets:
        return None

    n_plots = len(buckets)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes.flatten()

    dimensions = query_config.get('dimensions', [])
    metrics = query_config.get('metrics', ['count'])
    primary_metric = metrics[0] if metrics else 'doc_count'

    for idx, bucket in enumerate(buckets):
        ax = axes[idx] if n_plots > 1 else axes[0]
        outer_key = bucket.get('key', f'Group {idx}')

        sub_ags = bucket.get('sub_aggregations', {})
        inner_buckets = sub_ags.get('buckets', []) if sub_ags else []

        if not inner_buckets:
            dims = bucket.get('dimensions', {})
            if dims and dimensions:
                inner_buckets = dims.get(dimensions[0], {}).get('buckets', [])

        if inner_buckets:
            keys = [b.get('key', '') for b in inner_buckets]
            values = []
            for b in inner_buckets:
                if primary_metric in b:
                    values.append(b[primary_metric])
                elif 'metrics' in b and primary_metric in b['metrics']:
                    values.append(b['metrics'][primary_metric])
                else:
                    values.append(b.get('doc_count', 0))

            colors = sns.color_palette("husl", len(keys))
            bars = ax.bar(keys, values, color=colors, alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

            ax.set_title(f'{outer_key}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')

            # 设置横纵坐标标签：使用 metadata 或根据配置生成
            if metadata.get('xlabel'):
                ax.set_xlabel(metadata['xlabel'], fontsize=11)
            else:
                ax.set_xlabel(dimensions[0] if dimensions else 'Category', fontsize=11)

            if metadata.get('ylabel'):
                ax.set_ylabel(metadata['ylabel'], fontsize=11)
            else:
                ax.set_ylabel(f'{primary_metric.capitalize()}', fontsize=11)
        else:
            ax.text(0.5, 0.5, f'No data for {outer_key}',
                    ha='center', va='center', transform=ax.transAxes)

    for idx in range(n_plots, len(axes) if n_plots > 1 else 1):
        if n_plots > 1:
            axes[idx].axis('off')

    # 添加总标题
    fig.suptitle(metadata.get('title', 'OpenSearch Analysis'), fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为 suptitle 留出空间

    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath


def plot_heatmap(result_json: Dict[str, Any], query_config: Dict[str, Any],
                 metadata: Dict[str, str], save_dir: str) -> str:
    """
    通用热力图绘制函数：处理任意两层嵌套桶结构
    外层桶 -> 列 (X轴)
    内层桶 -> 行 (Y轴)
    数值 -> 单元格颜色强度

    单元格显示格式：主数值 (百分比)
    注意：始终生成单张热力图，不使用子图
    """
    buckets = result_json.get('buckets', [])
    if not buckets:
        return None

    dimensions = query_config.get('dimensions', [])
    groups = query_config.get('groups', [])
    metrics = query_config.get('metrics', ['count', 'percentage'])
    metrics_field = query_config.get('metrics_field', '')
    buckets_config = query_config.get('buckets', [])

    primary_metric = metrics[0] if metrics else 'count'
    # 确定百分比字段名称（可能是'percentage'或在metrics中）
    percentage_metric = 'percentage' if 'percentage' in metrics else (metrics[1] if len(metrics) > 1 else None)

    # 判断数据结构：单层分组还是双层分组
    has_nested = False
    for bucket in buckets:
        sub_ags = bucket.get('sub_aggregations', {})
        if sub_ags and 'buckets' in sub_ags and len(sub_ags['buckets']) > 0:
            has_nested = True
            break
        dims = bucket.get('dimensions', {})
        if dims:
            for dim_data in dims.values():
                if dim_data.get('buckets'):
                    has_nested = True
                    break
        if has_nested:
            break

    # 单层数据：创建1xN矩阵（行固定为metrics_field或primary_metric，列为外层桶）
    if not has_nested:
        fig, ax = plt.subplots(figsize=(10, 6))

        col_labels = [str(b.get('key', '')) for b in buckets]
        # 行标签：优先使用metrics_field，其次是primary_metric
        row_label = metrics_field if metrics_field else primary_metric
        row_labels = [row_label.capitalize() if isinstance(row_label, str) else str(row_label)]

        # 构建主数值矩阵和百分比矩阵
        values_matrix = []
        percentage_matrix = []

        for bucket in buckets:
            # 获取主数值
            if primary_metric in bucket:
                main_val = bucket[primary_metric]
            elif 'metrics' in bucket and primary_metric in bucket['metrics']:
                main_val = bucket['metrics'][primary_metric]
            else:
                main_val = bucket.get('doc_count', 0)
            values_matrix.append(main_val)

            # 获取百分比数值
            pct_val = None
            if percentage_metric and percentage_metric in bucket:
                pct_val = bucket[percentage_metric]
            elif 'metrics' in bucket and percentage_metric and percentage_metric in bucket['metrics']:
                pct_val = bucket['metrics'][percentage_metric]
            percentage_matrix.append(pct_val)

        # 单层数据：1行 x N列
        matrix = [values_matrix]

        # 构建注释矩阵：格式为 "数值 (百分比%)"
        annot_matrix = []
        row_annotations = []
        for main_val, pct_val in zip(values_matrix, percentage_matrix):
            if pct_val is not None:
                # 根据数值大小决定格式，避免小数位过多
                if isinstance(pct_val, float):
                    annot_str = f'{int(main_val)}\n({pct_val:.2f}%)'
                else:
                    annot_str = f'{int(main_val)}\n({pct_val}%)'
            else:
                annot_str = str(int(main_val))
            row_annotations.append(annot_str)
        annot_matrix.append(row_annotations)

        df_pivot = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

        # 使用自定义注释矩阵
        sns.heatmap(df_pivot, annot=annot_matrix, fmt='',
                    cmap='YlOrRd', ax=ax, cbar_kws={'label': primary_metric.capitalize()},
                    linewidths=0.5, linecolor='gray')

        ax.set_title(metadata.get('title', 'Heatmap Analysis'), fontsize=14, fontweight='bold')

        # 设置轴标签：优先使用metadata，其次从query_config推断
        # X轴：外层桶的field（如patient_age）或dimensions[0]
        xlabel = metadata.get('xlabel')
        if not xlabel and buckets_config and len(buckets_config) > 0:
            xlabel = buckets_config[0].get('field', '')
        if not xlabel and len(dimensions) > 0:
            xlabel = dimensions[0]
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)

        # Y轴：metrics_field或dimensions[0]（分析目标）
        ylabel = metadata.get('ylabel')
        if not ylabel:
            ylabel = metrics_field if metrics_field else (dimensions[0] if len(dimensions) > 0 else primary_metric)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

        plt.tight_layout()
        filename = generate_random_filename(8)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath

    # 双层数据：外层桶 -> 列(X轴)，内层桶 -> 行(Y轴)
    fig, ax = plt.subplots(figsize=(12, 8))

    # 收集所有可能的内层键（用于统一行顺序）- 现在内层键作为行
    all_inner_keys = set()
    bucket_data_list = []

    for bucket in buckets:
        outer_key = str(bucket.get('key', ''))
        inner_data = {}  # 存储主数值
        inner_percentage = {}  # 存储百分比

        # 尝试多种嵌套格式
        sub_ags = bucket.get('sub_aggregations', {})
        inner_buckets = sub_ags.get('buckets', []) if sub_ags else []

        if not inner_buckets:
            dims = bucket.get('dimensions', {})
            if dims and dimensions:
                inner_buckets = dims.get(dimensions[0], {}).get('buckets', [])

        # 处理三层嵌套（展平）
        if not inner_buckets:
            for sub_bucket in sub_ags.get('buckets', []):
                sub_sub_ags = sub_bucket.get('sub_aggregations', {})
                if sub_sub_ags and 'buckets' in sub_sub_ags:
                    for sub_sub_bucket in sub_sub_ags['buckets']:
                        inner_key = f"{sub_bucket.get('key', '')}_{sub_sub_bucket.get('key', '')}"

                        # 获取主数值
                        if primary_metric in sub_sub_bucket:
                            main_val = sub_sub_bucket[primary_metric]
                        elif 'metrics' in sub_sub_bucket and primary_metric in sub_sub_bucket['metrics']:
                            main_val = sub_sub_bucket['metrics'][primary_metric]
                        else:
                            main_val = sub_sub_bucket.get('doc_count', 0)
                        inner_data[inner_key] = main_val

                        # 获取百分比
                        pct_val = None
                        if percentage_metric and percentage_metric in sub_sub_bucket:
                            pct_val = sub_sub_bucket[percentage_metric]
                        elif 'metrics' in sub_sub_bucket and percentage_metric and percentage_metric in sub_sub_bucket[
                            'metrics']:
                            pct_val = sub_sub_bucket['metrics'][percentage_metric]
                        inner_percentage[inner_key] = pct_val

                        all_inner_keys.add(inner_key)

        # 标准两层嵌套处理
        for inner_bucket in inner_buckets:
            inner_key = str(inner_bucket.get('key', ''))

            # 获取主数值
            if primary_metric in inner_bucket:
                main_val = inner_bucket[primary_metric]
            elif 'metrics' in inner_bucket and primary_metric in inner_bucket['metrics']:
                main_val = inner_bucket['metrics'][primary_metric]
            else:
                main_val = inner_bucket.get('doc_count', 0)
            inner_data[inner_key] = main_val

            # 获取百分比
            pct_val = None
            if percentage_metric and percentage_metric in inner_bucket:
                pct_val = inner_bucket[percentage_metric]
            elif 'metrics' in inner_bucket and percentage_metric and percentage_metric in inner_bucket['metrics']:
                pct_val = inner_bucket['metrics'][percentage_metric]
            inner_percentage[inner_key] = pct_val

            all_inner_keys.add(inner_key)

        bucket_data_list.append({
            'outer_key': outer_key,  # 现在作为列标签
            'inner_data': inner_data,
            'inner_percentage': inner_percentage,
            'total_count': bucket.get('doc_count', 0)
        })

    # 统一行顺序（内层键作为行）
    all_inner_keys = sorted(list(all_inner_keys))
    # 列顺序（外层桶keys）
    outer_keys = [b['outer_key'] for b in bucket_data_list]

    # 构建矩阵：行=内层桶，列=外层桶（转置）
    matrix = []
    annot_matrix = []  # 注释矩阵，用于显示"数值 (百分比)"

    # 对每个内层键（行），收集所有外层桶（列）的数据
    for inner_key in all_inner_keys:
        row_values = []
        row_annotations = []

        for bucket_data in bucket_data_list:
            main_val = bucket_data['inner_data'].get(inner_key, 0)
            pct_val = bucket_data['inner_percentage'].get(inner_key)

            row_values.append(main_val)

            # 构建注释字符串
            if pct_val is not None:
                # 根据数值类型决定格式
                if isinstance(pct_val, (int, float)):
                    # 如果百分比值大于1，假设它已经是百分比形式（如6.5表示6.5%）
                    # 如果小于1，假设它是小数形式（如0.065表示6.5%）
                    if pct_val <= 1:
                        display_pct = pct_val * 100
                    else:
                        display_pct = pct_val
                    annot_str = f'{int(main_val)}\n({display_pct:.2f}%)'
                else:
                    annot_str = f'{int(main_val)}\n({pct_val}%)'
            else:
                annot_str = str(int(main_val))

            row_annotations.append(annot_str)

        matrix.append(row_values)
        annot_matrix.append(row_annotations)

    # DataFrame：index=内层键（行/Y轴），columns=外层桶（列/X轴）
    df_pivot = pd.DataFrame(matrix, index=all_inner_keys, columns=outer_keys)

    # 使用自定义注释矩阵，fmt='' 表示使用原始字符串格式
    sns.heatmap(df_pivot, annot=annot_matrix, fmt='',
                cmap='YlOrRd', ax=ax, cbar_kws={'label': primary_metric.capitalize()},
                linewidths=0.5, linecolor='gray')

    # 设置标题
    ax.set_title(metadata.get('title', 'Heatmap Analysis'), fontsize=14, fontweight='bold')

    # 设置轴标签：优先使用metadata，其次从query_config推断
    # X轴标签：外层桶的field（如buckets中的field）或dimensions[1]（如果存在）
    xlabel = metadata.get('xlabel')
    if not xlabel and buckets_config and len(buckets_config) > 0:
        xlabel = buckets_config[0].get('field', '')
    if not xlabel and len(dimensions) > 1:
        xlabel = dimensions[1]
    elif not xlabel and len(dimensions) > 0:
        xlabel = dimensions[0]
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)

    # Y轴标签：优先使用dimensions[0]（分析目标），其次是metrics_field，然后是groups
    ylabel = metadata.get('ylabel')
    if not ylabel:
        if len(dimensions) > 0:
            ylabel = dimensions[0]
        elif metrics_field:
            ylabel = metrics_field
        elif len(groups) > 0:
            ylabel = groups[0]
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath
def plot_histogram(ax, df: pd.DataFrame, metadata: Dict[str, str], primary_metric: str = 'doc_count'):
    """Plot histogram with axis labels from metadata"""
    if 'level' in df.columns:
        df = df[df['level'] == 0].copy()

    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return

    labels = df['key'].astype(str).values
    values = df[primary_metric].values if primary_metric in df.columns else df['doc_count'].values

    ax.bar(range(len(labels)), values, color='coral', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 设置横纵坐标标签
    if metadata.get('xlabel'):
        ax.set_xlabel(metadata['xlabel'], fontsize=12)
    else:
        ax.set_xlabel('Range', fontsize=12)

    if metadata.get('ylabel'):
        ax.set_ylabel(metadata['ylabel'], fontsize=12)
    else:
        ax.set_ylabel('Count', fontsize=12)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=9)


def plot_stacked_histogram(result_json: Dict[str, Any], query_config: Dict[str, Any],
                           metadata: Dict[str, str], save_dir: str) -> str:
    """
    通用半透明叠加直方图，支持Y轴截断+断裂标记以处理组内极端值
    """
    buckets = result_json.get('buckets', [])
    if not buckets:
        return None

    dimensions = query_config.get('dimensions', [])
    metrics = query_config.get('metrics', ['count'])
    primary_metric = metrics[0] if metrics else 'doc_count'

    # 数据解析
    outer_keys = []
    inner_keys_set = set()
    data_matrix = {}

    for bucket in buckets:
        outer_key = str(bucket.get('key', ''))
        outer_keys.append(outer_key)
        data_matrix[outer_key] = {}

        sub_ags = bucket.get('sub_aggregations', {})
        inner_buckets = sub_ags.get('buckets', []) if sub_ags else []

        if not inner_buckets and dimensions:
            dims = bucket.get('dimensions', {})
            if dims:
                inner_buckets = dims.get(dimensions[0], {}).get('buckets', [])

        if not inner_buckets:
            groups = bucket.get('groups', {})
            for group_name, group_data in groups.items():
                inner_buckets.extend(group_data.get('buckets', []))

        for inner_bucket in inner_buckets:
            inner_key = str(inner_bucket.get('key', ''))
            inner_keys_set.add(inner_key)

            if primary_metric in inner_bucket:
                value = inner_bucket[primary_metric]
            elif 'metrics' in inner_bucket and primary_metric in inner_bucket['metrics']:
                value = inner_bucket['metrics'][primary_metric]
            else:
                value = inner_bucket.get('doc_count', 0)

            data_matrix[outer_key][inner_key] = value

    if not inner_keys_set:
        fig, ax = plt.subplots(figsize=(12, 7))
        values = [sum(data_matrix[k].values()) if data_matrix[k] else 0 for k in outer_keys]

        bars = ax.bar(range(len(outer_keys)), values, alpha=0.8, edgecolor='black', color='steelblue')
        ax.set_xticks(range(len(outer_keys)))
        ax.set_xticklabels(outer_keys, rotation=45, ha='right')

        if metadata.get('xlabel'):
            ax.set_xlabel(metadata['xlabel'], fontsize=12)
        if metadata.get('ylabel'):
            ax.set_ylabel(metadata['ylabel'], fontsize=12)
        ax.set_title(metadata.get('title', 'Histogram'), fontsize=14, fontweight='bold')

        plt.tight_layout()
        filename = generate_random_filename(8)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath

    inner_keys = sorted(list(inner_keys_set))
    colors = sns.color_palette("Set1", len(inner_keys))

    # 关键修改：检查组内差异而非全局差异
    # 计算每个组内的最大/次大比值，取最小值作为判断依据
    # 如果任一组内存在极端差异，就启用截断
    group_ratios = []
    group_second_maxs = []

    for outer_key in outer_keys:
        group_values = sorted([data_matrix[outer_key].get(k, 0) for k in inner_keys], reverse=True)
        if len(group_values) >= 2 and group_values[1] > 0:
            ratio = group_values[0] / group_values[1]
            group_ratios.append(ratio)
            group_second_maxs.append(group_values[1])

    # 使用最小的比值（最极端的情况）作为判断标准
    # 或者使用所有组中第二大的最大值作为截断参考
    BREAK_THRESHOLD = 3.0

    if group_ratios:
        min_ratio = min(group_ratios)
        # 截断点：基于所有组中第二大的最大值，留一些余量
        reference_second_max = max(group_second_maxs) if group_second_maxs else 100
    else:
        min_ratio = 1.0
        reference_second_max = 100

    # 触发条件：任一组内最大/次大超过阈值
    use_break = min_ratio > BREAK_THRESHOLD

    if use_break:
        # 截断点略高于参考次大值
        break_point = reference_second_max * 1.3
        return _plot_broken_axis_histogram(outer_keys, inner_keys, data_matrix, colors,
                                           metadata, save_dir, primary_metric, break_point)
    else:
        return _plot_standard_overlay(outer_keys, inner_keys, data_matrix, colors,
                                      metadata, save_dir, primary_metric)


def _plot_broken_axis_histogram(outer_keys, inner_keys, data_matrix, colors,
                                metadata, save_dir, primary_metric, break_point):
    """
    截断Y轴叠加直方图：通过断裂标记处理组内极端值
    """
    # 计算全局最大值用于顶部子图范围
    all_values = []
    for outer_key in outer_keys:
        for inner_key in inner_keys:
            all_values.append(data_matrix[outer_key].get(inner_key, 0))
    global_max = max(all_values) if all_values else break_point * 2
    top_margin = global_max * 1.05

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True,
                                            figsize=(14, 10),
                                            gridspec_kw={'height_ratios': [1, 3]})

    x_pos = np.arange(len(outer_keys))
    bar_width = 0.6

    # 在两个子图中绘制相同的柱子
    for i, inner_key in enumerate(inner_keys):
        values = [data_matrix[outer_key].get(inner_key, 0) for outer_key in outer_keys]

        # 顶部子图（显示极端值顶部）
        bars_top = ax_top.bar(x_pos, values, bar_width,
                              color=colors[i], alpha=0.6,
                              edgecolor='black', linewidth=1)

        # 底部子图（显示主要数据）
        bars_bottom = ax_bottom.bar(x_pos, values, bar_width,
                                    label=str(inner_key),
                                    color=colors[i], alpha=0.6,
                                    edgecolor='black', linewidth=1)

        # 数值标签智能分配
        if len(outer_keys) <= 12:
            for j, (bt, bb, val) in enumerate(zip(bars_top, bars_bottom, values)):
                if val > 0:
                    if val > break_point:
                        # 大值在顶部子图标注（显示完整数值）
                        ax_top.text(bt.get_x() + bt.get_width() / 2., val,
                                    f'{int(val)}', ha='center', va='bottom',
                                    fontsize=8, color=colors[i], fontweight='bold')
                    else:
                        # 小值在底部子图标注
                        y_pos = val / 2 if val > break_point * 0.2 else val
                        va = 'center' if val > break_point * 0.2 else 'bottom'
                        fontcolor = 'white' if val > break_point * 0.2 else colors[i]
                        ax_bottom.text(bb.get_x() + bb.get_width() / 2., y_pos,
                                       f'{int(val)}', ha='center', va=va,
                                       fontsize=8, color=fontcolor, fontweight='bold')

    # 设置顶部子图Y轴范围（断裂的上部，只显示超过break_point的部分）
    ax_top.set_ylim(break_point, top_margin)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 设置底部子图Y轴范围（主要数据区域，0到截断点）
    ax_bottom.set_ylim(0, break_point)
    ax_bottom.spines['top'].set_visible(False)

    # 添加断裂标记（斜线）
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_top.plot((-d, +d), (-d * 3, +d * 3), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 3, +d * 3), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # 设置X轴
    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels(outer_keys, rotation=45, ha='right')

    if metadata.get('xlabel'):
        ax_bottom.set_xlabel(metadata['xlabel'], fontsize=12)

    # Y轴标签
    ylabel = metadata.get('ylabel') or primary_metric.capitalize()
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)

    # 标题
    ax_top.set_title(metadata.get('title', 'Broken-Axis Overlay Histogram'),
                     fontsize=14, fontweight='bold', pad=20)

    # 图例
    dimensions = metadata.get('dimensions', [])
    legend_title = dimensions[0] if dimensions else 'Category'
    ax_bottom.legend(title=legend_title, loc='upper right',
                     framealpha=0.9, fancybox=True, shadow=True)

    # 网格线
    ax_bottom.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_bottom.set_axisbelow(True)
    ax_top.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_top.set_axisbelow(True)

    # 断裂说明
    ax_top.text(0.98, 0.02, f'Break: >{int(break_point)}',
                transform=ax_top.transAxes, fontsize=9, color='gray',
                ha='right', va='bottom', style='italic')

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath


def _plot_standard_overlay(outer_keys, inner_keys, data_matrix, colors,
                           metadata, save_dir, primary_metric):
    """标准半透明叠加直方图（无截断）"""
    x_pos = np.arange(len(outer_keys))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, inner_key in enumerate(inner_keys):
        values = [data_matrix[outer_key].get(inner_key, 0) for outer_key in outer_keys]

        bars = ax.bar(x_pos, values, bar_width,
                      label=str(inner_key),
                      color=colors[i],
                      alpha=0.5,
                      edgecolor='black',
                      linewidth=0.8,
                      zorder=i)

        if len(outer_keys) <= 12:
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    height = bar.get_height()
                    if height < max(values) * 0.3 if max(values) > 0 else False:
                        y_pos = height
                        va = 'bottom'
                        fontcolor = colors[i]
                        fontweight = 'normal'
                    else:
                        y_pos = height / 2
                        va = 'center'
                        fontcolor = 'white'
                        fontweight = 'bold'

                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            f'{int(val)}', ha='center', va=va,
                            fontsize=8, color=fontcolor, fontweight=fontweight)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(outer_keys, rotation=45, ha='right')

    if metadata.get('xlabel'):
        ax.set_xlabel(metadata['xlabel'], fontsize=12)
    if metadata.get('ylabel'):
        ax.set_ylabel(metadata['ylabel'], fontsize=12)
    else:
        ax.set_ylabel(primary_metric.capitalize(), fontsize=12)

    ax.set_title(metadata.get('title', 'Overlay Histogram'), fontsize=14, fontweight='bold', pad=20)

    dimensions = metadata.get('dimensions', [])
    legend_title = dimensions[0] if dimensions else 'Category'
    ax.legend(title=legend_title, loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath

def plot_stats_chart(ax, result_json: Dict[str, Any], query_config: Dict[str, Any],
                     metadata: Dict[str, str], chart_type: str):
    """Handle stats type visualizations"""
    fields = query_config.get('fields', [])
    metrics = query_config.get('metrics', ['min', 'max', 'avg', 'count'])

    if chart_type == 'histogram':
        # For histogram: show metrics distribution across fields
        field_labels = []
        metric_data = {m: [] for m in metrics}

        for field in fields:
            if field in result_json:
                field_labels.append(field)
                for metric in metrics:
                    val = result_json[field].get(metric, 0)
                    metric_data[metric].append(val)

        x = np.arange(len(field_labels))
        width = 0.8 / len(metrics) if metrics else 0.8

        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2) * width + width / 2
            ax.bar(x + offset, metric_data[metric], width, label=metric, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(field_labels, rotation=45, ha='right')
        ax.legend()
        ax.set_title(metadata['title'], fontsize=14, fontweight='bold')

    elif chart_type == 'bar':
        # Bar chart comparing specific metric across fields, or all metrics for one field
        if len(fields) > 1:
            # Compare first metric across all fields
            metric = metrics[0] if metrics else 'avg'
            values = [result_json.get(f, {}).get(metric, 0) for f in fields]
            colors = sns.color_palette("husl", len(fields))
            ax.bar(fields, values, color=colors, alpha=0.8)
            ax.set_ylabel(f'{metric.capitalize()} Value')
        else:
            # Show all metrics for single field
            field = fields[0]
            values = [result_json.get(field, {}).get(m, 0) for m in metrics]
            ax.bar(metrics, values, color='steelblue', alpha=0.8)
            ax.set_ylabel('Value')
        ax.set_title(metadata['title'], fontsize=14, fontweight='bold')

    elif chart_type == 'heatmap':
        # Stats类型的热力图（如果有矩阵数据）
        ax.text(0.5, 0.5, 'Heatmap for stats type not implemented\n(Use distribution type for heatmap)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(metadata['title'], fontsize=14, fontweight='bold')


def visualize_opensearch_result(query_json: Dict[str, Any],
                                result_json: Dict[str, Any],
                                save_dir: str = "./tmp_imgs/") -> str:
    """Main function with metadata passing to sub-functions"""
    ensure_directory(save_dir)
    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)

    # 提取元数据（标题、横纵坐标标签）
    metadata = extract_chart_metadata(query_json, result_json)

    query = query_json.get('query', {})
    query_type = query.get('type', '')
    chart_type = query.get('chart_type', 'bar')
    config = query.get('config', {})

    # Stats type
    if query_type == 'stats':
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            plot_stats_chart(ax, result_json, config, metadata, chart_type)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            plt.close()
            raise e
        return filepath

    # Distribution type
    df = parse_distribution_buckets(result_json)

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(metadata['title'])
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath

    # 关键修复：传递 chart_type 给 determine_chart_strategy
    strategy = determine_chart_strategy(df, config, chart_type)

    # 如果是明确的热力图请求，强制使用heatmap策略
    if chart_type == 'heatmap':
        strategy = 'heatmap'

    try:
        # 新增：叠加直方图分支
        if strategy == 'stacked_histogram':
            filepath = plot_stacked_histogram(result_json, config, metadata, save_dir)
            return filepath
        elif strategy == 'heatmap':
            filepath = plot_heatmap(result_json, config, metadata, save_dir)
            return filepath
        elif strategy == 'grouped_bar':
            filepath = plot_grouped_bar(result_json, config, metadata, save_dir)
            return filepath
        else:
            fig, ax = plt.subplots(figsize=(12, 7))

            metrics = config.get('metrics', ['count', 'percentage'])
            primary_metric = metrics[0] if metrics else 'doc_count'
            if primary_metric == 'count':
                primary_metric = 'doc_count'

            has_percentage = 'percentage' in df.columns or any(
                'percentage' in str(row) for row in df.to_dict('records'))

            if strategy == 'histogram':
                plot_histogram(ax, df, metadata, primary_metric)
            elif strategy == 'bar_with_pie' or (strategy == 'single_bar' and has_percentage):
                plot_single_bar(ax, df, metadata, primary_metric, show_pie=True)
            else:
                plot_single_bar(ax, df, metadata, primary_metric, show_pie=False)

            # 设置标题和轴标签
            ax.set_title(metadata['title'], fontsize=14, fontweight='bold', pad=20)
            if metadata.get('xlabel'):
                ax.set_xlabel(metadata['xlabel'], fontsize=12)
            if metadata.get('ylabel'):
                ax.set_ylabel(metadata['ylabel'], fontsize=12)

            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

    except Exception as e:
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', wrap=True)
        ax.set_title('Error Processing Visualization')
        ax.axis('off')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    return filepath


if __name__ == "__main__":
    # 测试AMD年龄分布热力图
    test_query = {
        "query": {
            "type": "distribution",
            "chart_type": "heatmap",
            "config": {
                "dimensions": ["patient_age", "amd"],
                "buckets": [
                    {
                        "type": "range",
                        "field": "patient_age",
                        "ranges": [
                            {"key": "0-30", "from": 0, "to": 30},
                            {"key": "30-50", "from": 30, "to": 50},
                            {"key": "50-70", "from": 50, "to": 70},
                            {"key": "70+", "from": 70}
                        ]
                    }
                ],
                "metrics": ["count", "percentage"],
                "filters": []
            }
        }
    }

    test_result = {
        "buckets": [
            {
                "key": "0-30",
                "doc_count": 1058,
                "from": 0.0,
                "to": 30.0,
                "metrics": {"count": 1058, "percentage": 6.5},
                "sub_aggregations": {
                    "buckets": [
                        {"key": 0, "doc_count": 1058, "metrics": {"count": 1058, "percentage": 100.0}}
                    ]
                }
            },
            {
                "key": "30-50",
                "doc_count": 2058,
                "from": 30.0,
                "to": 50.0,
                "metrics": {"count": 2058, "percentage": 12.65},
                "sub_aggregations": {
                    "buckets": [
                        {"key": 0, "doc_count": 2058, "metrics": {"count": 2058, "percentage": 100.0}}
                    ]
                }
            },
            {
                "key": "50-70",
                "doc_count": 4573,
                "from": 50.0,
                "to": 70.0,
                "metrics": {"count": 4573, "percentage": 28.11},
                "sub_aggregations": {
                    "buckets": [
                        {"key": 0, "doc_count": 4538, "metrics": {"count": 4538, "percentage": 99.23}},
                        {"key": 1, "doc_count": 35, "metrics": {"count": 35, "percentage": 0.77}}
                    ]
                }
            },
            {
                "key": "70+",
                "doc_count": 3131,
                "from": 70.0,
                "to": None,
                "metrics": {"count": 3131, "percentage": 19.25},
                "sub_aggregations": {
                    "buckets": [
                        {"key": 0, "doc_count": 2932, "metrics": {"count": 2932, "percentage": 93.64}},
                        {"key": 1, "doc_count": 199, "metrics": {"count": 199, "percentage": 6.36}}
                    ]
                }
            }
        ]
    }

    path = visualize_opensearch_result(test_query, test_result)
    print(f"Saved to: {path}")