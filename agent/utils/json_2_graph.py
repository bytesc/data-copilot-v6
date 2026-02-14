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
            metadata['title'] = f"Distribution of {dimensions[0] if dimensions else 'Categories'}"
        elif chart_type == 'heatmap':
            # 热力图：x轴是dimension（AMD状态），y轴是bucket（年龄组）
            if dimensions:
                metadata['xlabel'] = dimensions[0]  # amd
            metadata['ylabel'] = 'Age Groups'
            metadata['title'] = f"Risk of {dimensions[0].upper() if dimensions else 'Disease'} by Age Group"

    metadata['title'] = ' | '.join(title_parts) if title_parts else 'OpenSearch Analysis Visualization'

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

    if df.empty:
        return 'single_bar'

    # Check for histogram/range data
    if 'from' in df.columns or 'to' in df.columns:
        return 'histogram'

    # Check if single layer (no parent keys)
    parent_cols = [c for c in df.columns if c.startswith('level_') or c.startswith('group_')]
    if not parent_cols:
        if 'percentage' in df.columns or 'metrics' in str(df.to_dict()):
            return 'bar_with_pie'
        return 'single_bar'

    # Check outer buckets count
    if 'level_0_group' in df.columns:
        outer_groups = df['level_0_group'].nunique()
        if outer_groups > 1:
            return 'grouped_bar'
        else:
            return 'single_bar'

    return 'single_bar'


def plot_single_bar(ax, df: pd.DataFrame, metadata: Dict[str, str], primary_metric: str = 'doc_count',
                    show_pie: bool = False):
    """Plot simple bar chart with axis labels from metadata"""
    if primary_metric not in df.columns:
        primary_metric = 'doc_count'

    values = df[primary_metric].values
    keys = df['key'].astype(str).values

    if len(keys) > 15:
        sorted_idx = np.argsort(values)[::-1]
        top_n = 14
        top_idx = sorted_idx[:top_n]
        other_sum = values[sorted_idx[top_n:]].sum()

        keys = list(keys[top_idx]) + ['Others']
        values = list(values[top_idx]) + [other_sum]

    colors = sns.color_palette("husl", len(keys))

    if show_pie and len(keys) <= 10:
        ax.set_visible(False)
        ax1 = ax.figure.add_subplot(121)
        ax2 = ax.figure.add_subplot(122)

        bars = ax1.bar(keys, values, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels(keys, rotation=45, ha='right')

        if metadata.get('xlabel'):
            ax1.set_xlabel(metadata['xlabel'], fontsize=11)
        if metadata.get('ylabel'):
            ax1.set_ylabel(metadata['ylabel'], fontsize=11)
        else:
            ax1.set_ylabel(primary_metric.capitalize(), fontsize=11)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        wedges, texts, autotexts = ax2.pie(values, labels=keys, autopct='%1.1f%%',
                                           startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title(f'{metadata.get("title", "Distribution")} (Pie View)', fontsize=11)

    else:
        bars = ax.bar(keys, values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')

        if metadata.get('xlabel'):
            ax.set_xlabel(metadata['xlabel'], fontsize=12)
        if metadata.get('ylabel'):
            ax.set_ylabel(metadata['ylabel'], fontsize=12)
        else:
            ax.set_ylabel(primary_metric.capitalize(), fontsize=12)

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

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

            ax.set_title(f'{outer_key}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')

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

    fig.suptitle(metadata.get('title', 'OpenSearch Analysis'), fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath


def plot_heatmap(result_json: Dict[str, Any], query_config: Dict[str, Any],
                 metadata: Dict[str, str], save_dir: str) -> str:
    """
    绘制热力图：支持两种数据结构
    1. 三层嵌套: camera × focus × artifacts (原有)
    2. 两层嵌套: age_group × amd_status (新增，用于AMD风险分析)
    """
    buckets = result_json.get('buckets', [])
    if not buckets:
        return None

    dimensions = query_config.get('dimensions', [])
    metrics = query_config.get('metrics', ['count', 'percentage'])
    primary_metric = metrics[0] if metrics else 'count'

    # 判断数据结构类型
    # 检查第一层bucket是否有range特征（年龄分组）
    is_age_structure = any(is_range_bucket(b) for b in buckets)

    if is_age_structure:
        # 结构: 年龄组 × AMD状态 (两层嵌套)
        return plot_heatmap_age_amd(buckets, dimensions, primary_metric, metadata, save_dir)
    else:
        # 结构: Camera × Focus × Artifacts (三层嵌套)
        return plot_heatmap_camera_focus_artifacts(buckets, query_config, dimensions,
                                                   primary_metric, metadata, save_dir)


def plot_heatmap_age_amd(buckets: List[Dict], dimensions: List[str],
                         primary_metric: str, metadata: Dict[str, str],
                         save_dir: str) -> str:
    """
    绘制年龄组 × AMD状态 的热力图
    显示每个年龄组的AMD风险百分比
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 准备数据：行=年龄组，列=AMD状态(0/1)
    age_groups = []
    amd_labels = ['No AMD', 'AMD']  # 假设0=无，1=有
    data_matrix = []

    for bucket in buckets:
        age_key = bucket.get('key', '')
        age_groups.append(age_key)

        # 获取该年龄组的AMD分布
        sub_ags = bucket.get('sub_aggregations', {})
        amd_buckets = sub_ags.get('buckets', [])

        row_data = {}
        total_count = bucket.get('doc_count', 0)

        for amd_bucket in amd_buckets:
            amd_status = str(amd_bucket.get('key', ''))

            # 获取百分比（优先使用内层percentage，表示该AMD状态在年龄组内的占比）
            if 'metrics' in amd_bucket and 'percentage' in amd_bucket['metrics']:
                value = amd_bucket['metrics']['percentage']
            elif 'percentage' in amd_bucket:
                value = amd_bucket['percentage']
            elif primary_metric in amd_bucket:
                value = amd_bucket[primary_metric]
            elif 'metrics' in amd_bucket and primary_metric in amd_bucket['metrics']:
                value = amd_bucket['metrics'][primary_metric]
            else:
                value = amd_bucket.get('doc_count', 0)

            row_data[amd_status] = value

        # 确保有两列数据（0和1）
        row_values = [row_data.get('0', 0), row_data.get('1', 0)]
        data_matrix.append(row_values)

    # 创建DataFrame
    df_pivot = pd.DataFrame(data_matrix,
                            index=age_groups,
                            columns=['No AMD (0)', 'AMD (1)'])

    # 绘制热力图
    sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                ax=ax, cbar_kws={'label': 'Percentage (%)'},
                linewidths=1, linecolor='white', vmin=0, vmax=100)

    # 设置标签
    ax.set_title(metadata.get('title', 'AMD Risk by Age Group'),
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(metadata.get('xlabel', 'AMD Status'), fontsize=12)
    ax.set_ylabel(metadata.get('ylabel', 'Age Groups'), fontsize=12)

    # 旋转标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    filename = generate_random_filename(8)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return filepath


def plot_heatmap_camera_focus_artifacts(buckets: List[Dict], query_config: Dict[str, Any],
                                        dimensions: List[str], primary_metric: str,
                                        metadata: Dict[str, str], save_dir: str) -> str:
    """
    原有的三层嵌套热力图：Camera × Focus × Artifacts
    """
    group_field = query_config.get('groups', ['camera'])[0] if query_config.get('groups') else 'camera'

    n_plots = len(buckets)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, bucket in enumerate(buckets):
        ax = axes[idx] if n_plots > 1 else axes[0]
        camera_model = bucket.get('key', f'Camera {idx}')

        pivot_data = {}
        focus_labels = set()
        artifact_labels = set()

        sub_ags = bucket.get('sub_aggregations', {})
        focus_buckets = sub_ags.get('buckets', [])

        for focus_bucket in focus_buckets:
            focus_key = str(focus_bucket.get('key', ''))
            focus_labels.add(focus_key)

            sub_sub_ags = focus_bucket.get('sub_aggregations', {})
            artifact_buckets = sub_sub_ags.get('buckets', [])

            for artifact_bucket in artifact_buckets:
                artifact_key = str(artifact_bucket.get('key', ''))
                artifact_labels.add(artifact_key)

                if primary_metric in artifact_bucket:
                    value = artifact_bucket[primary_metric]
                elif 'metrics' in artifact_bucket and primary_metric in artifact_bucket['metrics']:
                    value = artifact_bucket['metrics'][primary_metric]
                else:
                    value = artifact_bucket.get('doc_count', 0)

                if focus_key not in pivot_data:
                    pivot_data[focus_key] = {}
                pivot_data[focus_key][artifact_key] = value

        if pivot_data:
            focus_labels = sorted(list(focus_labels))
            artifact_labels = sorted(list(artifact_labels))

            matrix = []
            for focus in focus_labels:
                row = []
                for artifact in artifact_labels:
                    row.append(pivot_data.get(focus, {}).get(artifact, 0))
                matrix.append(row)

            df_pivot = pd.DataFrame(matrix, index=focus_labels, columns=artifact_labels)

            sns.heatmap(df_pivot, annot=True, fmt='.0f' if primary_metric == 'count' else '.2f',
                        cmap='YlOrRd', ax=ax, cbar_kws={'label': primary_metric.capitalize()},
                        linewidths=0.5, linecolor='gray')

            ax.set_title(f'{camera_model}\n(n={bucket.get("doc_count", 0)})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel(dimensions[1] if len(dimensions) > 1 else 'Artifacts', fontsize=11)
            ax.set_ylabel(dimensions[0] if len(dimensions) > 0 else 'Focus', fontsize=11)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.text(0.5, 0.5, f'No data for {camera_model}',
                    ha='center', va='center', transform=ax.transAxes)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(metadata.get('title', f'Heatmap Analysis by {group_field.capitalize()}'),
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

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

    if metadata.get('xlabel'):
        ax.set_xlabel(metadata['xlabel'], fontsize=12)
    else:
        ax.set_xlabel('Range', fontsize=12)

    if metadata.get('ylabel'):
        ax.set_ylabel(metadata['ylabel'], fontsize=12)
    else:
        ax.set_ylabel('Count', fontsize=12)

    for i, v in enumerate(values):
        ax.text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=9)


def plot_stats_chart(ax, result_json: Dict[str, Any], query_config: Dict[str, Any],
                     metadata: Dict[str, str], chart_type: str):
    """Handle stats type visualizations"""
    fields = query_config.get('fields', [])
    metrics = query_config.get('metrics', ['min', 'max', 'avg', 'count'])

    if chart_type == 'histogram':
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
        if len(fields) > 1:
            metric = metrics[0] if metrics else 'avg'
            values = [result_json.get(f, {}).get(metric, 0) for f in fields]
            colors = sns.color_palette("husl", len(fields))
            ax.bar(fields, values, color=colors, alpha=0.8)
            ax.set_ylabel(f'{metric.capitalize()} Value')
        else:
            field = fields[0]
            values = [result_json.get(field, {}).get(m, 0) for m in metrics]
            ax.bar(metrics, values, color='steelblue', alpha=0.8)
            ax.set_ylabel('Value')
        ax.set_title(metadata['title'], fontsize=14, fontweight='bold')

    elif chart_type == 'heatmap':
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

    strategy = determine_chart_strategy(df, config, chart_type)

    # 如果是明确的热力图请求，强制使用heatmap策略
    if chart_type == 'heatmap':
        strategy = 'heatmap'

    try:
        if strategy == 'heatmap':
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
    # 测试您的AMD数据
    test_query = {
        "query": {
            "type": "distribution",
            "chart_type": "heatmap",
            "config": {
                "dimensions": ["amd"],
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