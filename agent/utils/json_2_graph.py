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
            if len(groups) >= 2:
                metadata['xlabel'] = groups[0]
                metadata['ylabel'] = groups[1]
            elif len(dimensions) >= 2:
                metadata['xlabel'] = dimensions[0]
                metadata['ylabel'] = dimensions[1]

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


def determine_chart_strategy(df: pd.DataFrame, query_config: Dict[str, Any]) -> str:
    """
    Determine the best chart strategy based on data structure.

    Returns:
        'single_bar': single layer bar chart
        'histogram': range-based buckets
        'bar_with_pie': single layer with percentage metrics
        'grouped_bar': multiple outer groups, need subplots
    """
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


def plot_stats_chart(ax, result_json: Dict[str, Any], query_config: Dict[str, Any], chart_type: str):
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
            ax.set_title(metadata['title'], fontsize=14, fontweight='bold', pad=20)
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

    strategy = determine_chart_strategy(df, config)
    buckets = result_json.get('buckets', [])
    if buckets and any(is_range_bucket(b) for b in buckets):
        strategy = 'histogram'

    try:
        if strategy == 'grouped_bar':
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
    # Test with your provided data
    test_query = {
        "query": {
            "type": "distribution",
            "chart_type": "bar",
            "config": {
                "dimensions": ["exam_eye", "Illuminaton"],
                "metrics": ["count", "percentage"],
                "groups": ["hemorrhage"],
                "filters": [{"field": "hemorrhage", "operator": "eq", "value": 1}]
            }
        }
    }

    test_result = {
        "buckets": [
            {
                "key": 1,
                "doc_count": 95,
                "metrics": {"count": 95, "percentage": 100.0},
                "sub_aggregations": {
                    "buckets": [
                        {
                            "key": "1",
                            "doc_count": 48,
                            "metrics": {"count": 48, "percentage": 50.53},
                            "sub_aggregations": {
                                "buckets": [
                                    {"key": "1", "doc_count": 48, "metrics": {"count": 48, "percentage": 100.0}}]
                            }
                        },
                        {
                            "key": "2",
                            "doc_count": 47,
                            "metrics": {"count": 47, "percentage": 49.47},
                            "sub_aggregations": {
                                "buckets": [
                                    {"key": "1", "doc_count": 47, "metrics": {"count": 47, "percentage": 100.0}}]
                            }
                        }
                    ]
                }
            }
        ]
    }

    path = visualize_opensearch_result(test_query, test_result)
    print(f"Saved to: {path}")