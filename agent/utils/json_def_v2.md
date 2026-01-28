# OpenSearch Statistical Analysis JSON Interface Specification

## Overview

This document defines the JSON interface format for OpenSearch statistical analysis. Through unified JSON configuration, users can perform complex data statistical analysis, including numerical statistical calculations and multidimensional distribution analysis, with flexible visualization options.

---

## 1. Basic Structure

### JSON Root Structure
```json
{
  "query": {
    "type": "string",        // Required: Query type
    "chart_type": "string",  // Optional: Visualization chart type
    "config": {              // Required: Query configuration
      "fields": [],
      "metrics": [],
      "dimensions": [],
      "groups": [],
      "buckets": [],
      "filters": []
    }
  }
}
```

### Field Definitions
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query.type` | string | Yes | Query type: `stats` (statistical calculation) or `distribution` (distribution analysis) |
| `query.chart_type` | string | Yes | Visualization chart type: `bar`, `histogram`, `pie`, `line`, `heatmap` |
| `query.config` | object | Yes | Query configuration object |
| `query.config.fields` | string[] | Conditional | Numerical fields to analyze (required for stats type) |
| `query.config.metrics` | string[] | Conditional | Statistical metrics to calculate |
| `query.config.dimensions` | string[] | Conditional | Dimension fields to analyze (required for distribution type) |
| `query.config.groups` | string[] | No | Grouping fields |
| `query.config.buckets` | object[] | No | Bucket configuration array |
| `query.config.filters` | object[] | No | Filter conditions array |

---

## 2. Query Types (query.type)

### 2.1 Available Types

| Type | Value | Description |
|------|-------|-------------|
| Statistical Calculation | `"stats"` | Statistical analysis of numerical fields |
| Distribution Analysis | `"distribution"` | Multidimensional distribution and grouping analysis |

### 2.2 Type Selection Guide
• Pure numerical statistics (averages, percentiles, standard deviation, etc.) → Use `stats`

• Distribution analysis (grouped statistics, frequency distribution, range bucketing, etc.) → Use `distribution`

---

## 3. Chart Types (query.chart_type)

### 3.1 Available Chart Types

| Chart Type | Value | Description | Recommended Data Structure |
|------------|-------|-------------|---------------------------|
| Bar Chart | `"bar"` | Vertical or horizontal bars for comparing categorical data | Categories vs. values |
| Histogram | `"histogram"` | Distribution of numerical data using continuous intervals | Range buckets with frequencies |
| Pie Chart | `"pie"` | Circular statistical graphic divided into slices to illustrate numerical proportion | Categories with percentages |
| Line Chart | `"line"` | Series of data points connected by straight line segments | Time series or sequential data |
| Heatmap | `"heatmap"` | Graphical representation of data where values are depicted by color intensity | Two-dimensional matrix with magnitude |

### 3.2 Chart Type Compatibility Matrix

| Chart Type | Compatible with `stats` | Compatible with `distribution` | Best Use Case |
|------------|------------------------|-------------------------------|---------------|
| `bar` | ✅ Limited | ✅ Primary | Comparing counts or metrics across categories |
| `histogram` | ✅ Yes | ✅ Yes | Displaying distribution of numerical ranges |
| `pie` | ❌ No | ✅ Yes | Showing proportional composition of whole |
| `line` | ✅ Yes | ✅ Yes | Trends over time or ordered categories |
| `heatmap` | ❌ No | ✅ Yes | Correlation between two categorical variables with intensity |

### 3.3 Chart Type Selection Guide

• **Bar Chart**: Use when comparing discrete categories or groups. Best for showing exact values and facilitating direct comparisons between different segments.

• **Histogram**: Use specifically for continuous numerical data distribution. Ideal for understanding data shape, central tendency, and spread across defined intervals.

• **Pie Chart**: Use when showing the proportional composition of a whole, particularly effective with 2-7 categories. Avoid when precise comparisons are needed.

• **Line Chart**: Use for time-series analysis or tracking changes over sequential intervals. Excellent for revealing trends, patterns, and anomalies over time.

• **Heatmap**: Use for visualizing the relationship between two categorical dimensions with color intensity representing a third quantitative metric. Ideal for identifying patterns in dense datasets.

---

## 4. Statistical Calculation Interface (stats)

### 4.1 Specific Configuration Parameters

```json
{
  "query": {
    "type": "stats",
    "chart_type": "histogram",
    "config": {
      "fields": ["age", "salary", "score"],  // Required: Numerical field list
      "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation"], // Statistical metrics
      "filters": []  // Optional: Filter conditions
    }
  }
}
```

### 4.2 Chart Type Recommendations for Stats

| Chart Type | Configuration Notes |
|------------|-------------------|
| `histogram` | Default choice for single-field distribution visualization |
| `line` | Use when comparing multiple fields or showing statistical trends |
| `bar` | Suitable for comparing summary statistics (avg, max, min) across fields |

### 4.3 Supported Statistical Metrics (metrics)

| Metric | Value | Description |
|--------|-------|-------------|
| Count | `"count"` | Number of documents |
| Minimum | `"min"` | Minimum value |
| Maximum | `"max"` | Maximum value |
| Average | `"avg"` | Average value |
| Sum | `"sum"` | Sum of values |
| Median | `"median"` | Median (50th percentile) |
| First Quartile | `"q1"` | First quartile (25th percentile) |
| Third Quartile | `"q3"` | Third quartile (75th percentile) |
| Standard Deviation | `"std_deviation"` | Standard deviation |
| Variance | `"variance"` | Variance |

### 4.4 Default Metrics
• If `metrics` is not specified, defaults to: `["min", "max", "avg", "count"]`

---

## 5. Distribution Analysis Interface (distribution)

### 5.1 Specific Configuration Parameters

```json
{
  "query": {
    "type": "distribution",
    "chart_type": "bar",
    "config": {
      "dimensions": ["education", "job_title"],  // Required: Dimension fields to analyze
      "groups": ["department", "region"],        // Optional: Grouping fields
      "buckets": [                               // Optional: Bucket configuration
        {
          "type": "range",                       // Range bucket
          "field": "age",
          "ranges": [
            {"key": "Young", "from": 0, "to": 30},
            {"key": "Middle", "from": 30, "to": 60},
            {"key": "Old", "from": 60}
          ]
        },
        {
          "type": "terms",                       // Terms bucket
          "field": "education"
        }
      ],
      "metrics": ["count", "percentage", "avg"], // Metrics to calculate
      "metrics_field": "salary",                 // Optional: Target field for metric calculations
      "filters": []                              // Optional: Filter conditions
    }
  }
}
```

> ⚠️  Caution  
> If a numeric field is listed in `groups` and has **no pre-defined range bucket**, the engine will create one group per distinct numeric value, quickly producing sparse or high-cardinality buckets.  
> To bucket by interval (e.g. age brackets 0-20, 20-40 …):  
> 1. define a `type: range` bucket on that field;  
> 2. aggregate on the **bucket key** (or nest an agg and re-group by key), instead of dropping the raw numeric field into `groups`.

### 5.2 Chart Type Recommendations for Distribution

| Chart Type | Best Configuration | Limitations |
|------------|-------------------|-------------|
| `bar` | Single dimension with count/percentage metrics | May become cluttered with >20 categories |
| `histogram` | Range buckets on numerical fields | Requires `range` type buckets |
| `pie` | Single dimension showing percentage composition | Maximum 7-10 slices recommended |
| `line` | Date histogram buckets or ordered ranges | X-axis must be ordered/sequential |
| `heatmap` | Two grouping fields/dimensions with intensity metric | Requires exactly 2 categorical variables |

### 5.3 Bucket Types (buckets.type)

#### 5.3.1 Terms Bucket (terms)
For distribution analysis of categorical fields

```json
{
  "type": "terms",
  "field": "education_level",    // Field to bucket
  "size": 10                     // Optional: Number of buckets to return, default 10
}
```

#### 5.3.2 Range Bucket (range)
For range grouping of numerical fields

```json
{
  "type": "range",
  "field": "age",                // Numerical field to bucket
  "ranges": [                   // Range definitions
    {
      "key": "Young",           // Bucket identifier
      "from": 0,               // Start value (inclusive)
      "to": 30                 // End value (exclusive)
    },
    {
      "key": "Middle",
      "from": 30,
      "to": 60
    },
    {
      "key": "Old", 
      "from": 60               // Only from means >=60
    }
  ]
}
```

> ⚠️ **Single-point range filtering**  
> When you only need to query a single age band (e.g. 60–70), **use a `filters` clause** instead of creating a one-interval `range` bucket.  
> Buckets require **at least two intervals** to produce valid aggregations; a single-interval bucket will return empty or misleading results.  
> Example – correct way to restrict the whole query to 60 ≤ age < 70:
> ```json
> "filters": [
>   {
>     "field": "age",
>     "operator": "range",
>     "value": { "gte": 60, "lt": 70 }
>   }
> ]
> ```

#### 5.3.3 Date Histogram Bucket (date_histogram)
For time-based bucketing analysis

```json
{
  "type": "date_histogram",
  "field": "create_time",       // Time field
  "interval": "1M",            // Time interval: 1d(day), 1w(week), 1M(month), 1y(year)
  "format": "yyyy-MM"          // Optional: Time format
}
```

### 5.4 Distribution Analysis Metrics (metrics)

| Metric | Value | Description | Notes |
|--------|-------|-------------|-------|
| Count | `"count"` | Document count | Always available |
| Percentage | `"percentage"` | Percentage within group | Requires grouping |
| Average | `"avg"` | Average value | Requires metrics_field |
| Sum | `"sum"` | Sum of values | Requires metrics_field |
| Minimum | `"min"` | Minimum value | Requires metrics_field |
| Maximum | `"max"` | Maximum value | Requires metrics_field |

### 5.5 Default Metrics
• If `metrics` is not specified, defaults to: `["count", "percentage"]`

---

## 6. Common Configuration Parameters

### 6.1 Filter Conditions (filters)

#### Definition
```json
"filters": [
  {
    "field": "string",      // Field name
    "operator": "string",   // Operator
    "value": "any"         // Value (type depends on operator)
  }
]
```

#### Supported Operators

| Operator | Value | Description | Value Type | Example |
|----------|-------|-------------|------------|---------|
| Equals | `"eq"` | Field equals specified value | Any | `"value": "male"` |
| Not equals | `"neq"` | Field does not equal specified value | Any | `"value": "female"` |
| Greater than | `"gt"` | Field greater than specified value | Number/Date | `"value": 18` |
| Greater than or equal | `"gte"` | Field greater than or equal to specified value | Number/Date | `"value": "2023-01-01"` |
| Less than | `"lt"` | Field less than specified value | Number/Date | `"value": 100` |
| Less than or equal | `"lte"` | Field less than or equal to specified value | Number/Date | `"value": "2023-12-31"` |
| In | `"in"` | Field value is in specified list | Array | `"value": ["A", "B", "C"]` |
| Range | `"range"` | Field is within specified range | Object | `"value": {"gte": 10, "lte": 20}` |
| Exists | `"exists"` | Field exists (not null) | None | No value needed |

---

## 7. Query Examples

### 7.1 Statistical Calculation Examples

#### Example 7.1.1: Numerical Statistics with Histogram
```json
{
  "query": {
    "type": "stats",
    "chart_type": "histogram",
    "config": {
      "fields": ["age", "salary", "work_years"],
      "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation"],
      "filters": [
        {
          "field": "department",
          "operator": "eq",
          "value": "engineering"
        },
        {
          "field": "hire_date", 
          "operator": "gte",
          "value": "2020-01-01"
        }
      ]
    }
  }
}
```

#### Example 7.1.2: Multi-field Comparison with Bar Chart
```json
{
  "query": {
    "type": "stats", 
    "chart_type": "bar",
    "config": {
      "fields": ["blood_pressure", "cholesterol", "heart_rate"],
      "metrics": ["min", "max", "avg", "median"],
      "filters": [
        {
          "field": "gender",
          "operator": "eq", 
          "value": "male"
        }
      ]
    }
  }
}
```

### 7.2 Distribution Analysis Examples

#### Example 7.2.1: Age Distribution Histogram
```json
{
  "query": {
    "type": "distribution",
    "chart_type": "histogram",
    "config": {
      "dimensions": ["diabetic_retinopathy"],
      "buckets": [{
        "type": "range",
        "field": "patient_age",
        "ranges": [
          {"key": "0-20", "from": 0, "to": 20},
          {"key": "20-40", "from": 20, "to": 40},
          {"key": "40-60", "from": 40, "to": 60},
          {"key": "60-80", "from": 60, "to": 80},
          {"key": "80+", "from": 80}
        ]
      }],
      "metrics": ["count", "percentage"],
      "filters": []
    }
  }
}
```

#### Example 7.2.2: Categorical Distribution with Pie Chart
```json
{
  "query": {
    "type": "distribution",
    "chart_type": "pie",
    "config": {
      "dimensions": ["education_level", "job_title"],
      "groups": ["department"],
      "metrics": ["count", "percentage"],
      "filters": [
        {
          "field": "active",
          "operator": "eq",
          "value": true
        }
      ]
    }
  }
}
```

#### Example 7.2.3: Multi-level Grouping with Bar Chart
```json
{
  "query": {
    "type": "distribution",
    "chart_type": "bar",
    "config": {
      "dimensions": ["performance_rating", "attendance_rate"],
      "groups": ["company", "department"],
      "buckets": [
        {
          "type": "range",
          "field": "age",
          "ranges": [
            {"key": "20-30", "from": 20, "to": 30},
            {"key": "30-40", "from": 30, "to": 40},
            {"key": "40-50", "from": 40, "to": 50},
            {"key": "50+", "from": 50}
          ]
        },
        {
          "type": "range", 
          "field": "salary",
          "ranges": [
            {"key": "Low", "from": 0, "to": 10000},
            {"key": "Medium", "from": 10000, "to": 30000},
            {"key": "High", "from": 30000}
          ]
        }
      ],
      "metrics": ["count", "percentage", "avg"],
      "metrics_field": "salary"
    }
  }
}
```

#### Example 7.2.4: Time Series Analysis with Line Chart
```json
{
  "query": {
    "type": "distribution", 
    "chart_type": "line",
    "config": {
      "dimensions": ["product_category", "sales_region"],
      "buckets": [
        {
          "type": "date_histogram",
          "field": "sales_date",
          "interval": "1M",
          "format": "yyyy-MM"
        }
      ],
      "metrics": ["count", "sum", "avg"],
      "metrics_field": "sales_amount",
      "filters": [
        {
          "field": "sales_date",
          "operator": "range",
          "value": {
            "gte": "2023-01-01",
            "lte": "2023-12-31"
          }
        }
      ]
    }
  }
}
```

#### Example 7.2.5: Correlation Analysis with Heatmap
```json
{
  "query": {
    "type": "distribution",
    "chart_type": "heatmap",
    "config": {
      "dimensions": ["has_disease", "treatment_type"],
      "groups": ["hospital", "department"],
      "buckets": [
        {
          "type": "range",
          "field": "age",
          "ranges": [
            {"key": "Child", "from": 0, "to": 12},
            {"key": "Teen", "from": 12, "to": 18},
            {"key": "Adult", "from": 18, "to": 60},
            {"key": "Senior", "from": 60}
          ]
        },
        {
          "type": "terms",
          "field": "gender",
          "size": 5
        }
      ],
      "metrics": ["count", "percentage"],
      "metrics_field": "treatment_cost",
      "filters": [
        {
          "field": "visit_date",
          "operator": "gte",
          "value": "2023-01-01"
        }
      ]
    }
  }
}
```

---

## 8. Response Format

### 8.1 Statistical Calculation Response
```json
{
  "age": {
    "count": 1000,
    "min": 18,
    "max": 80,
    "avg": 45.5,
    "sum": 45500,
    "q1": 30.0,
    "median": 45.0,
    "q3": 60.0,
    "std_deviation": 15.2,
    "variance": 231.04
  },
  "salary": {
    "count": 1000,
    "min": 3000,
    "max": 50000,
    "avg": 15000.5,
    "sum": 15000500,
    "q1": 8000.0,
    "median": 12000.0,
    "q3": 20000.0
  }
}
```

### 8.2 Distribution Analysis Response
```json
{
  "buckets": [
    {
      "key": "20-30",
      "from": 20,
      "to": 30,
      "doc_count": 250,
      "metrics": {
        "count": 250,
        "percentage": 25.0,
        "avg_salary": 12000.5
      },
      "dimensions": {
        "education_level": {
          "buckets": [
            {
              "key": "Bachelor",
              "doc_count": 150,
              "percentage": 60.0,
              "avg_salary": 12500.0
            },
            {
              "key": "Master", 
              "doc_count": 100,
              "percentage": 40.0,
              "avg_salary": 14000.0
            }
          ]
        }
      },
      "groups": {
        "department": {
          "buckets": [
            {
              "key": "Engineering",
              "doc_count": 120,
              "percentage": 48.0
            }
          ]
        }
      }
    }
  ]
}
```

---

## 9. Error Handling and Validation

### 9.1 Common Error Types
| Error Type | Cause | Solution |
|------------|-------|----------|
| Missing required field | Required parameter not provided | Check query structure |
| Invalid chart_type | Unsupported chart type specified | Use only supported values: `bar`, `histogram`, `pie`, `line`, `heatmap` |
| Chart type incompatible | Incompatible chart_type with query type | Refer to Section 3.2 Compatibility Matrix |
| Invalid operator | Unsupported operator used | Use only supported operators |
| Type mismatch | Value type doesn't match field type | Ensure value type compatibility |
| Empty result | No documents match filters | Broaden filter criteria |
| Timeout | Query too complex or data too large | Add more filters, reduce fields |

### 9.2 Validation Rules
1. **Chart Type Compatibility**: 
   - `pie` and `heatmap` are only compatible with `distribution` type
   - `stats` type limited to `histogram`, `bar`, and `line` only
   
2. **Field existence**: Fields must exist in the index mapping
   
3. **Type compatibility**: Operators must be compatible with field types
   
4. **Range validity**: Range `from` must be less than `to` (if both specified)
   
5. **Array limits**: Maximum 10 fields, 3 group levels recommended

---

## 10. Performance Considerations

### 10.1 Optimization Guidelines
1. Indexing Strategy:
   • Ensure numerical fields are indexed as appropriate types
   • Use keyword type for categorical fields used in grouping
   • Create composite indices for frequently queried combinations

2. Query Design:
   • Always include relevant filters to reduce dataset size
   • Use range queries instead of multiple OR conditions
   • Limit the number of aggregation levels
   • Set appropriate size limits for terms aggregations
   • Select appropriate `chart_type` to minimize client-side processing

3. Resource Management:
   • Monitor aggregation memory usage
   • Use pagination for large result sets
   • Consider time-based partitioning for time-series data
   • Heatmap calculations may require additional computation resources

### 10.2 Recommended Limits
| Parameter | Recommended Limit | Hard Limit |
|-----------|------------------|------------|
| Fields per query | 5-10 | 20 |
| Group levels | 2-3 | 5 |
| Filters | 5-10 | 20 |
| Buckets per aggregation | 100-1000 | 10000 |
| Heatmap dimensions | 20x20 | 50x50 |

---
