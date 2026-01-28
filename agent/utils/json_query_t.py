import json
from typing import Dict, Any
import time

from agent.utils.json_2_dsl import OpenSearchJsonTranslator
from agent.utils.search_tools.opensearch_connection import search_by_dsl


def run_opensearch_demo():
    # 创建翻译器实例
    translator = OpenSearchJsonTranslator()

    print("🚀 开始OpenSearch查询演示")
    print("=" * 60)

    def demo_simple_stats():
        """演示简单统计查询"""
        print("\n📊 演示1: 患者年龄统计查询")
        print("-" * 40)

        # 构建统计查询：分析患者年龄的基本统计信息
        input_json = {
            "query": {
                "type": "stats",
                "config": {
                    "fields": ["patient_age"],
                    "metrics": ["min", "max", "avg", "count"],
                    "filters": [
                        {
                            "field": "diabetes",
                            "operator": "eq",
                            "value": "yes"
                        }
                    ]
                }
            }
        }

        try:
            # 翻译为 DSL
            dsl = translator.translate(input_json)
            print("生成的DSL:")
            print(json.dumps(dsl, indent=2, ensure_ascii=False))

            # 执行查询
            print("\n执行查询中...")
            result = search_by_dsl(dsl, index="brset", return_whole_response=True)
            print(result)
            # 处理结果
            processed = translator.process_result(result, input_json)
            print("\n处理后的统计结果:")
            print(json.dumps(processed, indent=2, ensure_ascii=False))

            return True
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return False

    def demo_diabetic_retinopathy_distribution():
        """演示糖尿病视网膜病变分布分析"""
        print("\n👁️ 演示2: 糖尿病视网膜病变年龄分布")
        print("-" * 40)

        input_json = {
            "query": {
                "type": "distribution",
                "config": {
                    "dimensions": ["diabetic_retinopathy"],
                    "buckets": [
                        {
                            "type": "range",
                            "field": "patient_age",
                            "ranges": [
                                {"key": "0-30岁", "from": 0, "to": 30},
                                {"key": "30-50岁", "from": 30, "to": 50},
                                {"key": "50-70岁", "from": 50, "to": 70},
                                {"key": "70岁以上", "from": 70}
                            ]
                        }
                    ],
                    "metrics": ["count", "percentage"],
                    "filters": [
                        {
                            "field": "diabetes",
                            "operator": "eq",
                            "value": "yes"
                        }
                    ]
                }
            }
        }

        try:
            # 翻译为 DSL
            dsl = translator.translate(input_json)
            print("生成的分布分析DSL:")
            print(json.dumps(dsl, indent=2, ensure_ascii=False))

            # 执行查询
            print("\n执行查询中...")
            result = search_by_dsl(dsl, index="brset", return_whole_response=True)
            print(result)

            # 处理结果
            processed = translator.process_result(result, input_json)
            print("\n糖尿病视网膜病变分布结果:")
            print(json.dumps(processed, indent=2, ensure_ascii=False))

            return True
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return False

    def demo_quality_analysis():
        """演示图像质量分析"""
        print("\n📷 演示3: 图像质量分析")
        print("-" * 40)

        input_json = {
            "query": {
                "type": "distribution",
                "config": {
                    "dimensions": ["quality"],
                    "groups": ["camera"],
                    "metrics": ["count", "percentage"],
                    "filters": [
                        {
                            "field": "Illuminaton",
                            "operator": "eq",
                            "value": 1  # 充足光照
                        }
                    ]
                }
            }
        }

        try:
            dsl = translator.translate(input_json)
            print("图像质量分析DSL:")
            print(json.dumps(dsl, indent=2, ensure_ascii=False))

            print("\n执行查询中...")
            result = search_by_dsl(dsl, index="brset", return_whole_response=True)
            print(result)
            processed = translator.process_result(result, input_json)
            print("\n图像质量分析结果:")
            print(json.dumps(processed, indent=2, ensure_ascii=False))

            return True
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return False

    # 执行所有演示
    demo_results = []

    # 执行演示用例
    demo_results.append(("患者年龄统计", demo_simple_stats()))
    time.sleep(1)  # 避免请求过快

    demo_results.append(("糖尿病视网膜病变分布", demo_diabetic_retinopathy_distribution()))
    time.sleep(1)

    demo_results.append(("图像质量分析", demo_quality_analysis()))

    # 输出演示总结
    print("\n" + "=" * 60)
    print("演示总结:")
    print("=" * 60)

    successful_demos = 0
    for demo_name, result in demo_results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{demo_name}: {status}")
        if result:
            successful_demos += 1

    print(f"\n总计演示: {len(demo_results)} 个")
    print(f"成功演示: {successful_demos} 个")

    success_rate = (successful_demos / len(demo_results)) * 100 if demo_results else 0
    print(f"成功率: {success_rate:.1f}%")


    return successful_demos == len(demo_results)


if __name__ == "__main__":
    print("=" * 60)
    print("OpenSearch JSON翻译器演示程序")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("开始运行演示...")
    print("=" * 60)

    # 第二步：运行演示
    try:
        success = run_opensearch_demo()

        if success:
            print("\n" + "🎊 所有演示完美完成！")
        else:
            print("\n" + "💡 演示完成，部分功能需要调整")

    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n💥 演示过程中发生错误: {e}")

    print("\n" + "=" * 60)
    print("演示程序结束")
    print("=" * 60)