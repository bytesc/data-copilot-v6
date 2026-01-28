docker pull opensearchproject/opensearch:3

docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=S202512sss" opensearchproject/opensearch:3

docker compose up -d

docker compose down


http://127.0.0.1:5601/
admin
S202512sss

DBeaver opensearch
jdbc:opensearch://https://127.0.0.1:9200?auth=basic&user=admin&password=S202512sss&trustSelfSigned=true

curl -X GET "https://localhost:9200/_cluster/health" -ku admin:S202512sss

curl -X DELETE "https://localhost:9200/medical_images" -ku "admin:S202512sss"



验证索引是否存在
curl -X GET "https://localhost:9200/brset" -u admin:S202512sss -k
获取索引mapping定义
curl -X GET "https://localhost:9200/brset/_mapping?pretty" -u admin:S202512sss -k
统计文档数量
curl -X GET "https://localhost:9200/brset/_count?pretty" -u admin:S202512sss -k
获取前5个文档验证数据
curl -X GET "https://localhost:9200/brset/_search?size=5&pretty" -u admin:S202512sss -k
检查索引健康状况
curl -X GET "https://localhost:9200/_cluster/health/brset?pretty" -u admin:S202512sss -k
查看所有索引
curl -X GET "https://localhost:9200/_cat/indices?v&s=index" -u admin:S202512sss -k
获取索引设置
curl -X GET "https://localhost:9200/brset/_settings?pretty" -u admin:S202512sss -k
验证特定字段meta
curl -X GET "https://localhost:9200/brset/_mapping/field/patient_age?pretty" -u admin:S202512sss -k
检查字段统计
curl -X GET "https://localhost:9200/brset/_field_stats?fields=patient_age,diabetic_retinopathy&pretty" -u admin:S202512sss -k
搜索包含meta的字段
curl -X GET "https://localhost:9200/brset/_search?pretty" -u admin:S202512sss -k -H "Content-Type: application/json" -d '{"query":{"exists":{"field":"_meta"}}}'

curl -X PUT "https://localhost:9200/brset/_mapping" -u admin:S202512sss -k -H "Content-Type: application/json" -d "{\"properties\":{\"diabetes\":{\"type\":\"keyword\",\"meta\":{\"description\":\"Diabetes status. e.g., yes/no.\"}}}}"