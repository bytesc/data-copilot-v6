import json

from agent.utils.json_2_dsl import OpenSearchJsonTranslator
from agent.utils.json_2_graph import visualize_opensearch_result
from agent.utils.json_query_t import run_opensearch_demo
from agent.utils.llm_access.LLM import get_llm
from agent.utils.llm_access.call_llm import call_llm
from agent.utils.llm_tools.parse_json import string_to_json
from agent.utils.search_tools.opensearch_connection import get_index_mapping, search_by_dsl

DB_NAME = 'brset'
from agent.utils.json_api_doc import json_doc
mapping = get_index_mapping(DB_NAME)
llm = get_llm()





def query_agent(question, tables=None,  retries=2, with_exp=False, img=True):
    pre_prompt = """
    You are a professional JSON query generation assistant. 
    Please generate the correct OpenSearch query JSON based on the user's question, 
    combined with the provided database structure and JSON API documentation.

    # Database Structure (Index Mapping):
    {mapping}

    # JSON API Documentation:
    {json_doc}

    Remind:
    1. Do not generate json which has not been defined in JSON API Documentation.
    2. generate the query JSON without any comments or explanation.

    Now please generate the query JSON based on this user question:
    User Question: {question}
    """
    full_prompt = pre_prompt.format(
        mapping=json.dumps(mapping, indent=2, ensure_ascii=False),
        json_doc=json_doc,
        question=question
    )

    ans = call_llm(full_prompt, llm).content
    print("\nllm ans: ########################")
    print(ans)

    query_json = string_to_json(ans)
    translator = OpenSearchJsonTranslator()
    dsl = translator.translate(query_json)
    print("\ndsl translated: ########################")
    print(dsl)

    result = search_by_dsl(dsl, index=DB_NAME, return_whole_response=True)
    print("\nquery result: ########################")
    print(result)

    processed = translator.process_result(result, query_json)
    print("\nans translated:")
    print(json.dumps(processed, indent=2, ensure_ascii=False))



    final_return = "\n```json\n" +json.dumps(processed, indent=2, ensure_ascii=False) + "\n```\n"

    if with_exp:
        final_return += get_data_explain(question, query_json, processed)+"\n"
    if img:
        path = visualize_opensearch_result(query_json, processed, "./graph")
        print(path)
        return final_return, path
    return final_return


def get_data_explain(question, query, result):
    pre_prompt="""
    You are a professional data assistant. 
    Another agent has just query the result based on the question and data mapping through JSON API. 
    Please describe the result in Natural language.
    
    # Database Structure (Index Mapping):
    {mapping}
    
    # JSON API Documentation:
    {json_doc}
    
    # query JSON:
    {query}
    
    # result
    {result}
    
    # User Question: 
    {question}
    
    Remind:
    1. Do not mention query detail, please focus on the result.
    2. with md format
    3. No more than 300 words.
    

    """
    full_prompt = pre_prompt.format(
        mapping=json.dumps(mapping, indent=2, ensure_ascii=False),
        json_doc=json_doc,
        question=question,
        query=query,
        result=result
    )
    ans = call_llm(full_prompt, llm).content
    return ans
