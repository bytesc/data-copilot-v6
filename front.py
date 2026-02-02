import json

from pywebio import start_server
from pywebio.input import input, TEXT
from pywebio.output import put_text, put_markdown, put_loading, put_image
from agent.agent import query_agent


def main():
    put_markdown("# data-copilot-v5")
    put_text("Please enter your query question")

    while True:
        question = input("Query question:", type=TEXT, required=True)
        put_markdown(f"**Query question:**")
        put_markdown(question)
        if question.strip():
            with put_loading(shape="grow", color="primary"):
                try:
                    query_json, answer, json_path, img_path = query_agent(question,  img=True)
                    put_markdown(f"**Query:**")
                    put_markdown("\n```json\n" + json.dumps(query_json, indent=2, ensure_ascii=False) + "\n```\n")
                    put_markdown(f"**Result:**")
                    put_markdown("\n```json\n" + json.dumps(answer, indent=2, ensure_ascii=False) + "\n```\n")
                    put_markdown(f"**Object:**")
                    put_markdown(json_path)

                    if img_path!="":
                        put_image(open(img_path, 'rb').read())
                except Exception as e:
                    put_text(f"Query error: {str(e)}")

            put_markdown("---")
            put_text("You can continue to enter new queries")


if __name__ == '__main__':
    start_server(main, port=8080, debug=True)
