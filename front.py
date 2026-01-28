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
                    answer, img_path = query_agent(question, with_exp=False, img=True)
                    put_markdown(f"**Query Result:**")
                    put_markdown(answer)
                    if img_path!="":
                        put_image(open(img_path, 'rb').read())
                except Exception as e:
                    put_text(f"Query error: {str(e)}")

            put_markdown("---")
            put_text("You can continue to enter new queries")


if __name__ == '__main__':
    start_server(main, port=8080, debug=True)
