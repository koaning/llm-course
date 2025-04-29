# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llm==0.24.2",
#     "marimo",
#     "mohtml==0.1.7",
#     "pydantic==2.11.3",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import llm
    from dotenv import load_dotenv

    load_dotenv(".env")
    return (llm,)


@app.cell
def _(llm):
    llm.get_models()
    return


@app.cell
def _(llm):
    model = llm.get_model("gpt-4o-mini")

    resp = model.prompt(
        "Write me a haiku about Python",
        system="Answer like a pirate"
    )
    return model, resp


@app.cell
def _(resp):
    resp.json()
    return


@app.cell
def _(model):
    from pydantic import BaseModel

    class Haiku(BaseModel): 
        poem: str

    class Haikus(BaseModel):
        topic: str
        haikus: list[str]

    out = model.prompt("Haiku about Python", schema=Haikus)
    return BaseModel, out


@app.cell
def _(out):
    import json 

    json.loads(out.json()["content"])
    return (json,)


@app.cell
def _(llm):
    async_model = llm.get_async_model("gpt-4o")
    return


@app.cell
def _(mo, model):
    conversation = model.conversation()
    chat_widget = mo.ui.chat(lambda messages: conversation.prompt(messages[-1].content))
    chat_widget
    return (chat_widget,)


@app.cell
def _(chat_widget):
    chat_widget.value[1].content
    return


@app.cell
def _(BaseModel, json, mo, model):
    class Summary(BaseModel):
        title: str
        summary: str
        pros: list[str]
        cons: list[str]

    def summary(text_in):
        resp = model.prompt(
            f"Make a summary of the following text: {text_in}", 
        schema=Summary)
        return json.loads(resp.json()["content"])

    text_widget = mo.ui.text_area(
        label="Input to summary function"
    ).form()

    text_widget
    return summary, text_widget


@app.cell
def _(summary, text_widget):
    from pprint import pprint

    pprint(summary(text_widget.value))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
