

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import llm
    from dotenv import load_dotenv

    load_dotenv(".env")
    return llm, mo, pl


@app.cell
def _(pl):
    df = pl.read_csv("spam.csv")
    df.head(200).group_by("label").len()
    return (df,)


@app.cell
async def _():
    import asyncio
    from mosync import async_map_with_retry


    async def delayed_double(x):
        await asyncio.sleep(1)
        return x * 2

    results = await async_map_with_retry(
        range(100), 
        delayed_double, 
        max_concurrency=10, 
        description="Showing a simple demo"
    )
    return (async_map_with_retry,)


@app.cell
def _(llm):
    for model in llm.get_async_models():
        print(model.model_id)
    return


@app.cell
def _(llm):
    from diskcache import Cache

    cache = Cache("accuracy-experiment")

    models = {
        "gpt-4": llm.get_async_model("gpt-4"), 
        "gpt-4o": llm.get_async_model("gpt-4o"), 
    }


    prompt = "is this spam or ham? only reply with spam or ham"
    mod = "gpt-4o"

    async def classify(text, prompt=prompt, model=mod):
        tup = (text, prompt, model)
        if tup in cache: 
            return cache[tup]
        resp = await models[model].prompt(prompt + "\n" + text).json()
        cache[tup] = resp
        return resp
    return cache, classify, mod, models, prompt


@app.cell
async def _(classify):
    await classify("hello there")
    return


@app.cell
async def _(async_map_with_retry, classify, df):
    n_eval = 200

    llm_results = await async_map_with_retry(
        [_["text"] for _ in df.head(n_eval).to_dicts()], 
        classify, 
        max_concurrency=3, 
        description="Running LLM experiments"
    )
    return llm_results, n_eval


@app.cell
def _(df, llm_results, mo, n_eval, pl, prompt):
    n_correct = pl.DataFrame({**d, "pred": p} for d, p in zip(
        df.head(200).to_dicts(),
        [i.result["content"] for i in llm_results]
    )).filter(pl.col("label") == pl.col("pred")).shape[0]

    mo.md(f"""
    ### Prompt: 
    ```
    {prompt}
    ```
    The accuracy is {n_correct}/{n_eval} = {n_correct/n_eval*100:.1f}%
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's jot down some summaries. 

        - "is this spam or ham? only reply with spam or ham" / `gpt-4` `67.0%`
        - "is this spam or ham? only reply with spam or ham" / `gpt-4o` `67.5%`
        - "sometimes we need to deal with spammy text messages, that often promise free/cheap good. is this spam or ham? only reply with spam or ham" / `gpt-4` `66.5%`
        - "sometimes we need to deal with spammy text messages, that often promise free/cheap good. is this spam or ham? only reply with spam or ham" / `gpt-4o` `72.5%`
        """
    )
    return


@app.cell
def _(cache, mod, models):
    from pydantic import BaseModel

    class Classification(BaseModel):
        spam: bool    # change to spamham: str for fun
        reasoning: str

    _prompt = "you are an expert text classification system that can detect spam in text message. these messages are from the early 2000s so expect plenty of messy messages about ringtones."

    async def classify_schema(text, prompt=_prompt, model=mod):
        tup = (text, prompt, model, "schema2")
        if tup in cache: 
            return cache[tup]
        resp = await models[model].prompt(prompt + "\n" + text, schema=Classification).json()
        cache[tup] = resp
        return resp
    return (classify_schema,)


@app.cell
async def _(async_map_with_retry, classify_schema, df, n_eval):
    more_llm_results = await async_map_with_retry(
        [_["text"] for _ in df.head(n_eval).to_dicts()], 
        classify_schema, 
        max_concurrency=3, 
        description="Running LLM experiments"
    )
    return (more_llm_results,)


@app.cell
def _(more_llm_results):
    import json 

    [json.loads(i.result["content"])["spam"] for i in more_llm_results][:20]
    return (json,)


@app.cell
def _(df, json, mo, more_llm_results, n_eval, pl, prompt):
    _n_correct = pl.DataFrame({**d, "pred": p} for d, p in zip(
        df.head(200).to_dicts(),
        ["spam" if json.loads(i.result["content"])["spam"] else "ham" for i in more_llm_results]
    )).filter(pl.col("label") == pl.col("pred")).shape[0]

    mo.md(f"""
    ### Prompt: 
    ```
    {prompt}
    ```
    The accuracy is {_n_correct}/{n_eval} = {_n_correct/n_eval*100:.1f}%
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Fun fact, in that last experiment the thing that went wrong was the JSON generation.

    Running this experiment cost me about $2. In fairness: I had to rerun a few things a few times. But at the same time: that's pretty darn expensive for 6 variants on just 200 examples! Especially when you consider you could also build a spaCy/scikit-learn pipeline for this task.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
