import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.testset import TestsetGenerator


load_dotenv()


@dataclass
class Provider:
    name: str
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    embedding_dimensions: int | None = None
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_docs(data_dir: str = "data"):
    return DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    return splitter.split_documents(docs)


def generate_evalset_from_docs(docs, testset_size: int = 5):
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4.1-mini",
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
            )
        ),
        embedding_model=LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
        ),
    )

    testset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    df = testset.to_pandas()

    rows = []
    for _, r in df.iterrows():
        q = r.get("user_input") or r.get("question")
        ref = r.get("reference") or r.get("reference_answer") or ""
        if q:
            rows.append({"question": str(q), "reference": str(ref)})
    return rows


def build_retriever(provider: Provider, chunks):
    emb = OpenAIEmbeddings(
        model=provider.embedding_model,
        openai_api_key=provider.api_key,
        openai_api_base=provider.base_url,
    )
    vs = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=emb,
        location=":memory:",
        collection_name=f"rag_eval_{provider.name}",
    )
    return vs.as_retriever(search_kwargs={"k": 4})


def _extract_token_usage(msg) -> tuple[int, int, int]:
    usage = getattr(msg, "response_metadata", {}).get("token_usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def _estimate_cost_usd(provider: Provider, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1_000_000) * provider.input_cost_per_1m
        + (completion_tokens / 1_000_000) * provider.output_cost_per_1m
    )


@traceable(name="rag_answer")
def run_rag_query(provider: Provider, retriever, question: str):
    llm = ChatOpenAI(
        model=provider.chat_model,
        openai_api_key=provider.api_key,
        openai_api_base=provider.base_url,
        temperature=0,
    )

    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    context_text = "\n\n".join(contexts)

    prompt = (
        "Use ONLY the context below to answer the question. "
        "If answer is not in context, say 'I don't know'.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
    )

    msg = llm.invoke(
        prompt,
        config={"tags": [provider.name], "run_name": f"{provider.name}_qa"},
    )
    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
    prompt_tokens, completion_tokens, total_tokens = _extract_token_usage(msg)

    return {
        "answer": answer,
        "contexts": contexts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": _estimate_cost_usd(
            provider, prompt_tokens, completion_tokens
        ),
    }


def build_eval_dataset(provider: Provider, eval_rows, chunks):
    retriever = build_retriever(provider, chunks)

    records = []
    for row in eval_rows:
        out = run_rag_query(provider, retriever, row["question"])
        records.append(
            {
                "user_input": row["question"],
                "response": out["answer"],
                "retrieved_contexts": out["contexts"],
                "reference": row["reference"],
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "total_tokens": out["total_tokens"],
                "estimated_cost_usd": out["estimated_cost_usd"],
            }
        )

    df = pd.DataFrame(records)
    return EvaluationDataset.from_pandas(df), df


def evaluate_provider(eval_dataset):
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4.1-mini",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    )

    return evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
        ],
        llm=evaluator_llm,
    )


def print_cost_summary(provider_name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {provider_name} Token/Cost Summary ===")
    print(f"queries: {len(df)}")
    print(f"prompt_tokens: {int(df['prompt_tokens'].sum())}")
    print(f"completion_tokens: {int(df['completion_tokens'].sum())}")
    print(f"total_tokens: {int(df['total_tokens'].sum())}")
    print(f"total_estimated_cost_usd: {float(df['estimated_cost_usd'].sum()):.6f}")
    print(
        f"avg_estimated_cost_usd_per_query: {float(df['estimated_cost_usd'].mean() if len(df) else 0.0):.6f}"
    )


def _ragas_summary_dict(result) -> dict:
    if not hasattr(result, "to_pandas"):
        return {}
    df = result.to_pandas()
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    return {col: float(numeric[col].mean()) for col in numeric.columns}


def print_comparison_table(
    fw_result, oa_result, fw_df: pd.DataFrame, oa_df: pd.DataFrame
) -> None:
    fw_metrics = _ragas_summary_dict(fw_result)
    oa_metrics = _ragas_summary_dict(oa_result)

    comparison = pd.DataFrame(
        [
            {
                "provider": "fireworks",
                **fw_metrics,
                "queries": len(fw_df),
                "total_tokens": int(fw_df["total_tokens"].sum()),
                "total_estimated_cost_usd": float(fw_df["estimated_cost_usd"].sum()),
                "avg_estimated_cost_usd_per_query": float(
                    fw_df["estimated_cost_usd"].mean() if len(fw_df) else 0.0
                ),
            },
            {
                "provider": "openai",
                **oa_metrics,
                "queries": len(oa_df),
                "total_tokens": int(oa_df["total_tokens"].sum()),
                "total_estimated_cost_usd": float(oa_df["estimated_cost_usd"].sum()),
                "avg_estimated_cost_usd_per_query": float(
                    oa_df["estimated_cost_usd"].mean() if len(oa_df) else 0.0
                ),
            },
        ]
    )
    print("\n=== Provider Comparison Table ===")
    print(comparison.to_string(index=False))


def main():
    fireworks = Provider(
        name="fireworks",
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
        chat_model=os.getenv("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"),
        embedding_model=os.getenv("FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-8b"),
        embedding_dimensions=4096,
        input_cost_per_1m=_float_env("FIREWORKS_INPUT_COST_PER_1M", 0.0),
        output_cost_per_1m=_float_env("FIREWORKS_OUTPUT_COST_PER_1M", 0.0),
    )

    openai = Provider(
        name="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=None,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input_cost_per_1m=_float_env("OPENAI_INPUT_COST_PER_1M", 0.0),
        output_cost_per_1m=_float_env("OPENAI_OUTPUT_COST_PER_1M", 0.0),
    )

    docs = load_docs("data")
    chunks = split_docs(docs)
    eval_rows = generate_evalset_from_docs(docs, testset_size=30)

    fw_eval_dataset, fw_df = build_eval_dataset(fireworks, eval_rows, chunks)
    oa_eval_dataset, oa_df = build_eval_dataset(openai, eval_rows, chunks)

    fw_result = evaluate_provider(fw_eval_dataset)
    oa_result = evaluate_provider(oa_eval_dataset)

    print("\n=== Fireworks RAGAS ===")
    print(fw_result)
    print_cost_summary("Fireworks", fw_df)

    print("\n=== OpenAI RAGAS ===")
    print(oa_result)
    print_cost_summary("OpenAI", oa_df)
    print_comparison_table(fw_result, oa_result, fw_df, oa_df)


if __name__ == "__main__":
    main()
