"""Adapted from https://github.com/jerryjliu/llama_index/blob/main/docs/examples/finetuning/embeddings/eval_utils.py"""

from llama_index.schema import TextNode
from llama_index import ServiceContext, VectorStoreIndex
import pandas as pd
from tqdm import tqdm


def evaluate(
    dataset,
    embed_model,
    top_k=10,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    ct = 0
    for query_id, query in tqdm(queries.items()):
        # We only evaluate the first 2000 queries. We have to do this because llama-index was way too slow. all 200k
        # test queries was scheduled to take 400 hours on the A100. So we bring it to 2k queries, which should take
        # 4 hours
        if ct >= 2000:
            break
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]

        rank = None
        for idx, id in enumerate(retrieved_ids):
            if id == expected_id:
                rank = idx + 1
                break

        is_hit = rank is not None  # assume 1 relevant doc
        mrr = 0 if rank is None else 1 / rank

        eval_result = {
            "is_hit": is_hit,
            "mrr": mrr,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
        ct += 1
    return eval_results


def display_results(names, results_arr):
    """Display results from evaluate."""

    hit_rates = []
    mrrs = []
    for name, results in zip(names, results_arr):
        results_df = pd.DataFrame(results)
        hit_rate = results_df["is_hit"].mean()
        mrr = results_df["mrr"].mean()
        hit_rates.append(hit_rate)
        mrrs.append(mrr)

    final_df = pd.DataFrame({"retrievers": names, "hit_rate": hit_rates, "mrr": mrrs})
    print(final_df)
