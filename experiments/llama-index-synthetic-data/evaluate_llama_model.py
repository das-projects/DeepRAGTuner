from llama_index.embeddings import LinearAdapterEmbeddingModel, resolve_embed_model
from llama_index.finetuning import EmbeddingQAFinetuneDataset
import pickle

from eval_utils import evaluate, display_results


def run_eval(val_data: str) -> None:
	val_dataset = EmbeddingQAFinetuneDataset.from_json(val_data)
	print("Loading model")
	embed_model_name = "local:BAAI/bge-large-en"
	base_embed_model = resolve_embed_model(embed_model_name)
	print("Loading adapter")
	embed_model = LinearAdapterEmbeddingModel(base_embed_model, "model_output_test", device="cuda")
	# Top k 10 to match our internal experiments
	print("Evaluating fine-tuned model")
	ft_val_results = evaluate(val_dataset, embed_model, top_k=10)
	print("Fine-Tuned Model Results")
	print(ft_val_results)
	with open("ft_results.pkl", "wb") as f:
		pickle.dump(ft_val_results, f)
	display_results(["ft"], [ft_val_results])

	print("Evaluating base model")
	bge_val_results = evaluate(val_dataset, embed_model_name, top_k=10)
	print("Base Model Results:")
	print(bge_val_results)
	with open("base_model_results.pkl", "wb") as f2:
		pickle.dump(bge_val_results, f2)
	display_results(["bge"], [bge_val_results])
	print("All Results")
	display_results(
		["bge", "ft"], [bge_val_results, ft_val_results]
	)



if __name__ == "__main__":
	run_eval("val.json")