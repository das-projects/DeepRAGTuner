# Domain Adapted Language Modeling Toolkit

## Manifesto

A great rift has emerged between general LLMs and the vector stores that are providing them with contextual information. The unification of these systems is an important step in grounding AI systems in efficient, factual domains, where they are utilized not only for their generality, but for their specificity and uniqueness. To this end, we are excited to open source the Arcee Domain Adapted Language Model (DALM) toolkit for developers to build on top of our Arcee open source Domain Pretrained (DPT) LLMs. We believe that our efforts will help as we begin next phase of language modeling, where organizations deeply tailor AI to operate according to their unique intellectual property and worldview. 

## Research Contents

This repository primarily contains code for fine-tuning a **fully differential** Retrieval Augmented Generation (RAG-end2end) architecture. 

![E2E](https://i.imgur.com/SDoY0oq.png)

For the first time in the literature, we modified the initial RAG-end2end model ([TACL paper](https://aclanthology.org/2023.tacl-1.1/), [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever)) to work with decoder-only language models like Llama, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

- Inside the [training](https://github.com/arcee-ai/DALM/tree/main/dalm/training) folder, you'll find two codes to train the RAG-end2end and Retriever with contrastive learning.

- All evaluations related to the Retriever and the Generator are located in the [eval](https://github.com/arcee-ai/DALM/tree/main/dalm/eval) folder.

- Additionally, we have data processing codes and synthetic data generation code inside the [datasets](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets) folder.

# Usage
To perform training and evaluation for both the retriever model and the new rag-e2e model, please adhere to the following steps.

## System Requirements

The system reqs depend on the retriever model, generator model, and batch size. But for reference (e2e rag), we used the following for our experiments (eval results below):
* retriever: `BAAI/bge-large-en`
* generator: `meta-llama/Llama-2-7b-hf`
* batch size: 18
* dataset size: 200k

This took 7 hours on a single A100 GPU (80GB).

## Installation

You can install this repo directly via `pip install indomain`

Alternatively, for development or research, you can clone and install the repo locally:
```shell
git clone https://github.com/arcee-ai/DALM.git && cd DALM
pip install --upgrade -e .
```
This will install the DALM repo and all necessary dependencies.

Make sure things are installed correctly by running `dalm version`.  On an non-intel Mac you may need to downgrade `transformers` library: `pip install transformers==4.30`.

## Data setup
### tl;dr
You can run `dalm qa-gen <path-to-dataset>` to preprocess your dataset for training. See `dalm qa-gen --help` for more options
<br>If you do not have a dataset, you can start with ours
```shell
# Note - our dataset already has queries and answers, so you don't actually need to run this.
# replace `toy_dataset_train.csv` with your dataset of titles and passages
dalm qa-gen dalm/datasets/toy_data_train.csv
```
- The setup for training and evaluation can be effortlessly executed provided you possess a [CSV](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets/toy_data_train.csv) file containing two/three columns: `Passage`, `Query` (and `Answer` if running e2e). You can utilize the script [question_answer_generation.py](https://github.com/arcee-ai/DALM/blob/main/dalm/datasets/qa_gen/question_answer_generation.py) to generate this CSV. 
- It's important to highlight that the retriever-only training method employs solely the passages and queries, whereas the rag-e2e training code utilizes all three columns.
- In our experiments, we utilize `BAAI/bge-large-en` as the default retriever and employ `meta-llama/Llama-2-7b-hf` as the default generator. The code is designed to be compatible with any embedding model or autoregressive model available in the Hugging Face model repository at https://huggingface.co/models.

## Training

You can leverage our scripts directly if you'd like, or you can use the `dalm` cli. The arguments for both are identical

### Train Retriever Only

Train `BAAI/bge-large-en` retriever with contrastive learning.
```shell
python dalm/training/retriever_only/train_retriever_only.py \
--dataset_path "./dalm/datasets/toy_data_train.csv" \
--retriever_name_or_path "BAAI/bge-large-en" \
--output_dir "retriever_only_checkpoints" \
--use_peft \
--with_tracking \
--report_to all \
--per_device_train_batch_size 150
```
or
```shell
dalm train-retriever-only "BAAI/bge-large-en" "./dalm/datasets/toy_data_train.csv" \
--output-dir "retriever_only_checkpoints" \
--use-peft \
--with-tracking \
--report-to all \
--per-device-train-batch-size 150
```

For all available arguments and options, see `dalm train-retriever-only --help`

### Train Retriever and Generator Jointly (RAG-e2e)
Train `Llama-2-7b` generator jointly with the retriever model `BAAI/bge-large-en`.

```shell
python dalm/training/rag_e2e/train_rage2e.py \
  --dataset_path "./dalm/datasets/toy_data_train.csv" \
  --retriever_name_or_path "BAAI/bge-large-en" \
  --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "rag_e2e_checkpoints" \
  --with_tracking \
  --report_to all \
  --per_device_train_batch_size 20
```
or
```shell
dalm train-rag-e2e \
"./dalm/datasets/toy_data_train.csv" \
"BAAI/bge-large-en" \
"meta-llama/Llama-2-7b-hf" \
--output-dir "rag_e2e_checkpoints" \
--with-tracking \
--report-to all \
--per-device-train-batch-size 20
```

For all available arguments and options, see `dalm train-rag-e2e --help`

## Evaluation

#### Summary of how evaluation is done

The Retriever in general is trained to be good at finding the most relevant passages in a corpus given a query.

Given a ground-truth test dataset that is a 200,000-line CSV containing patent abstracts and more importantly this evaluation dataset was not present in the training dataset, the below listed steps were followed:

1. Use the trained retriever to encode all passages into an ad-hoc indexed vector store using the HNSW library.
2. Take each query and use the trained retriever to encode it into an embedding vector (QE)
3. For each encoded passage (PE) in the vector store, find the nearest neighbor similarity search score between QE and PE (**Note**: with HNSW, exhaustiveness is avoided)
4. Find the top-K (eg, top 5) best matches based on nearest neighbor similarity search scores
5. Compare the matches against the ground truth top-K best matches to calculate `recall` and `hit rate`.

#### Results

| Type of Retriever | Recall | Hit rate |
| --- | ----- | ----|
| Plain Retriever | 0.45984 | 0.45984 |
| Retriever with contrastive learning | 0.46037 | 0.46038 |
| Retriever End2End | 0.73634 | 0.73634 |

To run retriever only eval 
(make sure you have the checkpoints in the project root)

```bash
 python dalm/eval/eval_retriever_only.py  \
 --dataset_path qa_pairs_test.csv \
 --retriever_name_or_path "BAAI/bge-large-en" \
 --passage_column_name Abstract \
 --query_column_name Question \
 --retriever_peft_model_path retriever_only_checkpoints
```
or 
```bash
dalm eval-retriever qa_pairs_test.csv \
 --retriever-name-or-path "BAAI/bge-large-en" \
 --passage-column-name Abstract \
 --query-column-name Question \
 --retriever-peft-model-path retriever_only_checkpoints
```
See `dalm eval-retriever --help` for all available arguments

For the e2e eval

```bash
python dalm/eval/eval_rag.py  \
 --dataset_path qa_pairs_test_2.csv \
 --retriever_name_or_path "BAAI/bge-large-en" \
 --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
 --passage_column_name Abstract \
 --query_column_name Question \
 --answer_column_name Answer \
 --evaluate_generator \
 --query_batch_size 5 \
 --retriever_peft_model_path rag_e2e_checkpoints/retriever \
 --generator_peft_model_path rag_e2e_checkpoints/generator
```
or
```bash
dalm eval-rag qa_pairs_test.csv \
 --retriever-name-or-path "BAAI/bge-large-en" \
 --generator-name-or-path "meta-llama/Llama-2-7b-hf" \
 --retriever-peft-model-path rag_e2e_checkpoints/retriever \
 --generator-peft-model-path rag_e2e_checkpoints/generator \
 --passage-column-name Abstract \
 --query-column-name Question \
 --answer-column-name Answer \
 --query-batch-size 5
```
See `dalm eval-rag --help` for all available arguments

# How to run evaluation

To run retriever only eval 
(make sure you have the checkpoints in the project root)

```bash
 python dalm/eval/eval_retriever_only.py  --dataset_path qa_paits_test.csv --retriever_name_or_path "BAAI/bge-large-en" --passage_column_name Abstract --query_column_name Question --retriever_peft_model_path retriever_only_checkpoints
```

For the e2e eval

```bash
python dalm/eval/eval_rag.py  --dataset_path qa_pairs_test_2.csv --retriever_name_or_path "BAAI/bge-large-en" --generator_name_or_path "meta-llama/Llama-2-7b-hf" --passage_column_name Abstract --query_column_name Question --answer_column_name Answer --evaluate_generator --query_batch_size 5 --retriever_peft_model_path retriever_only_checkpoints --generator_peft_model_path generator_only_checkpoints
```

## Contributing
See [CONTRIBUTING](https://github.com/arcee-ai/DALM/tree/main/CONTRIBUTING.md)

# gpt-fast
Simple and efficient pytorch-native transformer text generation.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding
6. Tensor parallelism
7. Supports Nvidia and AMD GPUs

This is *NOT* intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
Install sentencepiece and huggingface_hub
```bash
pip install sentencepiece huggingface_hub
```

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then login with `huggingface-cli login`



## Downloading Weights
Models tested/supported
```text
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

## Benchmarks
Benchmarks run on an A100-80GB, power limited to 330W.

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  104.9  | 1397.31 |
|           | 8-bit   | 155.58   | 1069.20 |
|           | 4-bit (G=32)   | 196.80   | 862.69 |
| Llama-2-70B | Base    | OOM     ||
|           | 8-bit   | 19.13    | 1322.58 |
|           | 4-bit (G=32)   | 25.25    | 1097.66 |

### Speculative Sampling
[Verifier: Llama-70B (int4), Draft: Llama-7B (int4)](./scripts/speculate_70B_int4.sh): 48.4 tok/s

### Tensor Parallelism
| Model    | Number of GPUs | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | 1    |  104.9  | 1397.31 |
|           | 2   | 136.27   | 954.01 |
|           | 4   | 168.78   | 635.09 |
|           | 8   | 179.27   | 395.85 |
| Llama-2-70B  | 1    |  OOM  |  |
|           | 2   | 20.53   | 1426.41 |
|           | 4   | 34.15   | 1204.62 |
|           | 8   | 47.25   | 858.28 |

### AMD
Benchmarks run on one GCD of a MI-250x.

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  76.33  | 1028.70 |
|           | 8-bit   | 101.86   | 700.06 |

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth
```

### Int4 Weight-Only Quantization
To generate int4 version of model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32
```

To run with int4, just pass the int4 checkpoint to generate.py.
```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --compile
```

## Speculative Sampling
To generate with speculative sampling (DRAFT_MODEL_REPO should point to a smaller model compared with MODEL_REPO).

In this example, the "smaller" model is just the int8 quantized version of the model.
```
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth
```

Note: Running on an A100 80GB, albeit power-limited to 330 watts. Empirically, seems like peak bandwidth is about 1700 GB/s.


## Tensor Parallelism
```bash
torchrun --standalone --nproc_per_node=2 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

## Experimental
### Evaluation
We use the EleutherAI evaluation harness to evaluate our model accuracy. To evaluate the accuracy, make sure the evaluation harness is installed and pass your model checkpoint and desired tasks to eval.py.

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

Note: Generative tasks are currently not supported for gpt-fast

Installation Instructions for the evaluation harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master#install

### GPTQ
We have a pure pytorch implementation of GPTQ that utilizes torch._dynamo.export to access the model structure. You can generate a GPTQ quantized
version of int4 quantization by using the same command to quantize it but adding 'gptq' to the quantization mode i.e.
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4-gptq.g32.pth
python quantize.py --mode int4-gptq --calibration_tasks wikitext --calibration_seq_length 2048
```

You can then eval or generate text with this model in the same way as above.

## License

`gpt-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements
Thanks to:
* Lightning AI for supporting pytorch and work in flash attention, int8 quantization, and LoRA fine-tuning.
* GGML for driving forward fast, on device inference of LLMs
* Karpathy for spearheading simple, interpretable and fast LLM implementations
* MLC-LLM for pushing 4-bit quantization performance on heterogenous hardware


<p align="center">
  <a href="https://nextjs-fastapi-starter.vercel.app/">
    <img src="https://assets.vercel.com/image/upload/v1588805858/repositories/vercel/logo.png" height="96">
    <h3 align="center">Next.js FastAPI Starter</h3>
  </a>
</p>

<p align="center">Simple Next.js boilerplate that uses <a href="https://fastapi.tiangolo.com/">FastAPI</a> as the API backend.</p>

<br/>

## Introduction

This is a hybrid Next.js + Python app that uses Next.js as the frontend and FastAPI as the API backend. One great use case of this is to write Next.js apps that use Python AI libraries on the backend.

## How It Works

The Python/FastAPI server is mapped into to Next.js app under `/api/`.

This is implemented using [`next.config.js` rewrites](https://github.com/digitros/nextjs-fastapi/blob/main/next.config.js) to map any request to `/api/:path*` to the FastAPI API, which is hosted in the `/api` folder.

On localhost, the rewrite will be made to the `127.0.0.1:8000` port, which is where the FastAPI server is running.

In production, the FastAPI server is hosted as [Python serverless functions](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python) on Vercel.

## Demo

https://nextjs-fastapi-starter.vercel.app/

## Deploy Your Own

You can clone & deploy it to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fdigitros%2Fnextjs-fastapi%2Ftree%2Fmain)

## Developing Locally

You can clone & create this repo with the following command

```bash
npx create-next-app nextjs-fastapi --example "https://github.com/digitros/nextjs-fastapi"
```

## Getting Started

First, install the dependencies:

```bash
npm install
# or
yarn
# or
pnpm install
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The FastApi server will be running on [http://127.0.0.1:8000](http://127.0.0.1:8000) – feel free to change the port in `package.json` (you'll also need to update it in `next.config.js`).

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - learn about FastAPI features and API.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!
