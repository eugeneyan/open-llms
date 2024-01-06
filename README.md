# Open LLMs

These LLMs (Large Language Models) are all licensed for commercial use (e.g., Apache 2.0, MIT, OpenRAIL-M). Contributions welcome!

| Language Model | Release Date | Checkpoints | Paper/Blog | Params (B) | Context Length | Licence | Try it                                                                                                                |
| --- | --- | --- | --- | --- | --- | --- |-----------------------------------------------------------------------------------------------------------------------|
| T5           | 2019/10 |[T5 & Flan-T5](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints), [Flan-T5-xxl (HF)](https://huggingface.co/google/flan-t5-xxl)      | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints) | 0.06 - 11       | [512](https://discuss.huggingface.co/t/does-t5-truncate-input-longer-than-512-internally/3602) | Apache 2.0         | [T5-Large](https://github.com/slai-labs/get-beam/tree/main/examples/t5)                                               |
| UL2          | 2022/10 | [UL2 & Flan-UL2](https://github.com/google-research/google-research/tree/master/ul2#checkpoints), [Flan-UL2 (HF)](https://huggingface.co/google/flan-ul2)          | [UL2 20B: An Open Source Unified Language Learner](https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html)                                                       | 20             | [512, 2048](https://huggingface.co/google/flan-ul2#tldr) | Apache 2.0         |                                                                                                                       |
| Cerebras-GPT | 2023/03 | [Cerebras-GPT](https://huggingface.co/cerebras)                                           | [Cerebras-GPT: A Family of Open, Compute-efficient, Large Language Models](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/) ([Paper](https://arxiv.org/abs/2304.03208)) | 0.111 - 13      | [2048](https://huggingface.co/cerebras/Cerebras-GPT-13B#model-details) | Apache 2.0         | [Cerebras-GPT-1.3B](https://github.com/slai-labs/get-beam/tree/main/examples/cerebras-gpt)                            |
| Open Assistant (Pythia family) | 2023/03 | [OA-Pythia-12B-SFT-8](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps), [OA-Pythia-12B-SFT-4](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5), [OA-Pythia-12B-SFT-1](https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b) | [Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327) | 12     | [2048](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps/blob/main/config.json)  | Apache 2.0                | [Pythia-2.8B](https://github.com/slai-labs/get-beam/tree/main/examples/pythia)                                        |
| ChatGLM-6B | 2023/03 | [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) | [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) | 6 | [2048](https://huggingface.co/THUDM/chatglm-6b/blob/main/config.json) | [License](https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE) | |
| Pythia       | 2023/04 | [pythia 70M - 12B](https://github.com/EleutherAI/pythia)                                   | [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)                                                                    | 0.07 - 12       | [2048](https://arxiv.org/pdf/2304.01373.pdf) | Apache 2.0         |                                                                                                                       |
| Dolly        | 2023/04 | [dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)                            | [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)             | 3, 7, 12     | [2048](https://github.com/databrickslabs/dolly#dolly) | MIT                |                                                                                                                       |
| DLite | 2023/05 | [dlite-v2-1_5b](https://huggingface.co/aisquared/dlite-v2-1_5b) | [Announcing DLite V2: Lightweight, Open LLMs That Can Run Anywhere](https://medium.com/ai-squared/announcing-dlite-v2-lightweight-open-llms-that-can-run-anywhere-a852e5978c6e) | 0.124 - 1.5 | [1024](https://huggingface.co/aisquared/dlite-v2-1_5b/blob/main/config.json) | Apache 2.0         | [DLite-v2-1.5B](https://github.com/slai-labs/get-beam/tree/main/examples/dlite-v2)                                    |
| RWKV         | 2021/08| [RWKV, ChatRWKV](https://github.com/BlinkDL/RWKV-LM#rwkv-parallelizable-rnn-with-transformer-level-llm-performance-pronounced-as-rwakuv-from-4-major-params-r-w-k-v) | [The RWKV Language Model (and my LM tricks)](https://github.com/BlinkDL/RWKV-LM)                                           | 0.1 - 14      | [infinity (RNN)](https://github.com/BlinkDL/RWKV-LM#rwkv-parallelizable-rnn-with-transformer-level-llm-performance-pronounced-as-rwakuv-from-4-major-params-r-w-k-v) | Apache 2.0         |                                                                                                                       |
| GPT-J-6B | 2023/06 | [GPT-J-6B](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b), [GPT4All-J](https://github.com/nomic-ai/gpt4all#raw-model) | [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/) | 6 | [2048](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b) | Apache 2.0 |                                                                                                                       |
| GPT-NeoX-20B | 2022/04 | [GPT-NEOX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) | [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745) | 20 | [2048](https://huggingface.co/EleutherAI/gpt-neox-20b) | Apache 2.0 |                                                                                                                       |
| Bloom | 2022/11 | [Bloom](https://huggingface.co/bigscience/bloom) | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | 176 | [2048](https://huggingface.co/bigscience/bloom) |  [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                                                                                                                       |
| StableLM-Alpha | 2023/04 | [StableLM-Alpha](https://github.com/Stability-AI/StableLM#stablelm-alpha) | [Stability AI Launches the First of its StableLM Suite of Language Models](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) | 3 - 65 | [4096](https://github.com/Stability-AI/StableLM#stablelm-alpha) | CC BY-SA-4.0 |                                                                                                                       |
| FastChat-T5 | 2023/04 | [fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) | [We are excited to release FastChat-T5: our compact and commercial-friendly chatbot!](https://twitter.com/lmsysorg/status/1652037026705985537?s=20) | 3 | [512](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0/blob/main/config.json) | Apache 2.0 |                                                                                                                       |
| h2oGPT | 2023/05 | [h2oGPT](https://github.com/h2oai/h2ogpt) | [Building the World’s Best Open-Source Large Language Model: H2O.ai’s Journey](https://h2o.ai/blog/building-the-worlds-best-open-source-large-language-model-h2o-ais-journey/) | 12 - 20 | [256 - 2048](https://huggingface.co/h2oai) | Apache 2.0 |                                                                                                                       |
| MPT-7B | 2023/05 | [MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) | [Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b) | 7 | [84k (ALiBi)](https://huggingface.co/mosaicml/mpt-7b#how-is-this-model-different) | Apache 2.0, CC BY-SA-3.0 |                                                                                                                       |
| RedPajama-INCITE | 2023/05 | [RedPajama-INCITE](https://huggingface.co/togethercomputer) | [Releasing 3B and 7B RedPajama-INCITE family of models including base, instruction-tuned & chat models](https://www.together.xyz/blog/redpajama-models-v1) | 3 - 7 | [2048](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1/blob/157bf3174feebb67f37e131ea68f84dee007c687/config.json#L13) | Apache 2.0 | [RedPajama-INCITE-Instruct-3B-v1](https://github.com/slai-labs/get-beam/tree/main/examples/redpajama-incite-instruct) |
| OpenLLaMA | 2023/05 | [open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b), [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b), [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b) | [OpenLLaMA: An Open Reproduction of LLaMA](https://github.com/openlm-research/open_llama) | 3, 7 | [2048](https://huggingface.co/h2oai) | Apache 2.0 | [OpenLLaMA-7B-Preview_200bt](https://github.com/slai-labs/get-beam/tree/main/examples/openllama)                      |
| Falcon | 2023/05 | [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B), [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b), [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) | [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116) | 180, 40, 7 | [2048](https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json) | Apache 2.0 | 
| MPT-30B | 2023/06 | [MPT-30B](https://huggingface.co/mosaicml/mpt-30b), [MPT-30B-instruct](https://huggingface.co/mosaicml/mpt-30b-instruct) | [MPT-30B: Raising the bar for open-source foundation models](https://www.mosaicml.com/blog/mpt-30b) | 30 | [8192](https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json) | Apache 2.0, CC BY-SA-3.0 | [MPT 30B inference code using CPU](https://github.com/abacaj/mpt-30B-inference) |
| LLaMA 2  | 2023/06 | [LLaMA 2 Weights](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-ham3-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=qhK-ahCbkBMAX94XV2X&_nc_ht=scontent-ham3-1.xx&oh=00_AfDB7dN8momft9nkv8X0gqrZdEnKltVjPOxhKBm0XLRinA&oe=64BE66FF)      | 7 - 70       | [4096](https://scontent-ham3-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=qhK-ahCbkBMAX94XV2X&_nc_ht=scontent-ham3-1.xx&oh=00_AfDB7dN8momft9nkv8X0gqrZdEnKltVjPOxhKBm0XLRinA&oe=64BE66FF)  | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) Free if you have under 700M users and you cannot use LLaMA outputs to train other LLMs besides LLaMA and its derivatives   | [HuggingChat](https://huggingface.co/blog/llama2#demo) |  
| ChatGLM2-6B & ChatGLM2-6B-32k | 2023/06 | [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) | [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) | 6 | [4 , 32K ](https://huggingface.co/THUDM/chatglm2-6b/blob/main/config.json) | [License](https://huggingface.co/THUDM/chatglm2-6b/blob/main/MODEL_LICENSE) | |
| OpenLM  | 2023/09 | [OpenLM 1B](https://huggingface.co/mlfoundations/open_lm_1B), [OpenLM 7B](https://huggingface.co/mlfoundations/open_lm_7B_1.25T) | [Open LM:  a minimal but performative language modeling (LM) repository](https://github.com/mlfoundations/open_lm#pretrained-models)      | 1, 7       | [2048](https://github.com/mlfoundations/open_lm/blob/main/open_lm/model_configs/open_lm_7b.json)  | MIT   |      |  
| Mistral 7B | 2023/09 | [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) | 7 | [4096-16K with Sliding Windows](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/config.json)| Apache 2.0 | [Mistral Transformer](https://github.com/mistralai/mistral-src)
| OpenHermes | 2023/09 | [OpenHermes-7B](https://huggingface.co/teknium/OpenHermes-7B), [OpenHermes-13B](https://huggingface.co/teknium/OpenHermes-13B) | [Nous Research](https://nousresearch.com/) | 7, 13 | [4096](https://huggingface.co/teknium/OpenHermes-13B/blob/main/config.json)| MIT | [OpenHermes-V2 Finetuned on Mistral 7B](https://huggingface.co/spaces/artificialguybr/OPENHERMES-2)
| ChatGLM3-6B & ChatGLM3-6B-32k | 2023/10 | [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) | [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) | 6 | [8192 , 32K ](https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json) | [License](https://huggingface.co/THUDM/chatglm3-6b/blob/main/MODEL_LICENSE) | |
| SOLAR | 2023/12 | [Solar-10.7B](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) | [Upstage](https://arxiv.org/abs/2312.15166) | 10.7 | [4096](https://huggingface.co/upstage/SOLAR-10.7B-v1.0/blob/main/config.json)| Apache 2.0 | |

## Open LLMs for code  

| Language Model | Release Date | Checkpoints | Paper/Blog | Params (B) | Context Length                                                                         | Licence | Try it                                                                                    |
| --- | --- | --- | --- | --- |----------------------------------------------------------------------------------------| --- |-------------------------------------------------------------------------------------------|
| SantaCoder | 2023/01 | [santacoder](https://huggingface.co/bigcode/santacoder) |[SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988) | 1.1 | [2048](https://huggingface.co/bigcode/santacoder/blob/main/README.md#model-summary)                | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) | [SantaCoder](https://github.com/slai-labs/get-beam/tree/main/examples/santacoder)         |
| StarCoder | 2023/05 | [starcoder](https://huggingface.co/bigcode/starcoder) | [StarCoder: A State-of-the-Art LLM for Code](https://huggingface.co/blog/starcoder), [StarCoder: May the source be with you!](https://drive.google.com/file/d/1cN-b9GnWtHzQRoE7M7gAEyivY0kl4BYs/view) | 1.1-15 | [8192](https://huggingface.co/bigcode/starcoder#model-summary)                         | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                                                                                           |
| StarChat Alpha | 2023/05 | [starchat-alpha](https://huggingface.co/HuggingFaceH4/starchat-alpha) | [Creating a Coding Assistant with StarCoder](https://huggingface.co/blog/starchat-alpha) | 16 | [8192](https://huggingface.co/bigcode/starcoder#model-summary)  | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                                                                                           |
| Replit Code | 2023/05 | [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) | [Training a SOTA Code LLM in 1 week and Quantifying the Vibes — with Reza Shabani of Replit](https://www.latent.space/p/reza-shabani#details) | 2.7 | [infinity? (ALiBi)](https://huggingface.co/replit/replit-code-v1-3b#model-description) | CC BY-SA-4.0 | [Replit-Code-v1-3B](https://github.com/slai-labs/get-beam/tree/main/examples/replit-code) |
| CodeGen2 | 2023/04 | [codegen2 1B-16B](https://github.com/salesforce/CodeGen2) | [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309) | 1 - 16 | [2048](https://arxiv.org/abs/2305.02309) | [Apache 2.0](https://github.com/salesforce/CodeGen2/blob/main/LICENSE)|                                                                                           |
| CodeT5+ | 2023/05 | [CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+)     | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922) | 0.22 - 16 | [512](https://arxiv.org/abs/2305.07922)                                                                                | [BSD-3-Clause](https://github.com/salesforce/CodeT5/blob/main/LICENSE.txt)                                                           | [Codet5+-6B](https://github.com/slai-labs/get-beam/tree/main/examples/codeT5%2B)          |
| XGen-7B | 2023/06 | [XGen-7B-8K-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base) | [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen/) | 7 | [8192](https://huggingface.co/Salesforce/xgen-7b-8k-base/blob/main/config.json) | [Apache 2.0](https://github.com/salesforce/xgen/blob/main/LICENSE) | 
| CodeGen2.5 | 2023/07 | [CodeGen2.5-7B-multi](https://huggingface.co/Salesforce/codegen25-7b-multi) | [CodeGen2.5: Small, but mighty](https://blog.salesforceairesearch.com/codegen25/) | 7 | [2048](https://huggingface.co/Salesforce/codegen25-7b-multi/blob/main/config.json) | [Apache 2.0](https://huggingface.co/Salesforce/codegen25-7b-multi/blob/main/README.md) | 
| DeciCoder-1B | 2023/08 | [DeciCoder-1B](https://huggingface.co/Deci/DeciCoder-1b#how-to-use) | [Introducing DeciCoder: The New Gold Standard in Efficient and Accurate Code Generation](https://deci.ai/blog/decicoder-efficient-and-accurate-code-generation-llm/) | 1.1 | [2048](https://huggingface.co/Deci/DeciCoder-1b#model-architecture) | Apache 2.0 |  [DeciCoder Demo](https://huggingface.co/spaces/Deci/DeciCoder-Demo)|
| Code Llama  | 2023 | [Inference Code for CodeLlama models]([https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://github.com/facebookresearch/codellama)) | [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)     | 7 - 34       | [4096](https://scontent-zrh1-1.xx.fbcdn.net/v/t39.2365-6/369856151_1754812304950972_1159666448927483931_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=wURKmnWKaloAX-ib5XW&_nc_ht=scontent-zrh1-1.xx&oh=00_AfAN1GB2K_XwIz54PqXTr-dhilI3CfCwdQoaLMyaYEEECg&oe=64F0A68F)  | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) Free if you have under 700M users and you cannot use LLaMA outputs to train other LLMs besides LLaMA and its derivatives   | [HuggingChat](https://huggingface.co/blog/codellama) | 


 ## Open LLM datasets for pre-training

| Name | Release Date | Paper/Blog | Dataset | Tokens (T) | License |
| --- | --- | --- | --- | --- | ---- | 
| starcoderdata | 2023/05 | [StarCoder: A State-of-the-Art LLM for Code](https://huggingface.co/blog/starcoder) | [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) |  0.25 | Apache 2.0 |
| RedPajama | 2023/04 | [RedPajama, a project to create leading open-source models, starts by reproducing LLaMA training dataset of over 1.2 trillion tokens](https://www.together.xyz/blog/redpajama) | [RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data) | 1.2 | Apache 2.0 | 

## Open LLM datasets for instruction-tuning

| Name | Release Date |  Paper/Blog | Dataset | Samples (K) | License |
| --- | --- | --- | --- | --- | ---- | 
| MPT-7B-Instruct | 2023/05 | [Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b) | [dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) | 59 | CC BY-SA-3.0 |
| databricks-dolly-15k | 2023/04 | [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |  [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15 |  CC BY-SA-3.0 |
| OIG (Open Instruction Generalist)   | 2023/03 | [THE OIG DATASET](https://laion.ai/blog/oig-dataset/) | [OIG](https://huggingface.co/datasets/laion/OIG) | 44,000 | Apache 2.0 |

## Open LLM datasets for alignment-tuning

| Name | Release Date |  Paper/Blog | Dataset | Samples (K) | License |
| --- | --- | --- | --- | --- | ---- |
| OpenAssistant Conversations Dataset | 2023/04 | [OpenAssistant Conversations - Democratizing Large Language Model Alignment](https://drive.google.com/file/d/10iR5hKwFqAKhL3umx8muOWSRm7hs5FqX/view) | [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 161 | Apache 2.0 |

## Evals on open LLMs

- [Leaderboard by lmsys.org](https://chat.lmsys.org/?leaderboard)
- [Evals by MosaicML](https://twitter.com/jefrankle/status/1654631746506301441)
- [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/latest/?groups=1)
- [LLM-Leaderboard](https://github.com/LudwigStumpp/llm-leaderboard)
- [TextSynth Server Benchmarks](https://bellard.org/ts_server/)
- [Open LLM Leaderboard by Hugging Face](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

### What do the licences mean?

- [Apache 2.0](https://en.wikipedia.org/wiki/Apache_License): Allows users to use the software for any purpose, to distribute it, to modify it, and to distribute modified versions of the software under the terms of the license, without concern for royalties.
- [MIT](https://en.wikipedia.org/wiki/MIT_License): Similar to Apache 2.0 but shorter and simpler. Also, in contrast to Apache 2.0, does not require stating any significant changes to the original code.
- [CC BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/): Allows (i) copying and redistributing the material and (ii) remixing, transforming, and building upon the material
for any purpose, even commercially. But if you do the latter, you **must distribute your contributions under the same license as the original.** (Thus, may not be viable for internal teams.)
- [OpenRAIL-M v1](https://www.bigcode-project.org/docs/pages/model-license/): Allows royalty-free access and flexible downstream use and sharing of the model and modifications of it, and comes with a set of use restrictions (see [Attachment A](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement))
- [BSD-3-Clause](https://en.wikipedia.org/wiki/BSD_licenses): This version allows unlimited redistribution for any purpose as long as its copyright notices and the license's disclaimers of warranty are maintained. 

**Disclaimer:** The information provided in this repo does not, and is not intended to, constitute legal advice. Maintainers of this repo are not responsible for the actions of third parties who use the models. Please consult an attorney before using models for commercial purposes.

---

### Improvements

- [x] Complete entries for context length, and check entries with `?`
- [ ] ~~Add number of tokens trained?~~ (see [considerations](https://github.com/eugeneyan/open-llms/issues/7))
- [ ] Add (links to) training code?
- [ ] Add (links to) eval benchmarks?
