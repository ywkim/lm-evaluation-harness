# 언어 모델 평가 도구

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## 공지사항
**lm-evaluation-harness v0.4.0 버전이 출시되었습니다!**

새로운 업데이트와 기능은 다음과 같습니다:

- 내부 리팩토링
- 설정 기반의 작업 생성 및 구성
- 외부에서 정의된 작업 구성 YAML 파일의 더 쉬운 가져오기와 공유
- Jinja2 프롬프트 설계 지원, 프롬프트 수정 용이, Promptsource로부터 프롬프트 가져오기 
- 출력 후처리, 답변 추출, 문서당 다중 LM 생성을 포함한 고급 구성 옵션, 구성 가능한 fewshot 설정 등
- 속도 향상 및 새로운 모델링 라이브러리 지원: HF 모델의 빠른 데이터 병렬 사용, vLLM 지원, HuggingFace의 MPS 지원 등
- 로깅 및 사용성 변경사항  
- CoT BIG-Bench-Hard, Belebele, 사용자 정의 작업 그룹화 등 새로운 작업 추가

자세한 내용은 `docs/`의 업데이트된 문서 페이지를 참조하세요.

개발은 `main` 브랜치에서 계속될 예정이며, 어떤 기능이 필요한지, 라이브러리를 어떻게 더 개선할 수 있을지에 대한 피드백을 GitHub의 이슈나 PR로, 또는 [EleutherAI 디스코드](https://discord.gg/eleutherai)에서 주시기 바랍니다!

## 개요

이 프로젝트는 많은 다양한 평가 작업에 대해 생성 언어 모델을 테스트하기 위한 통합 프레임워크를 제공합니다.

**주요 기능:**
- LLM을 위한 60개 이상의 표준 학술 벤치마크와 수백 개의 하위 작업 및 변형 구현.
- [transformers](https://github.com/huggingface/transformers/) ([AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)를 통한 양자화 포함), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/)를 통해 로드된 모델 지원. 유연한 토크나이제이션 독립 인터페이스 제공.
- [vLLM](https://github.com/vllm-project/vllm)을 사용한 빠르고 메모리 효율적인 추론 지원.
- [OpenAI](https://openai.com), [TextSynth](https://textsynth.com/) 등 상용 API 지원.  
- [HuggingFace의 PEFT 라이브러리](https://github.com/huggingface/peft)에서 지원되는 어댑터(예: LoRA)에 대한 평가 지원.
- 로컬 모델 및 벤치마크 지원.
- 공개적으로 사용 가능한 프롬프트를 사용한 평가로 재현성과 논문 간 비교 가능성 보장.
- 사용자 지정 프롬프트 및 평가 메트릭에 대한 쉬운 지원.

언어 모델 평가 도구는 유명한 🤗 Hugging Face의 [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)의 백엔드이며, [수백 편의 논문](https://scholar.google.com/scholar?oi=bibs&hl=en&authuser=2&cites=15052937328817631261,4097184744846514103,1520777361382155671,17476825572045927382,18443729326628441434,14801318227356878622,7890865700763267262,12854182577605049984,15641002901115500560,5104500764547628290)에 사용되었고, NVIDIA, Cohere, BigScience, BigCode, Nous Research, Mosaic ML 등 수십 개 조직에서 내부적으로 사용되고 있습니다.

## 설치

Github 저장소에서 `lm-eval` 패키지를 설치하려면 다음을 실행하세요:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness 
cd lm-evaluation-harness
pip install -e .
```

확장 기능을 위해 다양한 선택적 종속성도 제공됩니다. 자세한 목록은 이 문서 끝 부분에 있습니다.  

## 기본 사용법

### Hugging Face `transformers`

[HuggingFace Hub](https://huggingface.co/models)에 호스팅된 모델(예: GPT-J-6B)을 `hellaswag`에서 평가하려면 다음 명령을 사용할 수 있습니다(CUDA 호환 GPU를 사용한다고 가정):

```bash 
lm_eval --model hf \
--model_args pretrained=EleutherAI/gpt-j-6B \
--tasks hellaswag \
--device cuda:0 \
--batch_size 8
```

추가 인자는 `--model_args` 플래그를 사용하여 모델 생성자에게 직접 제공될 수 있습니다. 특히 Hub의 `revisions` 기능을 사용하여 부분적으로 학습된 체크포인트를 저장하거나 모델 실행을 위한 데이터 유형을 지정하는 일반적인 관행을 지원합니다:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8  
```

Huggingface에서 `transformers.AutoModelForCausalLM`(자기회귀, 디코더 전용 GPT 스타일 모델)과 `transformers.AutoModelForSeq2SeqLM`(T5와 같은 인코더-디코더 모델)을 통해 로드된 모델이 지원됩니다. 

배치 크기 선택은 ```--batch_size``` 플래그를 ```auto```로 설정하여 자동화할 수 있습니다. 이렇게 하면 장치에 맞는 가장 큰 배치 크기를 자동으로 감지합니다. 가장 긴 예제와 가장 짧은 예제 간에 큰 차이가 있는 작업에서는 가장 큰 배치 크기를 주기적으로 재계산하여 속도를 더 높일 수 있습니다. 이렇게 하려면 위 플래그에 ```:N```을 추가하여 가장 큰 배치 크기를 ```N```번 자동으로 다시 계산하세요. 예를 들어 배치 크기를 4번 재계산하려면 명령은 다음과 같습니다:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4
```

지원되는 전체 인자 목록은 [여기](./docs/interface.md)와 터미널에서 `lm_eval -h`를 호출하여 제공됩니다. 또는 `lm-eval` 대신 `lm_eval`을 사용할 수 있습니다.

> [!Note]
> `transformers.AutoModel`에 로컬 경로를 제공할 수 있는 것처럼 `lm_eval`에도 `--model_args pretrained=/path/to/model`을 통해 로컬 경로를 제공할 수 있습니다.

#### Hugging Face `accelerate`를 사용한 다중 GPU 평가

[accelerate 🚀](https://github.com/huggingface/accelerate) 라이브러리를 사용한 다중 GPU 평가를 위해 두 가지 주요 방법을 지원합니다.

*데이터 병렬 평가*를 수행하려면(각 GPU가 모델의 **별도의 전체 복사본**을 로드), 다음과 같이 `accelerate` 런처를 활용합니다:

```
accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
``` 
(또는 `accelerate launch --no-python lm_eval`을 통해).

모델이 단일 GPU에 맞는 경우 K개의 GPU에서 1개에서보다 K배 더 빠르게 평가할 수 있습니다.

**경고**: 이 설정은 FSDP 모델 샤딩과 작동하지 않으므로 `accelerate config`에서 FSDP를 비활성화하거나 NO_SHARD FSDP 옵션을 사용해야 합니다.

`accelerate`를 사용한 다중 GPU 평가의 두 번째 방법은 모델이 *단일 GPU에 맞지 않을 만큼 큰 경우*입니다.

이 설정에서는 `accelerate` 런처 *외부*에서 라이브러리를 실행하되 `--model_args`에 `parallelize=True`를 전달합니다:

```
lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=True \
    --batch_size 16 
```

이는 모델 가중치가 사용 가능한 모든 GPU에 분할됨을 의미합니다.  

고급 사용자 또는 더 큰 모델의 경우 `parallelize=True`일 때 다음 인자를 허용합니다:
- `device_map_option`: 사용 가능한 GPU에 모델 가중치를 분할하는 방법. 기본값은 "auto".
- `max_memory_per_gpu`: 모델 로드 시 GPU당 사용할 최대 GPU 메모리.  
- `max_cpu_memory`: 모델 가중치를 RAM에 오프로드할 때 사용할 최대 CPU 메모리.
- `offload_folder`: 필요한 경우 모델 가중치를 디스크에 오프로드할 폴더.

이 두 옵션(`accelerate launch`와 `parallelize=True`)은 상호 배타적입니다.

**참고: 현재 다중 노드 평가는 기본적으로 지원하지 않으며, 추론 요청을 실행하기 위해 외부에서 호스팅된 서버를 사용하거나 [GPT-NeoX 라이브러리에서 수행한 것처럼](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py) 분산 프레임워크와 사용자 지정 통합을 만드는 것을 권장합니다.**

### `vLLM`을 사용한 텐서 + 데이터 병렬 및 최적화된 추론

[지원되는 모델 유형](https://docs.vllm.ai/en/latest/models/supported_models.html), 특히 단일 GPU 또는 다중 GPU에서 모델을 분할할 때 더 빠른 추론을 위해 vLLM도 지원합니다. 단일 GPU 또는 다중 GPU - 텐서 병렬, 데이터 병렬 또는 둘 모두의 조합 - 추론의 경우 예를 들면:

```bash 
lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto
```

vllm을 사용하려면 `pip install lm_eval[vllm]`을 실행하세요. 지원되는 vLLM 구성의 전체 목록은 [vLLM 통합](https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075bce/lm_eval/models/vllm_causallms.py)과 vLLM 문서를 참조하세요.

vLLM은 때때로 Huggingface와 다른 출력을 생성합니다. Huggingface를 참조 구현으로 취급하고 [스크립트](./scripts/model_comparator.py)를 제공하여 HF에 대해 vllm 결과의 유효성을 확인할 수 있습니다.

### 모델 API 및 추론 서버 

또한 우리의 라이브러리는 여러 상용 API를 통해 제공되는 모델의 평가를 지원하며, 가장 일반적으로 사용되는 고성능 로컬/자체 호스팅 추론 서버에 대한 지원을 구현하기를 희망합니다.

호스팅된 모델을 호출하려면 다음을 사용하세요:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args model=davinci \
    --tasks lambada_openai,hellaswag
```

OpenAI Completions 및 ChatCompletions API를 미러링하는 서버로 고유한 로컬 추론 서버 사용도 지원합니다.

```bash
lm_eval --model local-chat-completions --tasks gsm8k --model_args model=facebook/opt-125m,base_url=http://{yourip}:8000/v1
```
외부에서 호스팅되는 모델의 경우 `--device` 및 `--batch_size`와 같은 구성은 사용해서는 안 되며 작동하지 않습니다. 로컬 모델의 모델 생성자에 임의의 인자를 전달하기 위해 `--model_args`를 사용할 수 있는 것처럼, 호스팅된 모델의 모델 API에 임의의 인자를 전달하는 데에도 사용할 수 있습니다. 지원되는 인자에 대한 정보는 호스팅 서비스 문서를 참조하세요.

| API 또는 추론 서버                                                                                                        | 구현 여부                       | `--model <xxx>` 이름                                                | 지원되는 모델:                                                                                | 요청 유형:                                                 |
|---------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------|
| OpenAI Completions                                                                                                        | :heavy_check_mark:              | `openai-completions`, `local-completions` | 모든 OpenAI Completions API 모델                                           | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| OpenAI ChatCompletions                                                                                                    | :heavy_check_mark:        | `openai-chat-completions`, `local-chat-completions`                                                               | [모든 ChatCompletions API 모델](https://platform.openai.com/docs/guides/gpt)                 | `generate_until` (logprobs 없음)                            |
| Anthropic                                                                                                                 | :heavy_check_mark:              | `anthropic`                                                         | [지원되는 Anthropic 엔진](https://docs.anthropic.com/claude/reference/selecting-a-model)  | `generate_until` (logprobs 없음)                             |
| Textsynth                                                                                                                 | :heavy_check_mark:                   | `textsynth`                                                         | [모든 지원 엔진](https://textsynth.com/documentation.html#engines)                     | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Cohere                                                                                                                    | [:hourglass: - Cohere API 버그로 차단됨](https://github.com/EleutherAI/lm-evaluation-harness/pull/395) | N/A                                                                 | [모든 `cohere.generate()` 엔진](https://docs.cohere.com/docs/models)                        | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) ([llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 통해) | :heavy_check_mark:              | `gguf`, `ggml`                                                      | [llama.cpp에서 지원되는 모든 모델](https://github.com/ggerganov/llama.cpp)                   | `generate_until`, `loglikelihood`, (perplexity 평가는 아직 구현되지 않음) |
| vLLM                                                                                                                      | :heavy_check_mark:       | `vllm`                                                              | [대부분의 HF 인과 언어 모델](https://docs.vllm.ai/en/latest/models/supported_models.html) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Mamba                       | :heavy_check_mark:       | `mamba_ssm`                                                                      | [Mamba 아키텍처 언어 모델 (`mamba_ssm` 패키지 통해)](https://huggingface.co/state-spaces) | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                             |
| Huggingface Optimum (인과 LM)    | ✔️         | `openvino`                                 |     Huggingface Optimum을 통해 OpenVINO™ Intermediate Representation(IR) 형식으로 변환된 모든 디코더 전용 AutoModelForCausalLM                         |  `generate_until`, `loglikelihood`, `loglikelihood_rolling`                         | ...                                                      |
| AWS Inf2를 통한 Neuron (인과 LM)    | ✔️         | `neuronx`                                 |     [huggingface-ami inferentia2 이미지](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2)에서 실행되도록 지원되는 모든 디코더 전용 AutoModelForCausalLM                         |  `generate_until`, `loglikelihood`, `loglikelihood_rolling`                         | ...                                                      |
| 여러분의 로컬 추론 서버!                                                                                              | :heavy_check_mark:                             | `local-completions` 또는 `local-chat-completions` (`openai-chat-completions` 모델 유형 사용)   | HF 모델을 사용하고 OpenAI의 Completions 또는 ChatCompletions 인터페이스를 미러링하는 GET 요청을 허용하는 모든 서버 주소                                  | `generate_until`                                           |                                | ...                |

logits 또는 logprobs를 제공하지 않는 모델은 `generate_until` 유형의 작업에만 사용할 수 있는 반면, 로컬 모델이나 프롬프트의 logprobs/logits를 제공하는 API는 모든 작업 유형(`generate_until`, `loglikelihood`, `loglikelihood_rolling`, `multiple_choice`)에서 실행할 수 있습니다.

서로 다른 작업 `output_types` 및 모델 요청 유형에 대한 자세한 내용은 [우리의 문서](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md#interface)를 참조하세요.

### 기타 프레임워크

GPT-NeoX, Megatron-DeepSpeed, mesh-transformer-jax 등 여러 라이브러리에는 해당 라이브러리를 통해 eval harness를 호출하는 스크립트가 포함되어 있습니다.

사용자 지정 통합을 생성하려면 [이 튜토리얼](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)의 지침을 따를 수 있습니다.

### 추가 기능 
> [!Note] 
> 신뢰할 수 없는 코드 실행과 관련된 위험이나 평가 프로세스의 복잡성으로 인해 직접 평가에 적합하지 않은 작업의 경우, 사후 평가를 위해 디코딩된 생성 결과를 얻으려면 `--predict_only` 플래그를 사용할 수 있습니다.

Metal 호환 Mac이 있다면 `--device cuda:0` 대신 `--device mps`를 사용하여 MPS 백엔드를 사용해 eval harness를 실행할 수 있습니다(PyTorch 버전 2.1 이상 필요).

> [!Note]
> 다음 명령을 실행하여 LM 입력이 어떻게 보이는지 검사할 수 있습니다:
> ```bash
> python write_out.py \
>     --tasks <task1,task2,...> \ 
>     --num_fewshot 5 \
>     --num_examples 10 \
>     --output_base_path /path/to/output/folder
> ```
> 이는 각 작업에 대해 하나의 텍스트 파일을 작성합니다.

작업 자체를 실행하는 것 외에도 수행 중인 작업의 데이터 무결성을 확인하려면 `--check_integrity` 플래그를 사용할 수 있습니다:

```bash
lm_eval --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## 고급 사용 팁

HuggingFace `transformers` 라이브러리로 로드된 모델의 경우 `--model_args`를 통해 제공되는 모든 인자는 관련 생성자에 직접 전달됩니다. 즉, `AutoModel`로 할 수 있는 모든 것을 우리 라이브러리로 할 수 있습니다. 예를 들어 `pretrained=`를 통해 로컬 경로를 전달하거나 `model_args` 인자에 `,peft=PATH`를 추가하여 기본 모델을 평가하는 호출을 실행하여 [PEFT](https://github.com/huggingface/peft)로 파인튜닝된 모델을 사용할 수 있습니다: 
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) 양자화 모델은 `model_args` 인자에서 `,autogptq=NAME`(또는 기본 이름의 경우 `,autogptq=True`)을 지정하여 파일 이름을 로드할 수 있습니다:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

작업 이름에 와일드카드를 지원합니다. 예를 들어 `--task lambada_openai_mt_*`를 통해 기계 번역된 모든 lambada 작업을 실행할 수 있습니다.

평가 결과를 저장하려면 `--output_path`를 제공하세요. 또한 `--log_samples` 플래그로 사후 분석을 위해 모델 응답을 기록할 수 있습니다.

또한 `--use_cache`로 디렉토리를 제공하여 이전 실행 결과를 캐시할 수 있습니다. 이를 통해 재채점을 위해 동일한 (모델, 작업) 쌍의 반복 실행을 피할 수 있습니다.

지원되는 인자의 전체 목록은 문서의 [인터페이스](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) 가이드를 확인하세요!

> [!Tip]
> lm-evaluation-harness를 외부 라이브러리로 실행하고 있고 사용 가능한 작업이 (거의) 없나요? `lm_eval.evaluate()` 또는 `lm_eval.simple_evaluate()`를 호출하기 전에 `lm_eval.tasks.initialize_tasks()`를 실행하여 라이브러리의 기본 작업을 로드하세요!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both Weights & Biases (W&B) and Zeno.

### Zeno

You can use [Zeno](https://zenoml.com) to visualize the results of your eval harness runs.

First, head to [hub.zenoml.com](https://hub.zenoml.com) to create an account and get an API key [on your account page](https://hub.zenoml.com/account).
Add this key as an environment variable:

```bash
export ZENO_API_KEY=[your api key]
```

You'll also need to install the `lm_eval[zeno]` package extra.

To visualize the results, run the eval harness with the `log_samples` and `output_path` flags.
We expect `output_path` to contain multiple folders that represent individual model names.
You can thus run your evaluation on any number of tasks and models and upload all of the results as projects on Zeno.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B
```

Then, you can upload the resulting data using the `zeno_visualize` script:

```bash
python scripts/zeno_visualize.py \
    --data_path output \
    --project_name "Eleuther Project"
```

This will use all subfolders in `data_path` as different models and upload all tasks within these model folders to Zeno.
If you run the eval harness on multiple tasks, the `project_name` will be used as a prefix and one project will be created per task.

You can find an example of this workflow in [examples/visualize-zeno.ipynb](examples/visualize-zeno.ipynb).

### Weights and Biases

With the [Weights and Biases](https://wandb.ai/site) integration, you can now spend more time extracting deeper insights into your evaluation results. The integration is designed to streamline the process of logging and visualizing experiment results using the Weights & Biases (W&B) platform.

The integration provide functionalities

- to automatically log the evaluation results,
- log the samples as W&B Tables for easy visualization,
- log the `results.json` file as an artifact for version control,
- log the `<task_name>_eval_samples.json` file if the samples are logged,
- generate a comprehensive report for analysis and visualization with all the important metric,
- log task and cli specific configs,
- and more out of the box like the command used to run the evaluation, GPU/CPU counts, timestamp, etc.

First you'll need to install the lm_eval[wandb] package extra. Do `pip install lm_eval[wandb]`.

Authenticate your machine with an your unique W&B token. Visit https://wandb.ai/authorize to get one. Do `wandb login` in your command line terminal.

Run eval harness as usual with a `wandb_args` flag. Use this flag to provide arguments for initializing a wandb run ([wandb.init](https://docs.wandb.ai/ref/python/init)) as comma separated string arguments.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples
```

In the stdout, you will find the link to the W&B run page as well as link to the generated report. You can find an example of this workflow in [examples/visualize-wandb.ipynb](examples/visualize-wandb.ipynb), and an example of how to integrate it beyond the CLI.

## How to Contribute or Learn More?

For more information on the library and how everything fits together, check out all of our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)! We plan to post a larger roadmap of desired + planned library improvements soon, with more information on how contributors can help.

### Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/new_task_guide.md).

In general, we follow this priority list for addressing concerns about prompting and other eval details:
1. If there is widespread agreement among people who train LLMs, use the agreed upon procedure.
2. If there is a clear and unambiguous official implementation, use that procedure.
3. If there is widespread agreement among people who evaluate LLMs, use the agreed upon procedure.
4. If there are multiple common implementations but not universal or widespread agreement, use our preferred option among the common implementations. As before, prioritize choosing from among the implementations found in LLM training papers.

These are guidelines and not rules, and can be overruled in special circumstances.

We try to prioritize agreement with the procedures used by other groups to decrease the harm when people inevitably compare runs across different papers despite our discouragement of the practice. Historically, we also prioritized the implementation from [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165) as our original goal was specifically to compare results with that paper.

### Support

The best way to get support is to open an issue on this repo or join the [EleutherAI Discord server](https://discord.gg/eleutherai). The `#lm-thunderdome` channel is dedicated to developing this project and the `#release-discussion` channel is for receiving support for our releases. If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!

## Optional Extras
Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| hf_transfer   | For speeding up HF Hub file downloads |
| ifeval        | For running the IFEval task           |
| neuronx       | For running on AWS inf2 instances     |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| optimum       | For running Intel OpenVINO models     |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| testing       | For running library test suite        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |

## Cite as

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```
