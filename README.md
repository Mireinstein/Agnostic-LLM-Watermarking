# Prompt-Based and Token-Level Watermarking for LLM Output

Aime Cesaire Mugishawayo, Miro Babin, Admire Madyira

Two ways of marking LLM-generated text so it can be identified later, built and
measured against each other on the same prompts.

**Approach A — prompt-level.** Lexical and stylistic constraints are injected
into the prompt, so the model's own output carries a hidden pattern. The
watermark lives in what the text says.

**Approach B — token-level.** A greenlist of tokens is favoured during sampling
via a logit bias, after [Kirchenbauer et al.
(2023)](https://arxiv.org/abs/2301.10226). The watermark is a statistical skew
in word choice rather than anything a reader would notice.

## Results

RoBERTa-base fine-tuned as a binary detector, against TF-IDF + logistic
regression baselines. Test accuracy, held-out split:

| Detector | Prompt-level watermark | Token-level watermark |
|---|---|---|
| TF-IDF, word 1–2 gram | 59.4% | 28.9% |
| TF-IDF, char 3–5 gram | 64.5% | — |
| TF-IDF, BPE | 64.8% | — |
| **RoBERTa-base** | **86.5%** | **65.6%** |

Prompt-level watermarks are far easier for a learned classifier to find. That
is the expected direction rather than a surprise: a semantic watermark changes
what the text says, which is what a text classifier is built to notice, while
a greenlist watermark is a frequency skew over token choice — a z-test on
greenlist rate detects it better than RoBERTa does. The 65.6% is therefore a
statement about the detector, not about the watermark's strength.

The word-level TF-IDF result on token watermarks (28.9%, below chance) reflects
the same thing from the other side: bag-of-words features latch onto topic, and
the greenlist skew is invisible to them.

## Corpus

2,000 prompts drawn from OpenAI Evals and the WikiText-103 validation set, with
three generations each, all from `gpt-3.5-turbo`:

| Condition | Responses | How |
|---|---|---|
| Control | 2,000 | Unmodified prompt |
| Prompt-watermarked | 1,913 | Rules injected via `PromptWrapper` |
| Token-watermarked | 1,999 | Static greenlist through the OpenAI `logit_bias` parameter |

Both watermarked conditions use `gpt-3.5-turbo`. The original plan called for
a local Llama-2-7B for Approach B; the greenlist is applied through the API's
logit bias instead, which constrains it to a small fixed token set rather than
the keyed pseudorandom partition in the paper.

## Layout

    prompts/                  prompt curation and sampling
    prompt_level/             Approach A: rule injection and detection
    token_level/              Approach B: greenlist generation
    vanilla_responses_gpt3.5turbo/   control corpus
    watermarked_responses/    both watermarked corpora
    classification/           detector training and evaluation notebooks

## Running it

    pip install -r requirements.txt
    export OPENAI_API_KEY=...

    python prompts/take_2000.py
    python vanilla_responses_gpt3.5turbo/generate_responses.py
    python prompt_level/generate_prompt_watermark.py
    python token_level/generate_watermarked.py

Then open the notebooks in `classification/` to train and evaluate the
detectors. The generated corpora are committed, so the notebooks run without
regenerating anything.

## References

1. [A Watermark for Large Language Models — Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226)
2. [Scalable Watermarking for Identifying Large Language Model Outputs — Lester et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499265/)
3. [Watermarking Language Models through Language Models — Zhong et al. (2024)](https://arxiv.org/abs/2411.05091)
4. [A Robust Semantics-Based Watermark for LLMs Against Paraphrasing — Ren et al. (2023)](https://arxiv.org/abs/2311.08721)
5. [Topic-Based Watermarks for Large Language Models — Wang et al. (2024)](https://arxiv.org/abs/2404.02138)
