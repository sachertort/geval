# G-Eval for evaluation of empathetic chatbots
Qualities were taken from [(Svikhnushina et al., 2022)](https://aclanthology.org/2022.sigdial-1.41). See the original instructions [here](https://github.com/Sea94/ieval).

## Usage
1. Put your data in the folder `data/freud`.
2. Set the arguments and run the following:
    ```
    python geval_freud.py --data_fp data/freud/<filename> --save_fp results/freud/<filename> --key <OpenAI_API_key>
    ```
    Required arguments:
    * `data_fp`: file path to data.
    * `save_fp`: file path to results; `<filename>_aggregated` will contain aggregated scores across each quality.
    * `key`: OpenAI API key.  
    
    Optional arguments:
    * `prompt_dir_fp`: path to directory with evaluation prompts (one quality — one prompt - one file); `prompts/freud` by default.
    * `model`: OpenAI model tag; probably, the script needs to be adapted for GPT-4; `gpt-3.5-turbo` by default.

## Original `README`: Code for paper "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" [https://arxiv.org/abs/2303.16634]

### Experiments on SummEval dataset
#### Evaluate fluency on SummEval dataset
```python .\gpt4_eval.py --prompt .\prompts\summeval\flu_detailed.txt --save_fp .\results\gpt4_flu_detailed.json --summeval_fp .\data\summeval.json --key XXXXX```

#### Meta Evaluate the G-Eval results

```python .\meta_eval_summeval.py --input_fp .\results\gpt4_flu_detailed.json --dimension fluency```

### Prompts and Evaluation Results

Prompts used to evaluate SummEval are in prompts/summeval

G-eval results on SummEval are in results

