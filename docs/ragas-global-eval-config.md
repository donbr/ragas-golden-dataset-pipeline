# Leveraging Global Configuration for RAGAS Evaluations

## define global settings

- define settings once at the start of your notebook or script
- `evaluator_config` settings shown below will minimize impacts of LLM rate limiting and let you run a stronger evaluation model (such as `gpt-4o`)
- to make your comparisons more accurate you should use the same model for all evaluations (baseline, fine-tuned, etc.)

```python
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)

## set global evaluation settings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
evaluator_config = RunConfig(
    timeout=300,          # 5 minutes max for operations
    max_retries=15,       # More retries for rate limits
    max_wait=90,          # Longer wait between retries
    max_workers=8,        # Fewer concurrent API calls
    log_tenacity=True     # Log retry attempts
)
evaluator_metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()]
```

## Evaluate baseline evaluation dataset with RAGAS

```python
baseline_evaluation_results = evaluate(
    dataset=evaluation_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```

## Evaluate fine-tuned dataset with RAGAS

```python
finetuned_evaluation_results = evaluate(
    dataset=ft_dataset,
    metrics=evaluator_metrics,
    llm=evaluator_llm,
    run_config=evaluator_config
)
```