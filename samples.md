# Sample PDF Downloads for RAGAS Testset Generation

```python
curl -L --ssl-no-revoke https://arxiv.org/pdf/2505.10468.pdf -o data/
ai_agents_vs_agentic_ai_2505.10468.pdf

curl -L --ssl-no-revoke https://arxiv.org/pdf/2505.06913.pdf -o data/
redteamllm_agentic_ai_framework_2505.06913.pdf

curl -L --ssl-no-revoke https://arxiv.org/pdf/2505.06817.pdf -o data/
control_plane_scalable_design_pattern_2505.06817.pdf
```

This file contains curl commands to download recent research papers about AI agents and agentic AI systems from arXiv. These papers will serve as input documents for generating the RAGAS testset, which is used to evaluate retrieval-augmented generation systems. The downloaded PDFs will be stored in the `data/` directory as configured in the project settings.