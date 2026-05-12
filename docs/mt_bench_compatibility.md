# MT-Bench Compatibility

We use PEPO for answer generation and a local vLLM judge, but the benchmark
logic is kept close to FastChat MT-Bench.

FastChat is pinned at commit `587d5cfa1609a43d192cedb8441cac3c17db105d`:
<https://github.com/lm-sys/FastChat/tree/587d5cfa1609a43d192cedb8441cac3c17db105d>

The parity tests check that:

- bundled MT-Bench questions, judge prompts, and GPT-4 reference answers match
  the pinned FastChat files;
- prompt selection and formatted judge prompts match FastChat for representative
  MT-Bench categories and both turns;
- score/verdict parsing follows FastChat behavior;
- pairwise two-game position swapping and leaderboard aggregation match
  FastChat's `llm_judge` code.

Intentional differences:

- answers are generated through PEPO `BaseModel.generate_responses`;
- judgments are produced by `ManagedVLLMJudge` instead of OpenAI/Anthropic APIs;
- outputs use PEPO's evaluator output layout.

Run the parity checks with:

```bash
git submodule update --init --recursive FastChat
uv run pytest tests/evaluator/test_mtbench_fastchat_compat.py
```
