SOME: Sparse Organic Modular Experts

A lightweight experimental MoE-style architecture with emergent specialization, built in JAX.

SOME is an architecture designed to explore organic specialization in sparse models.
Instead of enforcing balance or rigid expert routing, SOME allows experts to self-organize over training, forming roles naturally through gradient pressure and task structure.

This repo contains:
	•	A JAX/Flax implementation of the SOME block
	•	A tiny transformer using SOME as its feedforward module
	•	Training utilities
	•	A Palette-based TUI for visualizing expert behavior
	•	Early toy-task experiments showing specialization
	•	A growing framework for LM training and benchmarking

⸻

✨ Key Ideas

1. Organic Specialization

Experts are not assigned fixed roles.
The router selects experts based on token-level context, and over time experts self-organize into functional groups (syntax, math, structure, etc.).

2. Sparse Activation

Only a subset of experts run per token, giving:
	•	higher effective capacity
	•	lower compute cost
	•	reduced interference between subskills

3. Simple & Interpretable Routing

The router is a small MLP that outputs logits over experts.
Top-k routing (or threshold routing) picks the active experts.
No balancing loss is used by default — specialization emerges naturally.

4. Lightweight Research Platform

The codebase is intentionally small and hackable:
	•	easy to modify routing
	•	easy to add experts
	•	easy to scale depth/width
	•	ideal for experimentation on a single GPU

🧠 Roadmap

Phase 1 (done / in progress)
	•	Implement SOME block
	•	Build routing mechanism
	•	Toy dataset specialization
	•	TUI visualization

Phase 2
	•	Add tokenizer (GPT2 or SentencePiece)
	•	Train tiny language models (5M–20M params)
	•	Observe expert specialization on text
	•	Add dataset loaders (WikiText, small Wikipedia subsets)

Phase 3
	•	Scale to 150M parameters
	•	Train on mixed text + code
	•	Evaluate on:
	•	HLE (Humanity’s Last Exam)
	•	ARC-AGI (partial)
	•	HumanEval / MBPP
	•	AI “IQ” pattern tests

Phase 4
	•	Add tool use (MCP)
	•	Extend architecture (hierarchical experts, gated clusters)
	•	Write the research paper
	•	Release models + benchmarks

⸻

📊 Goals

SOME is not built to beat trillion-dollar labs.
Instead, its goals are:
	•	Show emergent specialization in a tiny sparse model
	•	Match or beat dense models 2–5× larger on structured tasks
	•	Provide a transparent research platform for sparse routing
	•	Inspire further exploration of decentralized expert behavior

⸻

📝 License

Choose whatever you prefer (MIT, Apache-2.0, GPL, etc.).

⸻

👥 Contributions

Open to:
	•	researchers
	•	engineers
	•	students
	•	people interested in sparse models

PRs welcome once the repo stabilizes.

⸻

🙏 Acknowledgments

Inspired by:
	•	Mixture-of-Experts literature
	•	GShard, Switch Transformer, Mixtral
	•	Sparse architectures and routing research

But SOME is intentionally not a direct copy of any of them.
