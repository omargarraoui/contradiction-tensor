# Contradiction Tensor

A data structure that maps the internal contradictions in a belief system.

Most tools track *what you believe*. This one tracks *where your beliefs contradict each other* — and by how much, under different conditions.

---

## Concept

Every belief you hold is a node. Every pair of beliefs has a contradiction score in `[0, 1]` computed by a local NLI model — no API, no cost, runs on your laptop.

The result is an **N × N × K tensor** where K is the number of market contexts you care about (e.g. default, crisis, bull market). The same pair of beliefs can have very different contradiction scores depending on the regime.

```
         fed_hawk   risk_on   dlr_str   em_bull
fed_hawk    —         0.87      0.05      0.61
risk_on    0.87        —        0.43      0.12
dlr_str    0.05       0.43       —        0.74
em_bull    0.61       0.12      0.74       —
```

That `0.87` between `fed_hawk` and `risk_on` tells you: you're holding two beliefs that the model reads as logically incompatible. You may not have noticed. Now you have.

---

## Structure

```
contradiction-tensor/
├── nli.py       # local NLI scorer (cross-encoder/nli-deberta-v3-small)
├── tensor.py    # ContradictionTensor and Belief
├── viz.py       # heatmap export
├── main.py      # interactive CLI
└── requirements.txt
```

### `Belief`
```python
@dataclass
class Belief:
    id: str
    text: str
    domain: str = "general"
```

### `ContradictionTensor`

| Method | Description |
|---|---|
| `add(belief, context_weights)` | Add a belief; scores against all existing ones |
| `max_contradiction(context)` | Returns the highest-tension pair |
| `min_removal_set(threshold)` | Greedy 2-approx of Minimum Vertex Cover — min beliefs to drop for internal consistency |
| `tension_clusters(threshold)` | Union-Find grouping of contradicting beliefs |
| `summary()` | Mean, max, p90 tension across all pairs |
| `matrix` | N × N view of the default context slice |

---

## Model

`cross-encoder/nli-deberta-v3-small` — 184M parameters, runs on CPU in ~200ms per pair. First run downloads ~700MB. After that it's fully offline.

Contradiction is scored **symmetrically**: the average of premise→hypothesis and hypothesis→premise. NLI is directional, contradiction intuition is not.

---

## Install

```bash
pip install -r requirements.txt
python main.py
```

---

## Usage

```
> add
  belief: The Fed will keep rates high for longer
  domain: macro
  crisis weight [0-2]: 1.5
  bull market weight [0-2]: 0.5

> add
  belief: Now is a good time to be long risk assets
  ...

> max
  0.871  ████████████████
  → The Fed will keep rates high for longer  (fed_hav_3a1f)
  → Now is a good time to be long risk assets  (now_is_9c2d)

> heatmap
  output path: heatmap.png
  saved → heatmap.png

> demo          # loads 5 pre-built macro beliefs to test immediately
```

---

## Key Operations

**`min_removal_set(threshold=0.7)`**

Answers: *which beliefs should I drop to make my view internally consistent?*

Uses a greedy approximation of Minimum Vertex Cover — at each step removes the belief with the most high-tension edges. NP-hard exactly, 2-approximation guarantee here.

**`tension_clusters(threshold=0.6)`**

Groups beliefs that are transitively connected by contradiction above a threshold. If A contradicts B and B contradicts C, all three land in the same cluster even if A and C score low directly.

**Context weights**

When adding a belief you set a multiplier per context. A belief like *"liquidity dries up fast"* might be neutral in calm markets (weight 1.0) but highly activating in a crisis (weight 2.0). This lets the tensor model regime-dependent coherence — the same thesis can be internally consistent in one regime and self-contradictory in another.

---

## Limitations

- NLI models score *semantic* contradiction, not logical entailment. "The sky is green" vs "The sky is blue" scores high; subtle logical contradictions in domain language may score lower than expected.
- `min_removal_set` is a greedy approximation. For N > ~50, consider switching to an LP relaxation.
- Scores are not calibrated probabilities. Use them comparatively, not in absolute terms.
