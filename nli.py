import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIScorer:
    MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.MODEL)
            .to(self.device)
            .eval()
        )
        self._contradiction_idx = self._label_index("contradiction")

    def _label_index(self, target: str) -> int:
        for idx, label in self.model.config.id2label.items():
            if label.lower() == target:
                return idx
        raise ValueError(
            f"'{target}' not in model labels: {self.model.config.id2label}"
        )

    def _raw_score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return F.softmax(logits, dim=1)[0][self._contradiction_idx].item()

    def score(self, a: str, b: str) -> float:
        # symmetric: average both directions since NLI is directional
        return (self._raw_score(a, b) + self._raw_score(b, a)) / 2
