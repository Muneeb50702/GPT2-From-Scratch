"""Character-level tokenizer for Tiny Shakespeare and small language-model experiments.

Mechanistic interpretability note:
    Character tokenization keeps the token-to-text mapping maximally transparent.
    This reduces confounders introduced by subword tokenization and helps when
    analyzing attention circuits at early stages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenizerState:
    stoi: dict[str, int]
    itos: dict[int, str]


class CharTokenizer:
    """Simple character-level tokenizer with deterministic vocabulary ordering.

    The vocabulary is sorted so runs are reproducible and checkpoints can be compared.
    """

    def __init__(self) -> None:
        self._state: TokenizerState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def vocab_size(self) -> int:
        self._require_fitted()
        return len(self._state.stoi)

    @property
    def stoi(self) -> dict[str, int]:
        self._require_fitted()
        return self._state.stoi

    @property
    def itos(self) -> dict[int, str]:
        self._require_fitted()
        return self._state.itos

    def fit(self, text: str) -> None:
        """Build vocabulary from text.

        Args:
            text: Entire corpus text.
        """
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text.")

        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        self._state = TokenizerState(stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""
        self._require_fitted()
        unknown = sorted(set(ch for ch in text if ch not in self._state.stoi))
        if unknown:
            raise ValueError(f"Encountered unknown characters: {unknown}")
        return [self._state.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back to text."""
        self._require_fitted()
        bad = [tok for tok in tokens if tok not in self._state.itos]
        if bad:
            raise ValueError(f"Encountered out-of-vocab token ids: {bad[:10]}")
        return "".join(self._state.itos[tok] for tok in tokens)

    def state_dict(self) -> dict[str, dict]:
        """Export tokenizer state for checkpointing."""
        self._require_fitted()
        return {"stoi": self._state.stoi, "itos": self._state.itos}

    def load_state_dict(self, state: dict[str, dict]) -> None:
        """Restore tokenizer state from checkpoint."""
        stoi = state.get("stoi")
        itos = state.get("itos")
        if not isinstance(stoi, dict) or not isinstance(itos, dict):
            raise ValueError("Invalid tokenizer state dict.")
        self._state = TokenizerState(stoi=stoi, itos={int(k): v for k, v in itos.items()})

    def _require_fitted(self) -> None:
        if self._state is None:
            raise RuntimeError("Tokenizer is not fitted. Call fit(text) first.")
