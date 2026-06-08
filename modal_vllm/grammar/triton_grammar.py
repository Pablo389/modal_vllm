from __future__ import annotations

from pathlib import Path

EBNF_PATH = Path(__file__).with_name("triton.ebnf")


def load_ebnf() -> str:
    return EBNF_PATH.read_text(encoding="utf-8")


def build_grammar_ebnf(*, fenced: bool = True) -> str:
    """Return an EBNF string with the requested root rule for xgrammar."""
    ebnf = load_ebnf()
    if fenced:
        return ebnf.replace("root ::= fenced_module | program", "root ::= fenced_module")
    return ebnf.replace("root ::= fenced_module | program", "root ::= program")


def validate_ebnf(ebnf: str, *, root_rule_name: str = "root") -> None:
    """Compile-check an EBNF grammar with xgrammar."""
    try:
        import xgrammar as xgr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgrammar is required to validate EBNF grammars. "
            "Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc

    xgr.Grammar.from_ebnf(ebnf, root_rule_name=root_rule_name)
