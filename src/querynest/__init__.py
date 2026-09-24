"""QueryNest: chat with your database."""


def main() -> None:
    """Quick setup check (`uv run querynest-check`): providers, settings (secrets hidden)."""
    from querynest.config import settings
    from querynest.llm import build_providers

    chain = [f"{p.name} ({p.model})" for p in build_providers()]
    print("LLM fallback chain:", " -> ".join(chain))
    print("Settings (secrets hidden):", settings)
