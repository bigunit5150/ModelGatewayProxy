# scripts/test_all_providers.py
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to the repo root so this script works from any cwd.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from llmgateway.providers.litellm_wrapper import LLMGatewayProvider  # noqa: E402
from llmgateway.providers.models import CompletionRequest

PROVIDERS_TO_TEST = [
    {"name": "OpenAI GPT-4o-mini", "model": "gpt-4o-mini", "required_env": "OPENAI_API_KEY"},
    {
        "name": "Anthropic Claude Haiku",
        "model": "claude-haiku-4-5-20251001",
        "required_env": "ANTHROPIC_API_KEY",
    },
    {
        "name": "Together Llama 3 70B",
        "model": "together/meta-llama/Llama-3-70b-chat-hf",
        "required_env": "TOGETHER_API_KEY",
    },
]


async def test_provider(provider_info: dict):
    """Test a single provider"""

    # Check if API key is available
    if not os.getenv(provider_info["required_env"]):
        print(f"⏭️  Skipping {provider_info['name']} (no API key)")
        return

    print(f"\n🧪 Testing {provider_info['name']}...")

    provider = LLMGatewayProvider()
    request = CompletionRequest(
        model=provider_info["model"],
        messages=[{"role": "user", "content": "Say 'Hello from LiteLLM!' in one sentence."}],
        temperature=0.7,
        stream=True,
    )

    try:
        print("   Response: ", end="")
        async for chunk in provider.generate(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.usage:
                print(f"\n   Tokens: {chunk.usage}")
        print(f"\n   ✅ {provider_info['name']} working!")

    except Exception as e:
        print(f"\n   ❌ Error: {type(e).__name__}: {e}")


async def main():
    print("=" * 60)
    print("Multi-Provider Test Suite")
    print("=" * 60)

    for provider_info in PROVIDERS_TO_TEST:
        await test_provider(provider_info)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
