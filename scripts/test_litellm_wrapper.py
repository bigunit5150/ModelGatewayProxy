# scripts/test_litellm_wrapper.py
import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)  # must run before litellm is imported so it sees ANTHROPIC_API_KEY

from llmgateway.providers.litellm_wrapper import LLMGatewayProvider  # noqa: E402
from llmgateway.providers.models import CompletionRequest  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"


async def test_anthropic_streaming():
    """Test Anthropic streaming with LiteLLM wrapper"""
    provider = LLMGatewayProvider()

    request = CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Count to 5"}],
        temperature=0.7,
        stream=True,
    )

    print("Testing Anthropic streaming...")
    async for chunk in provider.generate(request):
        print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            print(f"\nFinish reason: {chunk.finish_reason}")
        if chunk.usage:
            print(f"Tokens: {chunk.usage}")
    print()


async def test_anthropic_non_streaming():
    """Test Anthropic non-streaming with LiteLLM wrapper"""
    provider = LLMGatewayProvider()

    request = CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0,
        stream=False,
    )

    print("Testing Anthropic non-streaming...")
    async for chunk in provider.generate(request):
        print(f"Response: {chunk.content}")
        print(f"Tokens: {chunk.usage}")
    print()


async def test_error_handling():
    """Test error mapping with invalid API key"""
    real_key = os.environ.get("ANTHROPIC_API_KEY")

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-invalid-key-for-testing"

    provider = LLMGatewayProvider()
    request = CompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    print("Testing error handling (expect AuthError)...")
    try:
        async for chunk in provider.generate(request):
            print(chunk.content)
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}: {e}")

    if real_key:
        os.environ["ANTHROPIC_API_KEY"] = real_key
    print()


async def test_token_counting():
    """Test token counting functionality"""
    provider = LLMGatewayProvider()

    text = "Hello world, this is a test message"
    count = await provider.count_tokens(text, MODEL)

    print("Token counting test:")
    print(f"  Text: '{text}'")
    print(f"  Token count: {count}")
    print()


async def main():
    print("=" * 60)
    print("LiteLLM Wrapper Manual Testing")
    print("=" * 60)
    print()

    await test_anthropic_streaming()
    await test_anthropic_non_streaming()
    await test_token_counting()
    await test_error_handling()

    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
