import time
import pytest
import pytest_asyncio
import asyncio

from duckai import DuckAI

CHAT_MODELS = list(DuckAI._chat_models)[0:1]


@pytest.fixture(autouse=True)
def pause_between_tests() -> None:
    time.sleep(2)


def test_context_manager() -> None:
    with DuckAI(proxy="socks5h://192.168.50.1:23456") as duckai:
        results = duckai.chat("cars")
        assert len(results) >= 1


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_chat(model: str) -> None:
    results = DuckAI(proxy="socks5h://192.168.50.1:23456").chat("cat", model=model)
    assert  len(results) >= 1


# Async context manager fixture
@pytest_asyncio.fixture
async def async_duckai():
    async with DuckAI(proxy="socks5h://192.168.50.1:23456") as duckai:
        yield duckai


# Test async context manager
@pytest.mark.asyncio
async def test_async_context_manager(async_duckai) -> None:
    results = await async_duckai.chat_async("cars")
    assert len(results) >= 1


# Test chat_async
@pytest.mark.asyncio
@pytest.mark.parametrize("model", CHAT_MODELS)
async def test_chat_async(model: str) -> None:
    duckai = DuckAI(proxy="socks5h://192.168.50.1:23456")
    results = await duckai.chat_async("cat", model=model)
    assert len(results) >= 1


# Test chat_yield_async
@pytest.mark.asyncio
@pytest.mark.parametrize("model", CHAT_MODELS)
async def test_chat_yield_async(model: str) -> None:
    duckai = DuckAI(proxy="socks5h://192.168.50.1:23456")
    chunks = []
    async for chunk in duckai.chat_yield_async("dog", model=model):
        chunks.append(chunk)
    
    # Assert we got at least one chunk
    assert len(chunks) >= 1
    
    # Assert the combined result is not empty
    combined_result = "".join(chunks)
    assert len(combined_result) >= 1


# Test multiple concurrent requests
@pytest.mark.asyncio
async def test_concurrent_requests() -> None:
    duckai = DuckAI(proxy="socks5h://192.168.50.1:23456")
    
    # Create multiple tasks
    tasks = [
        duckai.chat_async("What is Python?", model="gpt-4o-mini"),
        duckai.chat_async("What is asyncio?", model="gpt-4o-mini"),
        duckai.chat_async("Tell me about cats", model="gpt-4o-mini")
    ]
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    
    # Check each result
    for result in results:
        assert len(result) >= 1