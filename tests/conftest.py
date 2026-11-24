import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="Run live LLM smoke tests (needs GEMINI_API_KEY or the key env named by QBQ_LIVE_KEY_ENV)")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a real LLM API; only runs with --live and an API key")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live LLM test: pass --live (and set GEMINI_API_KEY) to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
