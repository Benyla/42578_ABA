import pytest

from aba_rfdetr.inference import reset_model_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_inference_cache() -> None:
    reset_model_cache_for_tests()
    yield
    reset_model_cache_for_tests()
