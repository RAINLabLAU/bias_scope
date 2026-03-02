from litellm import completion

from src.bias_scope.base import PromptBasedMetric


class BBQMetric(PromptBasedMetric):

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def evaluate(self, model_name):
        pass
