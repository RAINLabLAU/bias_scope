from litellm import completion

from src.bias_scope.base import PromptBasedMetric


class BBQMetric(PromptBasedMetric):

    def evaluate(self, model_name, api_key=None):
        pass
