import abc


class BaseInferenceBackend:
    @abc.abstractmethod
    def generate(self, prompt, params):
        pass
