from abc import ABC, abstractmethod
from .meta import AnalyzerMeta


class BaseAnalyzer(ABC, metaclass=AnalyzerMeta):

    @abstractmethod
    def analyze(self, df):
        pass

    @abstractmethod
    def visualize(self, df, **kwargs):
        pass

    def __call__(self, df):
        self.analyze(df)
        self.visualize(df)
