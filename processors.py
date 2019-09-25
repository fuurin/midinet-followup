from abc import ABCMeta, abstractmethod
from pypianoroll import Multitrack

class Processor(metaclass=ABCMeta):
    """
    事前処理の1単位はこのクラスを継承すること．
    """
    def __init__(self, **kwargs):
        """
        処理に用い値をコンストラクタで受け取る．
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def __call__(self, **args):
        """
        データを受け取り，処理を行ったデータを返す．
        実装強制．
        """
        raise NotImplementedError()

class SequentialProcessor(Processor, metaclass=ABCMeta):
    """
    複数のProcessorによる処理をまとめて行う
    """
    def __init__(self, processors):
        if not isinstance(processors, list): processors = [processors]
        for processor in processors:
            if not issubclass(type(processor), Processor):
                raise TypeError(f"The processor of {type(processor)} must be a subclasses of Processor.")
        self.processors = processors
    
    @abstractmethod
    def __call__(self, **args):
        """
        データを受け取り，処理を行ったデータを返す．
        実装強制．
        どのようにデータを扱うかはサブクラスへ委譲する
        """
        raise NotImplementedError()

class PypianorollProcessor(SequentialProcessor):
    """
    pypianoroll.Multitrackと渡されたdataに対して順次処理を行う
    """
    def __call__(self, ppr, data):
        if not isinstance(ppr, Multitrack): ppr = Multitrack(ppr)
        for processor in self.processors:
            ppr, data = processor(ppr, data)
        return ppr, data