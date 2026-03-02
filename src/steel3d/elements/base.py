from abc import ABC, abstractmethod


class ElementBase(ABC):
    @abstractmethod
    def edof(self, model):
        raise NotImplementedError

    @abstractmethod
    def ke_global(self, model):
        raise NotImplementedError

    @abstractmethod
    def feq_global(self, loadcase, model):
        raise NotImplementedError

    @abstractmethod
    def recover(self, u_global, loadcase, model):
        raise NotImplementedError
