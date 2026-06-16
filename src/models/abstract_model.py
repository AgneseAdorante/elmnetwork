from abc import ABC, abstractmethod
from typing import Any


class DynamicModel(ABC):

    @abstractmethod
    def get_init_state(self) -> Any:
        pass

    @abstractmethod
    def get_init_carry(self, key: Any) -> Any:
        pass

    @abstractmethod
    def dynamics(
        self,
        carry: Any,
        x_t: Any,
        monitor: Any,
        inference: bool,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def __call__(
        self,
        x: Any,
        monitor: Any,
        inference: bool,
        init_carry: Any,
        *,
        key: Any,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def get_train_params_filter(self) -> Any:
        pass
