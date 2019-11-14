from abc import ABC, abstractmethod

from common import get_input_fn_and_steps_per_epoch
from constants import TFRECORDS_SAVE_PATH


class AbstractModel(ABC):
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_input_fn_and_steps_per_epoch(self, set_name, batch_size=None):
        pass
