import abc

from PIL import Image
from torch.utils.data import Dataset


class AbstractBaseDataset(Dataset, abc.ABC):
    """Base class for a dataset."""

    def __init__(self, root_path, split='train', img_transform=None, text_transform=None):
        pass

    @classmethod
    @abc.abstractmethod
    def code(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def vocab_path(cls):
        raise NotImplementedError


class AbstractBaseTestDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def sample_img_for_visualizing(self, gt) -> Image:
        """return gt image"""
        raise NotImplementedError
