# encoding: utf-8
from torch.utils.data import Dataset


class LRWDatasetInterface(Dataset):
    """
    An interface to implement LRW dataset object
    """

    def set_labels(self) -> list:
        """
        reads the labels file
        :return: a list of labels
        """

        raise NotImplementedError

    def append_files(self) -> list:
        """
        collects the relevant files to work on and return a list of them
        :return: list of files to work on
        """

        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """
        implement the [] operator
        :param idx: index for [] operator
        :return: the data from corresponding index
        """

        raise NotImplementedError

    def __len__(self) -> int:
        """
        implements the len operator
        :return: len of data to work on
        """

        raise NotImplementedError
