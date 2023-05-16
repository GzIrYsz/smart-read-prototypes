import numpy as np


class DatasetLoader:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __get_x_original_shape(self, x_raw_set):
        x, y = x_raw_set.shape
        return x_raw_set.reshape(x, self.img_size, self.img_size)

    def __get_y_original_shape(self, y_raw_set):
        x = y_raw_set.shape
        return y_raw_set.reshape(x[0], 1)

    def get_y_and_x_set_from_dataframe(self, dataframe) -> tuple:
        y_raw_set = dataframe["label"].to_numpy(dtype=str)
        x_raw_set = dataframe.iloc[:, 2:].to_numpy(dtype=np.uint8)

        y_set = self.__get_y_original_shape(y_raw_set)
        x_set = self.__get_x_original_shape(x_raw_set)

        return y_set, x_set
