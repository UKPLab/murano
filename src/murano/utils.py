from typing import List, Union


class Location:
    pass


class LayerLocation(Location):
    def __init__(self, layers: Union[slice, int, List[int]]):
        self.layers = layers
