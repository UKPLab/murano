from typing import List, Union


class Location:
    """
    Location specifies where to extract or intervene in model activations.
    
    Args:
        layers: Layer index/indices (int, list of ints, or slice)
        modules: Module name(s) within layers (e.g., "mlp", "attn", "output")
        token_pos: Token position(s) to extract (int, list of ints, or None for all tokens)
    """
    def __init__(
        self, 
        layers: Union[int, List[int], slice], 
        modules: Union[str, List[str]] = "mlp",
        token_pos: Union[int, List[int], None] = None
    ):
        # Normalize layers to list (or keep as slice)
        if isinstance(layers, slice):
            self.layers = layers
        else:
            self.layers = layers if isinstance(layers, list) else [layers]
        
        # Normalize modules to list
        self.modules = modules if isinstance(modules, list) else [modules]
        
        # Keep token_pos as is (can be int, list, or None)
        self.token_pos = token_pos

    def __repr__(self):
        return f"Location(layers={self.layers}, modules={self.modules}, token_pos={self.token_pos})"


# Backward compatibility alias
LayerLocation = Location
