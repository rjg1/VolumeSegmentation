import numpy as np

class PlanePoint:
    def __init__(self, id, position, traits = {}):
        self.position = np.array(position)
        self.id = id
        self.traits = traits
        # {"trait" : {"threshold": <int>, "metric": <str>}}

    def __str__(self):
        return f"PlanePoint <{self.id}> at: {self.position}"
    
    def __repr__(self):
        return str(self)