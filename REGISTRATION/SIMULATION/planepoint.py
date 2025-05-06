import numpy as np

class PlanePoint:
    def __init__(self, id, position, traits=None):
        self.id = id
        self.position = np.array(position)
        self.traits = traits.copy() if traits else {}  # { "trait_name": float }

        self._verify_trait_structure()

    def __str__(self):
        return f"PlanePoint <{self.id}> at {self.position} with traits: {self.traits}"

    def __repr__(self):
        return str(self)

    def _verify_trait_structure(self):
        for trait_name, value in self.traits.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Trait '{trait_name}' must be a numeric value, but got type '{type(value).__name__}'"
                )

    def add_trait(self, trait_name, value):
        """Adds or updates a trait."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Trait '{trait_name}' value must be numeric.")
        self.traits[trait_name] = value
