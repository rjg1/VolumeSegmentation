import numpy as np

class PlanePoint:
    def __init__(self, id, position, traits = None):
        self.position = np.array(position)
        self.id = id
        self.traits = traits.copy() if traits else {}
        # {"trait" : {"threshold": <int>, "metric": <str>, "value": int}}
        # <threshold> max total difference in <metric> for all compared <value>s
        self._verify_trait_structure() # validate structure

    def __str__(self):
        return f"PlanePoint <{self.id}> at: {self.position} with traits: {self.traits}"
    
    def __repr__(self):
        return str(self)
    
    def _verify_trait_structure(self):
        for trait_name, trait_data in self.traits.items():
            if not isinstance(trait_data, dict):
                raise ValueError(f"Trait '{trait_name}' should be a dictionary.")

            required_keys = {"threshold", "metric", "value"}
            actual_keys = set(trait_data.keys())
            if actual_keys != required_keys:
                raise ValueError(
                    f"Trait '{trait_name}' must contain exactly the keys {required_keys}, "
                    f"but got {actual_keys}"
                )

            if not isinstance(trait_data["threshold"], (int, float)):
                raise TypeError(f"Trait '{trait_name}': 'threshold' must be int or float.")
            if not isinstance(trait_data["metric"], str):
                raise TypeError(f"Trait '{trait_name}': 'metric' must be a string.")
            if not isinstance(trait_data["value"], (int, float)):
                raise TypeError(f"Trait '{trait_name}': 'value' must be int, float, or str.")
            

            acceptable_metrics = ["mse", "rmse", "mean", "max", "sum", "range", "std"]
            if trait_data["metric"] not in acceptable_metrics:
                raise ValueError(
                    f"Trait '{trait_name}' has metric {trait_data['metric']}, "
                    f"which is not in the acceptable metrics of {acceptable_metrics}"
                )
            
    # Adds a trait to a point
    def add_trait(self, trait, threshold, metric, value):
        self.traits[trait] = {}
        self.traits[trait]["threshold"] = threshold
        self.traits[trait]["metric"] = metric
        self.traits[trait]["value"] = value
        self._verify_trait_structure() # validate structure