import pandas as pd
import types

class DataCollector:
    def __init__(self, model_reporters=None):
        """
        Initializes a DataCollector object.

        model_reporters: A dictionary of {name: function}, where function extracts a value from the model.
        """
        if model_reporters is None:
            model_reporters = {}

        # Store each reporter function and its collected data together
        self.model_data = {name: {"reporter": reporter, "values": []} for name, reporter in model_reporters.items()}

    def collect(self, model):
        """Collects data for each model variable by applying its reporter function to the model."""
        for name, data in self.model_data.items():
            reporter = data["reporter"]
            if callable(reporter):  # Ensure it's a function
                data["values"].append(reporter(model))
            else:
                raise Exception(f"Reporter for {name} is not callable.")

    def get_model_vars_dataframe(self):
        """Returns a Pandas DataFrame of the collected model variables."""
        if not self.model_data:
            return pd.DataFrame()  # Return an empty DataFrame if no reporters exist
        return pd.DataFrame({name: data["values"] for name, data in self.model_data.items()})