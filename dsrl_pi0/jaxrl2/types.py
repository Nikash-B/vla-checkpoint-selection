"""Type aliases shared across jaxrl2: nested array data, PRNG keys and parameter trees."""

from typing import Any, Dict, Union

import flax
import numpy as np

DataType = Union[np.ndarray, Dict[str, "DataType"]]
# Changed from upstream 7f48937: DatasetDict moves here from jaxrl2/data/dataset.py, which is
# removed with its Dataset class (its seed() set a generator that nothing read).
DatasetDict = Dict[str, DataType]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
