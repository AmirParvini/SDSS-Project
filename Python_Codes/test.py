from ast import Tuple
from joblib import parallel
import numpy as np
import random
from scipy.spatial.distance import cdist
import json
from typing import Dict, Any, Optional
import re
from copy import deepcopy

a = """
hi 
 my name is amir
""".strip()

print(a)