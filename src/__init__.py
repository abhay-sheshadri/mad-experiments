import os
import sys

# Add the directory containing the cupbearer modules to the Python path
cupbearer_path = os.path.join(os.path.dirname(__file__), 'cupbearer/src/')
sys.path.insert(0, cupbearer_path)

from cupbearer import detectors, scripts, tasks
from .utils import *
from .mad_datasets import *
from .mad_detectors import *
from .mad_exps import *