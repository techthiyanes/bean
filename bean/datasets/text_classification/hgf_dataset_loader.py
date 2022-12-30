from collections import defaultdict
from typing import Dict, Tuple
import os
import logging
from datasets import load_dataset, Value, Features

logger = logging.getLogger(__name__)


class HgfTextClassificationLoader:

    def __init__(self):
        pass