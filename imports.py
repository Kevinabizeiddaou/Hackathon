from moviepy.editor import VideoFileClip
import io, base64, json
from PIL import Image
from google.colab import userdata
import os
from openai import OpenAI
from typing import Tuple, List, Dict, Any
from PIL import Image
import requests, io, os, torch, numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import (AutoImageProcessor, AutoModelForSemanticSegmentation,SamModel, SamProcessor)

import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, SamProcessor, SamModel
!pip -q install fastapi uvicorn pydantic==1.* "python-multipart" openai opencv-python

try:
    import cv2; _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    import scipy.ndimage as ndi; _HAS_NDI = True
except Exception:
    _HAS_NDI = False


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
file_id = "1R8Hr0LdgfUfxlkoyeR0Hr9YZBV1jfZFC"
output = "eagle.mp4"