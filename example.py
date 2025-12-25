01_data_prep_03_fast_talking_02_psola_seed

import os
import sys
import logging

# memory efficient attention is not available for cpu and mps
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "darwin" in sys.platform:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True" # 少一点显存预留占用 会导致变慢
logging.disable(logging.CRITICAL)

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import bz2
import gzip
import pickle
import base64

import re
import cv2
import json
import copy
import time
import scipy
import random
import librosa
import warnings
import itertools
import functools
import subprocess

import soundfile
try:
    import sounddevice
except Exception as e:
    print("Failed to import sounddevice. Unable to play audio.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from wurlitzer import pipes as warning_sup_pipes

from pydub import AudioSegment
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from qwen_vl_utils import process_vision_info

import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import LlamaForCausalLM
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor

import my1dlib.tsm as tsm
import my1dlib.io as audio_io
import my1dlib.utils as utils_1d
import mydllib.nlp.utils_nlp as utils_nlp

from my1dlib.feature.mel import melspectrogramdb
from my2dlib.utils import get_lr, set_lr, count_param, format_od, reformat_od
from my2dlib.utils import seed_everything, num_add_comma, num2sci
from my2dlib.data_prep import search_imgs, search_files, read_jsonl_file
from my2dlib.data_prep import resize_short, resize_long, resize_width, resize_height
from my2dlib.data_prep import center_pad, center_crop, rand_crop, patchify, resize_as_patch_multiples
from mydllib.glm_tts import load_models, generate_long
from mydllib.nlp.conversion import num2chinese
from mydllib.transformer.model_starling import StarlingConfig, StarlingForCausalLM
from mydllib.transformer.model_starling_vision import StarlingVisionConfig, StarlingVisionForCausalLM
from mydllib.transformer.tokenizer import Tokenizer, PuncNormalizer

from matplotlib.font_manager import FontProperties
plt_font = FontProperties(fname="../../DATASETS/fonts/Yuanti.ttc")
# plt.gca().set_title(name, fontproperties=plt_font)




def energy_pre_norm(wave, target_energy):
    energy = np.mean(wave ** 2)
    scale = np.sqrt(target_energy / energy)
    return np.clip(wave * scale, -1, 1)

def energy_norm(wave, target_energy):
    wave = energy_pre_norm(wave, 0.005)
    energy = wave ** 2
    energy_mask = energy > 0.05
    avg_energy = np.sum(energy * energy_mask) / np.sum(energy_mask)
    scale = np.sqrt(target_energy / avg_energy)
    return np.clip(wave * scale, -1, 1)


seed_paths = search_files([
    "./data/glm_voice_seeds", "./data/glm_voice_seeds_aishell2",
], suffixes=["mp3"])[0]
audio_seeds = []
for seed_path in tqdm(seed_paths):
    wave, sr = audio_io.load_audio(seed_path)
    wave = energy_norm(wave, target_energy=0.1)
    prompt_text = seed_path.split("/")[-1][2:-4]
    audio_seeds.append([wave, sr, prompt_text])
