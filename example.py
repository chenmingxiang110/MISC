01_data_prep_03_fast_talking_02_psola_seed

# b站视频解析：https://greenvideo.cc/bilibili
# b站音频解析：https://downloader.bhwa233.com/zh
# 素材
#     城市：https://www.bilibili.com/video/BV1RB4y1j7LW
#     开车（放音乐）：https://www.bilibili.com/video/BV1ykHLzxEF2
#     开车（不放音乐）：https://www.bilibili.com/video/BV1EuJdzWENL

# HOMEBREW_NO_AUTO_UPDATE=1 brew install openfst
# export CPLUS_INCLUDE_PATH=/opt/homebrew/include
# export LIBRARY_PATH=/opt/homebrew/lib
# pip install pynini -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install WeTextProcessing -i https://pypi.tuna.tsinghua.edu.cn/simple

# coding reference: https://github.com/zai-org/GLM-TTS/tree/main







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







def resample_wave(wave, sr, tgt_sr):
    assert len(wave)>0
    if sr==tgt_sr:
        return wave
    wave_ = utils_1d.resize1d(wave, int(max(len(wave) / sr * tgt_sr, 1)))
    return wave_

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

def get_biased_energy(wave, sr, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(max(sr * 0.005, 1)) * 2 + 1
    assert kernel_size%2==1
    energy = wave**2
    head_len = sr//5
    if len(wave)<head_len*2:
        bias = np.linspace(1,0,len(wave)//2)
    else:
        bias = np.linspace(1,0,head_len)
    bias = np.concatenate([bias, [0] * (len(wave)-len(bias)*2), bias[::-1]]) ** 2
    biased_energy = np.maximum(bias, energy)
    
    biased_energy = np.concatenate([[1] * (kernel_size//2), biased_energy, [1] * (kernel_size//2)])
    kernel = (1 - np.abs(np.linspace(-1,1,kernel_size))) ** 2
    kernel /= np.sum(kernel)
    biased_energy = np.convolve(biased_energy, kernel, mode='valid')
    return biased_energy

def shrink_wave(wave, sr, max_drop=0.001, ratio_keep=0.5):
    biased_energy = get_biased_energy(wave, sr)
    biased_energy_sorted = sorted(list(enumerate(biased_energy)), key=lambda x: -x[1])
    wave_shrink = []
    indices_keep = set([x[0] for x in biased_energy_sorted[:int(np.round(len(wave) * ratio_keep))]])
    for i, d in enumerate(wave):
        if i in indices_keep or biased_energy[i]>=max_drop * 0.75:
            wave_shrink.append(d)
    wave_shrink = np.array(wave_shrink)
    if len(wave_shrink) / len(wave) > ratio_keep:
        max_drop = max_drop
        wave = wave_shrink
        biased_energy = get_biased_energy(wave, sr)
        biased_energy_sorted = sorted(list(enumerate(biased_energy)), key=lambda x: -x[1])
        wave_shrink = []
        indices_keep = set([x[0] for x in biased_energy_sorted[:int(np.round(len(wave) * ratio_keep))]])
        for i, d in enumerate(wave):
            if i in indices_keep or biased_energy[i]>=max_drop:
                wave_shrink.append(d)
        wave_shrink = np.array(wave_shrink)
    return wave_shrink








seed_paths = search_files([
    "./data/glm_voice_seeds", "./data/glm_voice_seeds_aishell2",
], suffixes=["mp3"])[0]
audio_seeds = []
for seed_path in seed_paths:
    wave, sr = audio_io.load_audio(seed_path)
    wave = energy_norm(wave, target_energy=0.1)
    prompt_text = seed_path.split("/")[-1][2:-4]
    audio_seeds.append([wave, sr, prompt_text])
    break






index = 0
wave, sr, text = audio_seeds[index]
wave_int16 = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
# AudioSegment.from_mp3(path)
audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
print(text)
audio_segment







scale = 0.5
wave_processed = tsm.wsola(wave, scale)

wave_int16 = np.clip(wave_processed * 32767, -32767, 32767).astype(np.int16)
# AudioSegment.from_mp3(path)
audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
audio_segment





seed_paths = search_files(["./data/glm_voice_seeds", "./data/glm_voice_seeds_aishell2"], suffixes=["mp3"])[0]
audio_seeds = []
for seed_path in tqdm(seed_paths):
    wave, sr = audio_io.load_audio(seed_path)
    tgt_sr = 24000
    wave = resample_wave(wave, sr, tgt_sr=tgt_sr)
    wave = energy_norm(wave, target_energy=0.1)
    prompt_text = seed_path.split("/")[-1][2:-4]
    audio_seeds.append([wave, tgt_sr, prompt_text])
seeds_full_info = list(zip(seed_paths, audio_seeds))
print(len(audio_seeds))







out_folder = "./data/glm_voice_seeds_psola_80"
os.makedirs(out_folder, exist_ok=True)

for path, (raw_wave, sr, text) in tqdm(seeds_full_info):
    f_name = path.split("/")[-1]
    raw_wave = energy_norm(raw_wave, target_energy=0.1)
    wave_shrink = shrink_wave(raw_wave, sr, ratio_keep=0.1, max_drop=0.00015)
    scale = 0.8
    wave_shrink = tsm.wsola(wave_shrink, scale)
    
    out_path = os.path.join(out_folder, f_name)
    wave_int16 = np.clip(wave_shrink * 32767, -32767, 32767).astype(np.int16)
    audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    _ = audio_segment.export(out_path, format="mp3")





out_folder = "./data/glm_voice_seeds_psola_60"
os.makedirs(out_folder, exist_ok=True)

for path, (raw_wave, sr, text) in tqdm(seeds_full_info):
    f_name = path.split("/")[-1]
    raw_wave = energy_norm(raw_wave, target_energy=0.1)
    wave_shrink = shrink_wave(raw_wave, sr, ratio_keep=0.1, max_drop=0.00015)
    scale = 0.6
    wave_shrink = tsm.wsola(wave_shrink, scale)
    
    out_path = os.path.join(out_folder, f_name)
    wave_int16 = np.clip(wave_shrink * 32767, -32767, 32767).astype(np.int16)
    audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    _ = audio_segment.export(out_path, format="mp3")






wave_int16 = np.clip(raw_wave * 32767, -32767, 32767).astype(np.int16)
# AudioSegment.from_mp3(path)
audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
audio_segment







wave_int16 = np.clip(wave_shrink * 32767, -32767, 32767).astype(np.int16)
# AudioSegment.from_mp3(path)
audio_segment = AudioSegment(wave_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
audio_segment





text




