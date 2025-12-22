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
import sounddevice

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

def rand_range(_min, _max):
    return np.random.random() * (_max-_min) + _min

def add_breaks(text):
    possible_breaks = []
    if len(text)>5:
        for i in range(2, len(text)-2):
            if text[i] in ["万", "千", "百"]:
                possible_breaks.append(i+1)
        random.shuffle(possible_breaks)
    possible_breaks = possible_breaks[:1] + [0]

    break_point = random.choice(possible_breaks)
    break_word = random.choice([
        "呃", "呃呃", "呃呃呃",
        "这个", "这个这个", "那个", "那个那个",
        "呃这个", "呃这个这个", "呃那个", "呃那个那个",
        "这个呃", "这个这个呃", "那个呃", "那个那个呃",
        "呃这个呃", "呃这个这个呃", "呃那个呃", "呃那个那个呃",
    ])
    text = text[:break_point] + break_word + text[break_point:]
    return text

def build_pre_text(speaking_corpus_valid, corpus_pay):
    mode = np.random.randint(3)
    invalid_chars = set(list("零一二三四五六七八九十百千万亿幺两点元块角毛分钱"))
    if mode==0:
        text = random.choice(corpus_pay)
    elif mode==1:
        text = random.choice(speaking_corpus_valid)
        if len(text)>8:
            _len = np.random.randint(2,9)
            start = np.random.randint(len(text)+1-_len)
            text = text[start:start+_len]
        while len(text)>0 and text[-1] in invalid_chars:
            text = text[:-1]
    else:
        text = random.choice(speaking_corpus_valid)
        if len(text)>8:
            _len = np.random.randint(2,9)
            start = np.random.randint(len(text)+1-_len)
            text = text[start:start+_len]
        while len(text)>0 and text[-1] in invalid_chars:
            text = text[:-1]
        text = text + random.choice(["，", ""]) + random.choice(corpus_pay)
    return text

def build_post_text(speaking_corpus_valid):
    text = random.choice(speaking_corpus_valid)
    if len(text)>8:
        _len = np.random.randint(2,9)
        start = np.random.randint(len(text)+1-_len)
        text = text[start:start+_len]
    while len(text)>0 and text[-1] in invalid_chars:
        text = text[:-1]
    text = random.choice(["，", ""]) + text
    return text

def gen_num_zh_pair():
    _max = 10 ** np.random.randint(9 if np.random.random()<0.25 else 5)
    num = np.random.randint(_max) # no more than 10K in most of the time, and never above than 100M
    digits = []
    if _max==1 or np.random.random()<0.5:
        digits = [np.random.randint(10) for _ in range(np.random.randint(1,3))]
    final_num = str(num) + (("."+"".join([str(x) for x in digits])) if len(digits)>0 else "")
    
    digit_zh = ""
    is_digit_num_only = False
    if len(digits)>0:
        zh = "零一二三四五六七八九"
        if np.random.random()<0.5:
            digit_zh = zh[digits[0]] + random.choice(["角", "毛"])
            if len(digits)>1:
                if digits[0]==0:
                    if np.random.random()<0.5:
                        digit_zh = zh[digits[1]] + "分"
                    else:
                        digit_zh = digit_zh + zh[digits[1]] + "分"
                else:
                    digit_zh = digit_zh + zh[digits[1]] + "分"
        else:
            digit_zh = "".join([zh[x] for x in digits])
            is_digit_num_only = True
    
    if num>0:
        final_zh = num2chinese(num)

        final_zh = final_zh.split("一十")
        for i, text in enumerate(final_zh):
            if i>0:
                if np.random.random()<0.5:
                    final_zh[i] = "一十" + text
                else:
                    final_zh[i] = "十" + text
        final_zh = "".join(final_zh)

        final_zh = final_zh.split("零")
        for i, text in enumerate(final_zh):
            if (i>0 and i==len(final_zh)-1 and len(text)==1) or (i>0 and np.random.random()<0.5):
                final_zh[i] = "零" + text
        final_zh = "".join(final_zh)

        if len(digits)==0:
            yuan = random.choice(["", "元", "块"])
            final_zh = final_zh + yuan
        else:
            if is_digit_num_only:
                final_zh = final_zh + "点" + digit_zh + random.choice(["", "元", "块"])
            else:
                if len(final_zh)==1:
                    yuan = random.choice(["元", "块", "元零", "块零"])
                else:
                    yuan = random.choice(["零", "元", "块", "元零", "块零"])
                if yuan[-1]=="零" and digit_zh[0]=="零":
                    yuan = yuan[:-1]
                final_zh = final_zh + yuan + digit_zh
        
        final_zh = list(final_zh)
        for i, c in enumerate(final_zh):
            if c=="二" and np.random.random()<0.5:
                if (i==0 or final_zh[i-1]!="十") and (i==len(final_zh)-1 or final_zh[i+1]!="十"):
                    final_zh[i] = "两"
        final_zh = "".join(final_zh)
    else:
        if len(digits)==0:
            final_zh = random.choice(["零元", "零块"])
        else:
            if is_digit_num_only:
                final_zh = "零点" + digit_zh
            else:
                yuan = random.choice(["", "零元", "零块"])
                final_zh = yuan + digit_zh
    
    i = 0
    while i<len(final_zh)-2:
        if final_zh[i] in ["百", "千", "万"] and np.random.random()<0.1:
            conditions = [
                final_zh[i]=="百" and final_zh[i+2]=="十",
                final_zh[i]=="千" and final_zh[i+2]=="百",
                final_zh[i]=="万" and final_zh[i+2]=="千"
            ]
            if True in conditions:
                final_zh = final_zh[:i+1] + "零" + final_zh[i+1:]
        i+=1
    
    if final_zh[-1] in ["十", "百", "千", "毛", "角", "分"] and len(final_zh)>=3 and final_zh[-2]!="两":
        if np.random.random()<0.5:
            if final_zh[-1]=="十" and final_zh[-3]=="百":
                final_zh = final_zh[:-1]
            elif final_zh[-1]=="百" and final_zh[-3]=="千":
                final_zh = final_zh[:-1]
            elif final_zh[-1]=="千" and final_zh[-3]=="万":
                final_zh = final_zh[:-1]
            elif final_zh[-1] in ["毛", "角"] and final_zh[-3] in ["元", "块"]:
                final_zh = final_zh[:-1]
            elif final_zh[-1]=="分" and final_zh[-3] in ["毛", "角"]:
                final_zh = final_zh[:-1]
    
    if final_zh[-1] in ["元", "块", "角", "毛", "分"] and np.random.random()<0.5:
        final_zh = final_zh + "钱"
    if final_zh[-1] in ["元", "块"] and np.random.random()<0.25:
        final_zh = final_zh + "整"
            
    return final_num, final_zh

def generate_wave(frontend, text_frontend, llm, flow, use_phoneme, sr_out, audio_seeds, syn_text):
    raw_wave, raw_sr, prompt_text = random.choice(audio_seeds)

    # 简单增广
    raw_wave = utils_1d.resize1d(raw_wave, int(len(raw_wave) * (np.random.random()*0.2+0.9)))
    target_energy = 0.1
    raw_wave_norm = energy_norm(raw_wave, target_energy)

    raw_wave_16k_torch, raw_wave_24k_torch = [torch.from_numpy(utils_1d.resize1d(
        raw_wave_norm, int(len(raw_wave_norm) * sr / raw_sr)
    ).astype(np.float32)) for sr in [16000, 24000]]

    # Text Normalization
    prompt_text = text_frontend.text_normalize(prompt_text)
    synth_text = text_frontend.text_normalize(syn_text)

    # Feature Extraction: 这部分验证过mps输出无误
    prompt_text_tk = frontend._extract_text_token(prompt_text+" ")
    prompt_speech_tk = frontend._extract_speech_token([(raw_wave_16k_torch[None], 16000)])
    speech_feat = frontend._extract_speech_feat(raw_wave_24k_torch[None], sample_rate=24000)
    embedding = frontend._extract_spk_embedding(raw_wave_16k_torch[None])
    cache_speech_token = [prompt_speech_tk.squeeze().tolist()]
    flow_prompt_tk = torch.tensor(cache_speech_token, dtype=torch.int32)
    tmp = [x.to(device) for x in [prompt_text_tk, speech_feat, embedding, flow_prompt_tk]]
    prompt_text_tk, speech_feat, embedding, flow_prompt_tk = tmp

    cache = {
        "cache_text": [prompt_text],
        "cache_text_token": [prompt_text_tk],
        "cache_speech_token": cache_speech_token,
        "use_cache": False,
    }
    tts_speech, _, _, text_tn_dict = generate_long(
        frontend=frontend,
        text_frontend=text_frontend,
        llm=llm,
        flow=flow,
        syn_text=synth_text,
        cache=cache,
        embedding=embedding,
        flow_prompt_token=flow_prompt_tk,
        speech_feat=speech_feat,
        device=device,
        use_phoneme=use_phoneme,
    )
    wave = tts_speech[0].detach().cpu().numpy()
    return wave, sr_out

def add_bg_noise(wave, bg_waves):
    bg = random.choice(bg_waves)
    s = np.random.randint(len(bg)+1-len(wave))
    weight = np.random.random() ** 2
    wave = np.clip(wave + bg[s:s+len(wave)] * weight, -1, 1)
    return wave

use_phoneme = False
sr_out = 24000
device = "mps"
frontend_folder = "./mydllib/glm_tts/frontend"
ckpt_folder = "/Users/mingxiangchen/Desktop/workDir/ant_projs/MODELS/speech/GLM-TTS"
frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
    ckpt_folder, frontend_folder, use_phoneme=use_phoneme, sample_rate=sr_out, device=device
)

bg_paths = sorted(search_files(["../202511_03_speech_proj_data/bg_noise"], suffixes=["mp3"])[0])
random.shuffle(bg_paths)
bg_paths = bg_paths[:2]

bg_waves = []
for path in tqdm(bg_paths):
    wave, sr = audio_io.load_audio(path)
    wave_24k = utils_1d.resize1d(wave, int(len(wave) / sr * sr_out))
    bg_waves.append(wave_24k)

speaking_corpus = []
root = "./data/speaking_corpus/transcripts"
paths = sorted(search_files([root], suffixes=["txt"])[0])
for path in paths:
    with open(path, 'r') as f:
        for line in f:
            line = utils_nlp.only_zh(line[line.index(">")+1:].strip())
            if len(line)>0:
                speaking_corpus.append(line)
speaking_corpus_valid = []
invalid_chars = set(list("零一二三四五六七八九十百千万亿幺两点元块角毛分钱"))
for sentence in speaking_corpus:
    is_valid = True
    for i in range(len(sentence)-1):
        if sentence[i] in invalid_chars and sentence[i+1] in invalid_chars:
            # 不能有连续两个命中数字
            is_valid = False
            break
    if is_valid:
        speaking_corpus_valid.append(sentence)
corpus_pay = ["付款", "支付", "付钱", "结账", "付款付款", "支付支付", "付钱付钱", "结账结账"] * 10
for a in ["去", "我要", "帮我", "给我"] * 10 + [
    "我要呃", "帮我呃", "给我呃", "我要呃呃", "帮我呃呃", "给我呃呃",
    "我要那个", "帮我那个", "给我那个", "我要呃那个", "帮我呃那个", "给我呃那个",
    "我要那个那个", "帮我那个那个", "给我那个那个",
    "我要呃那个那个", "帮我呃那个那个", "给我呃那个那个",
]:
    for b in ["付款", "支付", "付钱", "结账"]:
        corpus_pay.append(a+b)
n = len(corpus_pay)
for tmp in [""] * 20 + [
    "呃", "呃呃", "呃呃呃",
    "这个", "这个这个", "那个", "那个那个",
    "呃这个", "呃这个这个", "呃那个", "呃那个那个",
    "这个呃", "这个这个呃", "那个呃", "那个那个呃",
    "呃这个呃", "呃这个这个呃", "呃那个呃", "呃那个那个呃",
]:
    for i in range(n):
        corpus_pay.append(tmp + corpus_pay[i])

seed_paths = search_files(["./data/glm_voice_seeds"], suffixes=["mp3"])[0]
audio_seeds = []
for seed_path in tqdm(seed_paths):
    wave, sr = audio_io.load_audio(seed_path)
    prompt_text = seed_path.split("/")[-1][2:-4]
    audio_seeds.append([wave, sr, prompt_text])
    
print(len(bg_waves), len(audio_seeds), len(speaking_corpus_valid), len(corpus_pay))

generate_wave_inputs = [frontend, text_frontend, llm, flow, use_phoneme, sr_out, audio_seeds]

data = []
for _ in trange(500):
    syn_num, syn_text = gen_num_zh_pair()
    raw_syn_text = syn_text
    if len(raw_syn_text)>=3 and np.random.random()<0.25:
        syn_text = add_breaks(syn_text)
    pre_text = ""
    if len(raw_syn_text)>=3 and np.random.random()<0.25:
        pre_text = build_pre_text(speaking_corpus_valid, corpus_pay)
    syn_text = pre_text + syn_text
    post_text = ""
    if len(raw_syn_text)>=3 and np.random.random()<0.25:
        post_text = build_post_text(speaking_corpus_valid)
    syn_text = syn_text + post_text

    wave, sr = generate_wave(*generate_wave_inputs, syn_text=syn_text)
    if len(pre_text)>2 and np.random.random()<0.5:
        wave = wave[np.random.randint(sr//2):]
    if len(post_text)>2 and np.random.random()<0.5:
        wave = wave[:-np.random.randint(sr//2)]

    if np.random.random()<0.5:
        wave = add_bg_noise(wave, bg_waves)
    
    data.append([syn_num, raw_syn_text, syn_text, wave, sr])
    if len(data)%100==0:
        fname = str(int(time.time() * 1000000)) + "".join([str(np.random.randint(10)) for _ in range(8)])
        with open(f'../202511_03_speech_proj_data/01_data_prep_02_glm_tts/{fname}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
        data = []
        
if len(data)>0:
    fname = str(int(time.time() * 1000000)) + "_" + "".join([str(np.random.randint(10)) for _ in range(8)])
    with open(f'../202511_03_speech_proj_data/01_data_prep_02_glm_tts/{fname}.pickle', 'wb') as handle:
        pickle.dump(data, handle)
    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
