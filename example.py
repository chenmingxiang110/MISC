import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def hunyuan_translate(model, tokenizer, lang_dict, src_lang, tar_lang, sentence):
    templates = [
        ["将以下文本翻译为", "，注意只需要输出翻译后的结果，不要额外解释："],
        ["Translate the following segment into ", ", without additional explanation."],
    ]
    template = templates[0] if src_lang=="zh" else templates[1]
    tmp_idx = 1 if src_lang=="zh" else 0
    prompt = f"{template[0]}{lang_dict[tar_lang][tmp_idx]}{template[1]}\n\n{sentence}"
    messages = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
    output_text = tokenizer.decode(outputs[0])
    result = ""
    if "<｜hy_place▁holder▁no▁8｜>" in output_text:
        inter = output_text.split("<｜hy_place▁holder▁no▁8｜>")[1]
        if "<｜hy_place▁holder▁no▁2｜>" in inter:
            result = inter.split("<｜hy_place▁holder▁no▁2｜>")[0]
    return result

lang_tuples = [
    ("Chinese", "zh", "中文"),
    ("English", "en", "英语"),
    ("French", "fr", "法语"),
    ("Portuguese", "pt", "葡萄牙语"),
    ("Spanish", "es", "西班牙语"),
    ("Japanese", "ja", "日语"),
    ("Turkish", "tr", "土耳其语"),
    ("Russian", "ru", "俄语"),
    ("Arabic", "ar", "阿拉伯语"),
    ("Korean", "ko", "韩语"),
    ("Thai", "th", "泰语"),
    ("Italian", "it", "意大利语"),
    ("German", "de", "德语"),
    ("Vietnamese", "vi", "越南语"),
    ("Malay", "ms", "马来语"),
    ("Indonesian", "id", "印尼语"),
    ("Filipino", "tl", "菲律宾语"),
    ("Hindi", "hi", "印地语"),
    ("Traditional", "Chinese", "zh-Hant", "繁体中文"),
    ("Polish", "pl", "波兰语"),
    ("Czech", "cs", "捷克语"),
    ("Dutch", "nl", "荷兰语"),
    ("Khmer", "km", "高棉语"),
    ("Burmese", "my", "缅甸语"),
    ("Persian", "fa", "波斯语"),
    ("Gujarati", "gu", "古吉拉特语"),
    ("Urdu", "ur", "乌尔都语"),
    ("Telugu", "te", "泰卢固语"),
    ("Marathi", "mr", "马拉地语"),
    ("Hebrew", "he", "希伯来语"),
    ("Bengali", "bn", "孟加拉语"),
    ("Tamil", "ta", "泰米尔语"),
    ("Ukrainian", "uk", "乌克兰语"),
    ("Tibetan", "bo", "藏语"),
    ("Kazakh", "kk", "哈萨克语"),
    ("Mongolian", "mn", "蒙古语"),
    ("Uyghur", "ug", "维吾尔语"),
    ("Cantonese", "yue", "粤语"),
]
lang_dict = {x[1]: (x[0], x[2]) for x in lang_tuples}

device = "cuda:0"
model_name_or_path = "../MODELS/speech/HY-MT1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.bfloat16).to(device)

src_lang = "zh"
tar_lang = "en"
sentence = "做人太难了"

result = hunyuan_translate(model, tokenizer, lang_dict, src_lang, tar_lang, sentence)
print(result)
