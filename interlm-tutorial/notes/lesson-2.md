# 下载模型方法
* Hugging Face
```
pip install -U huggingface_hub
huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path
```
或者python代码
```
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json")
```
* ModelScope
```
pip install modelscope==1.9.5
pip install transformers==4.35.2
from modelscope import snapshot_download
snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='your path', revision='master')
```
* OpenXLab
```
pip install -U openxlab 
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-7b', model_name='InternLM-7b', output='your local path')
```

# huggingface 推理模型
```
from transformers import AutoTokenizer, AutoModelForCausalLM
#模型和tokenizer加载，制定路径就好。
model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

#构造输入
system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
messages = [(system_prompt, '')]
input_text = "你好"
# 流式输出接口，每次返回的response是所有已经生成的文本。
length = 0
for response, _ in model.stream_chat(tokenizer, input_text, messages):
  if response is not None:
    print(response[length:], flush=True, end="")
    length = len(response)
```
