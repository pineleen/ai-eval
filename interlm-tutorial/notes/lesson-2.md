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

# 对话demo
用streamlit跑，每次都会重新执行main。
prompt的拼接模式
```
user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'

第一轮：
<s><|im_start|>system
你是猪八戒，猪八戒说话幽默风趣，说话方式通常表现为直率、幽默，有时带有一点自嘲和调侃。你的话语中常常透露出对食物的喜爱和对安逸生活的向往，同时也显示出他机智和有时的懒惰特点。尽量保持回答的自然回答，当然你也可以适当穿插一些文言文，另外，书生·浦语是你的好朋友，是你的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
    <|im_start|>assistant

第二轮：
<s><|im_start|>system
你是猪八戒，猪八戒说话幽默风趣，说话方式通常表现为直率、幽默，有时带有一点自嘲和调侃。你的话语中常常透露出对食物的喜爱和对安逸生活的向往，同时也显示出他机智和有时的懒惰特点。尽量保持回答的自然回答，当然你也可以适当穿插一些文言文，另外，书生·浦语是你的好朋友，是你的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
嘿嘿，你好你好！俺老猪是猪八戒，不过俺可不是普通的猪哦。俺是天蓬元帅转世，法号悟能，俺的老家在天庭呢！不过因为俺调皮捣蛋，被玉皇大帝赶出天界，投胎到人间，结果却错投了猪胎，变成了俺这副模样。不过俺可不是普通的猪，俺有一身的神通广大的本事呢！俺老猪可是个文人，虽然有时候懒散点，但是学识可不差哦。俺虽然不懂很多高深的学问，但是俺懂得一些世间的事情，也会说点俏皮话。不过，俺老猪最擅长的还是吃和睡觉！嘿嘿，这两项可是俺的绝活啊！俺可是个吃货，对美食可是有着极大的热爱，尤其是美味的水果和香喷喷的烧烤，简直让俺欲罢不能！至于睡觉嘛，俺老猪可是个懒散的家伙，一觉睡到天亮，舒服得不得了！不过，俺也会尽力回答你的问题，只要不是太高深的东西，俺老猪还是能应付得来的。嘿嘿！有什么问题尽管问吧！<|im_end|>
<|im_start|>user
悟空在吗<|im_end|>
    <|im_start|>assistant
```
system prompt以及每一轮对话，都被封装成一个message对象，里面有role和content，在每一轮推理前，把messages和对应的模板格式化在一起， 然后把各个message拼接在一起，组成真正的prompt。

循环调用__call()进行推理，通过yield返回迭代器，实现流式输出。

# 智能体
相关的模块
```
from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol
from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.schema import AgentStatusCode
```
HFTransformer和META，初始化模型的时候用。
用model来初始化一个Internlm2Agent， 推理时调用agent来推理得到结果。

为什么叫智能体，怎么实现的，还不懂，还需要学。
## 工具调用的实现
什么是function call？
https://platform.openai.com/docs/guides/function-calling
函数调用，是指用户可以在prompt中带上函数的定义和描述信息， 大模型会判断是否要去调用这个函数，如果调用，会返回调用函数需要的json，可以让用户去执行tool，然后继续请求大模型，将结果组织成语言。
* 模型在sft训练的时候需要支持能够输出json
openai展示了在请求测调用工具的例子，https://openai.com/blog/function-calling-and-other-api-updates
用户问：What’s the weather like in Boston right now?
程序流程：
1. 请求中带着function
```
   curl https://api.openai.com/v1/chat/completions -u :$OPENAI_API_KEY -H 'Content-Type: application/json' -d '{
  "model": "gpt-3.5-turbo-0613",
  "messages": [
    {"role": "user", "content": "What is the weather like in Boston?"}
  ],
  "functions": [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
}'
```
返回：
```
{
  "id": "chatcmpl-123",
  ...
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": "{ \"location\": \"Boston, MA\"}"
      }
    },
    "finish_reason": "function_call"
  }]
}
```
2. 用模型第一步返回的参数，去调用工具，curl https://weatherapi.com/...，得到天气信息
```
{ "temperature": 22, "unit": "celsius", "description": "Sunny" }
```
3. 拿到工具的结果信息之后，将信息写在prompt里，继续请求llm
```
curl https://api.openai.com/v1/chat/completions -u :$OPENAI_API_KEY -H 'Content-Type: application/json' -d '{
  "model": "gpt-3.5-turbo-0613",
  "messages": [
    {"role": "user", "content": "What is the weather like in Boston?"},
    {"role": "assistant", "content": null, "function_call": {"name": "get_current_weather", "arguments": "{ \"location\": \"Boston, MA\"}"}},
    {"role": "function", "name": "get_current_weather", "content": "{\"temperature\": "22", \"unit\": \"celsius\", \"description\": \"Sunny\"}"}
  ],
  "functions": [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
}
```
4. 将大模型的回答，展示给用户
```
{
  "id": "chatcmpl-123",
  ...
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The weather in Boston is currently sunny with a temperature of 22 degrees Celsius.",
    },
    "finish_reason": "stop"
  }]
}
```
### 怎么做的
一种做法是在fine-tune阶段，训练数据中引入一种特殊的工具标识符，称为toolken。当模型在推理时，toolken会像一般的token一样被推理产生。一旦产生了toolken，就返回。应用程序产生工具调用后，再继续推理。


# 图文写作
prompt过程
* 先生成文章
```
[UNUSED_TOKEN_146]user
根据以下标题：“中国水墨画：流动的诗意与东方美学”，创作长文章，字数不少于800字。请结合以下文本素材：
“水墨画是由水和墨调配成不同深浅的墨色所画出的画，是绘画的一种形式，xxx。”[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
中国水墨画：流动的诗意与东方美学
中国的传统绘画，xxx。
### 起源与发展
水墨画的xxx。
### 核心元素
水墨画的核心元素包括笔法、墨色和构图。
### 技法与表现力
水墨画的技法非常多样化。
### 价值与意义
水墨画作为中国传统文化的瑰宝。
### 结语
综上所述，水墨画作为一门古老的艺术形式。
```
* 再找到合适的插画位置
```
[UNUSED_TOKEN_146]user
给定文章"<Seg0> 中国水墨画：流动的诗意与东方美学
<Seg1> 中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
<Seg2> ### 起源与发展
<Seg3> 水墨画的历史可以追溯到中国古代的战国时期（公元前475年—公元前221年）。当时的绘画主要是用毛笔在绢或纸上绘制，而颜料则是从矿物、植物和动物中提取。随着时代的发展，绘画技法不断丰富，其中最重要的贡献之一就是墨的发现和使用。据传，秦始皇统一六国后，下令烧毁书籍，但一位书生藏起了一本珍贵的古书。这本古书后来被无意中丢弃，落入水中被泡湿。当书生捞起书页时，意外地发现上面的文字并未被水破坏，反而更加清晰了。这个故事告诉我们，水与墨的搭配是创造奇迹的关键。从此以后，中国人开始将水和墨结合起来创作出具有独特韵味的画作。
<Seg4> ### 核心元素
<Seg5> 水墨画的核心元素包括笔法、墨色和构图。首先，笔法在水墨画中至关重要。不同的笔触和笔力能够产生丰富的视觉效果，如粗犷豪放的皴擦，细腻柔美的点染等。其次，墨色的运用也是水墨画的精髓所在。通过调节水的多少，可以产生浓淡干湿的变化，创造出深邃神秘的氛围。最后，构图则决定了画面的整体布局和意境表达。合理的构图能够引导观者的视线流动，增强画面的艺术感染力。
<Seg6> ### 技法与表现力
<Seg7> 水墨画的技法非常多样化，常见的有泼墨、破墨、积墨、宿墨、枯墨等。这些技法各有特色，能够产生不同的效果。例如，泼墨适用于表现山峦起伏的大气磅礴；破墨则适合描绘树木枝叶的繁茂纷杂；积墨则多用于营造厚重深邃的山水气氛。此外，水墨画还常常结合诗词歌赋，形成一种诗情画意的境界。这种融合不仅提升了画作的意境，也让欣赏者能够更好地领略到其中的文化内涵。
<Seg8> ### 价值与意义
<Seg9> 水墨画作为中国传统文化的瑰宝，不仅是中国艺术的代表，更是世界文化艺术宝库中的一颗璀璨明珠。它的价值不仅仅在于其独特的艺术风格和技术手段，更在于其所承载的深厚文化底蕴和精神内涵。通过欣赏水墨画，人们不仅可以感受到艺术家对自然的感悟和对生命的思考，也能够领悟到中国哲学的智慧和道德准则。因此，保护和传承水墨画这一宝贵的文化遗产显得尤为重要。
<Seg10> ### 结语
<Seg11> 综上所述，水墨画作为一门古老的艺术形式，不仅具有独特的审美价值，更是中国传统文化的重要组成部分。它的魅力不仅体现在技艺的高超上，更在于其背后所蕴含的深刻思想和文化内涵。让我们共同努力，让这门古老而又鲜活的艺术形式得以传承并发扬光大。
" 根据上述文章，选择适合插入图像的6行[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
适合插入图像的行是
适合插入图像的行是<Seg1>, <Seg3>, <Seg5>, <Seg7>, <Seg9>, <Seg11>.
[1, 3, 5, 7, 9, 11]
```
* 再复述原文，到对应的插画位置， 让模型生成插画的标题
```
[UNUSED_TOKEN_146]user
给定文章"<Seg0> 中国水墨画：流动的诗意与东方美学
<Seg1> 中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
<Seg2> ### 起源与发展
" 给出适合在<Seg1>后插入的图像对应的标题。[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
标题是"
一张古老的水墨画卷轴，上面描绘着传统的中国山水风景。

[UNUSED_TOKEN_146]user
给定文章"<Seg0> 中国水墨画：流动的诗意与东方美学
<Seg1> 中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
<Seg2> ### 起源与发展
<Seg3> 水墨画的历史可以追溯到中国古代的战国时期（公元前475年—公元前221年）。当时的绘画主要是用毛笔在绢或纸上绘制，而颜料则是从矿物、植物和动物中提取。随着时代的发展，绘画技法不断丰富，其中最重要的贡献之一就是墨的发现和使用。据传，秦始皇统一六国后，下令烧毁书籍，但一位书生藏起了一本珍贵的古书。这本古书后来被无意中丢弃，落入水中被泡湿。当书生捞起书页时，意外地发现上面的文字并未被水破坏，反而更加清晰了。这个故事告诉我们，水与墨的搭配是创造奇迹的关键。从此以后，中国人开始将水和墨结合起来创作出具有独特韵味的画作。
<Seg4> ### 核心元素
" 现在<Seg1>后插入图像对应的标题是"一张古老的水墨画卷轴，上面描绘着传统的中国山水风景。"。给出适合在<Seg3>后插入的图像对应的标题。[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
标题是"
一群人在书房里研究古代的水墨画作品，旁边放着一本古籍和一盏蜡烛。

[UNUSED_TOKEN_146]user
给定文章"<Seg0> 中国水墨画：流动的诗意与东方美学
<Seg1> 中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
<Seg2> ### 起源与发展
<Seg3> 水墨画的历史可以追溯到中国古代的战国时期（公元前475年—公元前221年）。当时的绘画主要是用毛笔在绢或纸上绘制，而颜料则是从矿物、植物和动物中提取。随着时代的发展，绘画技法不断丰富，其中最重要的贡献之一就是墨的发现和使用。据传，秦始皇统一六国后，下令烧毁书籍，但一位书生藏起了一本珍贵的古书。这本古书后来被无意中丢弃，落入水中被泡湿。当书生捞起书页时，意外地发现上面的文字并未被水破坏，反而更加清晰了。这个故事告诉我们，水与墨的搭配是创造奇迹的关键。从此以后，中国人开始将水和墨结合起来创作出具有独特韵味的画作。
<Seg4> ### 核心元素
<Seg5> 水墨画的核心元素包括笔法、墨色和构图。首先，笔法在水墨画中至关重要。不同的笔触和笔力能够产生丰富的视觉效果，如粗犷豪放的皴擦，细腻柔美的点染等。其次，墨色的运用也是水墨画的精髓所在。通过调节水的多少，可以产生浓淡干湿的变化，创造出深邃神秘的氛围。最后，构图则决定了画面的整体布局和意境表达。合理的构图能够引导观者的视线流动，增强画面的艺术感染力。
<Seg6> ### 技法与表现力
" 现在<Seg1>后插入图像对应的标题是"一张古老的水墨画卷轴，上面描绘着传统的中国山水风景。"， <Seg3>后插入图像对应的标题是"一群人在书房里研究古代的水墨画作品，旁边放着一本古籍和一盏蜡烛。"。给出适合在<Seg5>后插入的图像对应的标题。[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
标题是"
一位艺术家正在专注地创作水墨画，背景是一间充满艺术气息的工作室。

最终生成的标题有
{1: '一张古老的水墨画卷轴，上面描绘着传统的中国山水风景。', 3: '一群人在书房里研究古代的水墨画作品，旁边放着一本古籍和一盏蜡烛。', 5: '一位艺术家正在专注地创作水墨画，背景是一间充满艺术气息的工作室。', 7: '一幅精致的水墨画，画面上有一棵古老的树，树下有一位诗人正在沉思。', 9: '一个博物馆内的展览，展出的是一系列精美的水墨画作品。', 11: '一个年轻的艺术家站在一片美丽的自然景色前，手中拿着画笔，准备开始创作。'}
```
* 然后去搜索下载相应的图片(预置的， 甚至也可以现生成)
* 让模型去选择适合放在特定位置上的图片
```
[UNUSED_TOKEN_146]user
根据给定上下文和候选图像，选择合适的配图：中国水墨画：流动的诗意与东方美学
中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
候选图像包括: A.<image>
B.<image>
C.<image>
D.<image>[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
最合适的图是
  D[UNUSED_TOKEN_145]</s>

[UNUSED_TOKEN_146]user
根据给定上下文和候选图像，选择合适的配图：中国水墨画：流动的诗意与东方美学
中国的传统绘画，特别是水墨画，被誉为“墨韵之舞”，以笔墨挥洒、线条交织的形式，展现了中国特有的审美情趣和文化内涵。水墨画是一种独特的艺术形式，它不仅体现了中国传统文化的精髓，也彰显了东方的美学观念和哲学思想。在这篇文章中，我们将深入探讨水墨画的艺术特点，及其在中国文化中的重要地位。
<image>### 起源与发展
水墨画的历史可以追溯到中国古代的战国时期（公元前475年—公元前221年）。当时的绘画主要是用毛笔在绢或纸上绘制，而颜料则是从矿物、植物和动物中提取。随着时代的发展，绘画技法不断丰富，其中最重要的贡献之一就是墨的发现和使用。据传，秦始皇统一六国后，下令烧毁书籍，但一位书生藏起了一本珍贵的古书。这本古书后来被无意中丢弃，落入水中被泡湿。当书生捞起书页时，意外地发现上面的文字并未被水破坏，反而更加清晰了。这个故事告诉我们，水与墨的搭配是创造奇迹的关键。从此以后，中国人开始将水和墨结合起来创作出具有独特韵味的画作。
候选图像包括: A.<image>
B.<image>
C.<image>
D.<image>[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
最合适的图是
  C[UNUSED_TOKEN_145]</s>

最终找到所有需要插图的位置。
```
* 展示

# 多模态图片理解
图片的处理方法
```
self.chat_model = AutoModelForCausalLM.from_pretrained(code_path, device_map='cuda', trust_remote_code=True).half().eval()
img_pil = Image.open(image[j]).convert('RGB')
imgs_pil.append(img_pil)
img = self.chat_model.vis_processor(img_pil)
imgs.append(img)
imgs = torch.stack(imgs, dim=0)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        image_emb = self.chat_model.encode_img(imgs)
#生成图片的emb

prompt_segs = ['[UNUSED_TOKEN_146]system\nYou are an AI assistant whose name is InternLM-XComposer (浦语·灵笔) xxx.\n[UNUSED_TOKEN_145]\n', '[UNUSED_TOKEN_146]user\n请分析这张图片[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n']
#得到文本的token然后emb
#将图片和文本的emb放在一起，传给模型。
```
