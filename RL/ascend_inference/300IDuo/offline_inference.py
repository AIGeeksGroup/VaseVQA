from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "VLM-R1-Qwen2.5VL-3B-OVD-0321"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=16384,
    dtype="float16", 
    enforce_eager=True
)

sampling_params = SamplingParams(
    max_tokens=512
)

image_path = "./resources/test.jpg"
describe = "杯子在哪个位置？请输出杯子的bbox坐标。"
event = "杯子"
image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"{image_path}",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": f"Please analyze the image and answer the following question. Your answer should include a brief description of the image content and the final answer. Wrap the description with `<description></description>` tags and the answer with `</think></think>` tags. The answer must be in JSON format, containing \"yes\" or \"no\", and provide bounding box coordinates of relevant objects as explanation. If no specific objects are involved, set \"explanations\" to \"None\". The output format is as follows:\n\n<description>Brief description of the image content goes here</description>\n\n<|FunctionCallBegin|>\n```json\n{{\"answer\": \"yes or no\",\n\"explanations\": [\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }},\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }}]\n}}\n```\n<escapeShell \n\nSpecific question: According to the rules or recognition requirements, {describe}. Does {event} appear in the image?"}
        ],
    },
]

messages = image_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)