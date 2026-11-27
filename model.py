from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
        )
    return model, tokenizer

def get_response(prompt,model, tokenizer, max_length=32768):
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True,enable_thinking=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=max_length,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
    # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content
model,tokenizer=load_model()


