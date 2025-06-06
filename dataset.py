from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import copy
import json
from tqdm import tqdm

# hook the 15th decoder output of lm
activations = {}
def forward_hook(module, input, output):
    activations["decoder15"] = copy.deepcopy(output[0].cpu().detach() \
                                    if isinstance(output, (tuple, list)) else output.cpu().detach())
    activations["decoder15"] = activations["decoder15"].numpy().tolist()


# Generate a story that begins with a specific sentence
# Return generated tokens and corresponding activations
def generate_single_story(lm, query = "I saw a Truck on the way to school."):

    prompt = f"Give me a short story that begins with: {query} \n\n"
    prompt += "Try to include more transportation-related words in the story.\n\n"
    prompt += "The story should contain at most 50 words"

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dummy_input = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
    eos_token_id = tokenizer.eos_token_id

    token_activation = []
    for i in range(150):
        out = lm(dummy_input)
        
        next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)  # shape: (1, 1)
        
        dummy_input = torch.cat([dummy_input, next_token_id], dim=1)
        token_activation.append([str(next_token_id.cpu().numpy().tolist()[0][0]), activations["decoder15"][0][-1]])

        if next_token_id.item() == eos_token_id:
            break

    return token_activation


def generate_stories(lm, sentences, save_dir='./data/dataset.json'):

    '''
    structure of stories
    {
        transportation1: {
            First sentence1 : [[token1, activation1], [token2, activation2], ...]
            First sentence2 : [[token1, activation1], [token2, activation2], ...]
            ...
            First sentence10 : [[token1, activation1], [token2, activation2], ...]
            }
        transportation2: {
            First sentence1 : [[token1, activation1], [token2, activation2], ...]
            First sentence2 : [[token1, activation1], [token2, activation2], ...]
            ...
            First sentence10 : [[token1, activation1], [token2, activation2], ...]
            }
            
        ...

        transportation30: {
            First sentence1 : [[token1, activation1], [token2, activation2], ...]
            First sentence2 : [[token1, activation1], [token2, activation2], ...]
            ...
            First sentence10 : [[token1, activation1], [token2, activation2], ...]
            }
    }
    '''

    stories = {}
    torch.cuda.empty_cache() 
    with torch.no_grad():
        for obj in sentences:
            for k, sentences in obj.items():
                for sentence in tqdm(sentences, desc=f"Processing '{k}'", leave=False):
                    token_activation = generate_single_story(lm, sentence)
                    try:
                        stories[k][sentence] = token_activation
                    except:
                        stories[k] = {sentence: token_activation}

    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(stories, f, indent=4, ensure_ascii=False)
    
    print(f'\nAll stories have been generated and stored in {save_dir}\n')
    
    return stories


if __name__ == '__main__':

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    lm.eval().cuda()

    # Hook the output from decoder15 of qwen
    hook_handle = lm.model.layers[14].register_forward_hook(forward_hook)

    # Generate 300 stories for 30 types of transportation
    file_path = './data/transportation_sentences.json'
    with open(file_path, 'r') as file:
        sentences = json.load(file)
    stories = generate_stories(lm, sentences)

    # test case
    story = stories['Car']['The Car is parked outside the building.']
    print(f"Story: {tokenizer.decode([int(token[0]) for token in story], skip_special_tokens=True)}")




