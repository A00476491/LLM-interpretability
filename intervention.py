import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import SparseAutoencoder

class Intervene:
    def __init__(self, 
                 sae_model_dir='./model/20250403-041718/best_model.pth',
                 token_feature_tabel_dir='./data/token_feature_table.json'):
        # Load SAE model
        self.sae_model = SparseAutoencoder(input_dim=896, hidden_dim=896*20).cuda()
        self.sae_model.load_state_dict(torch.load(sae_model_dir))
        self.sae_model.eval()

        # Load LLM
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.lm.eval()

        # Load token-feature table
        with open(token_feature_tabel_dir, 'r', encoding='utf-8') as f:
            self.token_feature_table = json.load(f)
        self.words = [self.get_word(k) for k in self.token_feature_table.keys() if self.get_word(k) != ""]
        print(self.words)

    def hook_SAE(self, module, input, output):
        with torch.no_grad():
            output[0][0, -1, :], _ = self.sae_model(output[0][0, -1, :])
        return output

    def to_max_only_sparse_vector(self, vector):
        
        sparse_vec = torch.zeros_like(vector).cuda()
        max_index = torch.argmax(vector)
        sparse_vec[max_index] = vector[max_index]
        return sparse_vec

    def make_hook_activate(self, token_id, scale=1, strength=15):

        def hook_activate(module, input, output):
            with torch.no_grad():
                _, z = self.sae_model(output[0][0, -1, :])
                z /= scale
                token_vec = torch.Tensor(self.token_feature_table[token_id]).cuda()
                z += strength * self.to_max_only_sparse_vector(token_vec)
                output[0][0, -1, :] = self.sae_model.decode(z).reshape(-1)

            return output
        
        return hook_activate

    def make_hook_suppress(self, token_id, strength=30):

        def hook_suppress( module, input, output):
            with torch.no_grad():
                _, z = self.sae_model(output[0][0, -1, :])
                token_vec = torch.tensor(self.token_feature_table[token_id]).cuda()
                z -= strength * self.to_max_only_sparse_vector(token_vec)
                z = torch.clamp(z, min=0)
                output[0][0, -1, :] = self.sae_model.decode(z).reshape(-1)
            return output
        
        return hook_suppress

    def generate(self, question="Generally, which is smaller, a car or a train?", 
                 require="Give me a one-word factual answer"):

        prompt = f"{require}\n\n{question}"
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        dummy_input = self.tokenizer(text, return_tensors="pt")["input_ids"].cuda()
        eos_token_id = self.tokenizer.eos_token_id

        torch.cuda.empty_cache()

        for _ in range(80):
            with torch.no_grad():
                out = self.lm(dummy_input)
                next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
                dummy_input = torch.cat([dummy_input, next_token_id], dim=1)

            if next_token_id.item() == eos_token_id:
                break

        full_response = self.tokenizer.decode(dummy_input[0], skip_special_tokens=True)

        if question in full_response:
            answer = full_response.split(question)[-1].strip().replace('assistant', '').replace('\n', '')
        else:
            answer = full_response.strip().replace('assistant', '').replace('\n', '')
        
        if '.' in answer:
            answer = answer.rpartition('.')[0].strip() + '.'

        print(f"Q: {question}")
        print(f"A: {answer}")
        return answer

    def delete_exsiting_hook(self):
    
        try:
            self.handle.remove()
        except:
            pass
    
    def get_token(self, word):

        return str(self.tokenizer(word, return_tensors="pt")\
                    ["input_ids"].numpy().tolist()[0][0])
    
    def get_word(self, token):
            return self.tokenizer.decode([int(token)], skip_special_tokens=True)

    def case1(self):

        self.delete_exsiting_hook()

        print("=== Case 1 ===")
        print("=== Original Output ===")
        self.generate()

        print("\n=== Activate 'train' ===")
        token_id_train = self.get_token(" train")
        print(token_id_train)
        self.handle = self.lm.model.layers[14].\
            register_forward_hook(self.make_hook_activate(token_id=token_id_train, strength=15))
        self.generate()
        self.handle.remove()
    
    def case2(self):
        
        Question = 'Imagine a traffic scene.'
        Require = 'The answer is at most 50 words'
        # activated_words = self.words
        # activated_words = [' slowly', ' quickly', ' feeling', ' glow', ' day', ' friends']
        activated_words = [[None, -1], [' chatting', 10], [' helpful', 15], [' brightly', 15], [' protect', 15]]
        for word, strength in activated_words:
            print(f'\n\nActivate {word}:\n')
            if word is not None:
                token_id = self.get_token(word)
                self.handle = self.lm.model.layers[14].\
                    register_forward_hook(self.make_hook_activate(token_id=token_id, strength=strength))
            self.generate(question=Question, require=Require)
            if word is not None:
                self.handle.remove()

if __name__ == '__main__':

    exp =  Intervene(sae_model_dir='./model/20250403-041718/best_model.pth')
    exp.case1()
    exp.case2()
