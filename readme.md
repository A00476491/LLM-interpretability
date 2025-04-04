# What do we need to do?

This assignment requires us to train an **autoencoder** to interpret a **large language model** (LLM) on a specific or mulitple **topics** and conduct **intervention experiments**.

---
# Run Code

- `exp_build_dataset.ipynb`: Generates `dataset.json`, containing 19K tokens and their corresponding features.  
- `exp_train.ipynb`: Trains the SAE and produces `token_feature_count.json`, which records how often each feature is activated for each token.  
- `exp_intervene.ipynb`: Adjusts features to control the LLM's output.

> ⚠️ Generating `dataset.json` and `token_feature_count.json` can be slow.  
> You can download these files and the trained model from the following link:  
> https://drive.google.com/drive/folders/1kxUiiE06ZV_dKm-MTSR7QBiIf1BaSSVH?usp=sharing


---

# current progress

## 1. LLM: Qwen2.5-0.5B
- **Model Size**: 542M parameters  
  (For comparison: GPT-2 Medium: 345M, GPT-2 Large: 762M)
- **Advantages**:
  - Lightweight: can run on my laptop (~2 tokens/sec)
  - Trained on more data through **Knowledge Distillation**
  - Uses **modern training methods**: Instruction Learning and Reinforcement Learning from Human Feedback (RLHF)
- **Intermediate Layer**:
  - Hooked output from **decoder layer 15** (out of 24 total layers)
  - Dimension: 896

## 2. Topic: Transportation
- **Categories**: 30 types of transportation (e.g., Car, Bus, Truck, Motorcycle, Bicycle, Scooter)
- **Data Generation**:
  - For each category, generate 10 sentences by gpt like:  
    *"I saw a ***car*** on the way to school."*
    *"The ***Bus*** is parked outside the building."*
  - Use Qwen2.5-0.5B to do text continuation to get 300 short stories realated to transportation.

### Prompt Format:
```python
sentence = "I saw a car on the way to school."
prompt = f"Give me a short story that begins with: {sentence} \n\n"
prompt += "Try to include more transportation-related words in the story.\n\n"
prompt += "The story should contain at most 50 words"
```
- **Feature Generation**:
    - Use a hook to extract the activations from **decoder15** for the generated continuation tokens.
    - got **19K** tokens with thier features (19K x 896).

## 3. Sparse Autoencoder: Vanilla Setting
- **Structure**:
    ```python
    z = nn.relu(nn.linear(input))
    output = nn.linear(z)
- **Loss Function**:
  - Mean Squared Error (MSE) + L1 Regularization
- **Dimensions**:
  - Input: 896
  - Latent: 896 x 20
  - Output: 896

### Training result
![train_val_mse Curve](./asset/train_val_mse.png)

![l1_vs_mse](./asset/l1_vs_mse.png)

![activated features for car](./asset/activated_features_car.png)


## 4. Intervention Experiment

- **Original** *(no intervention)*:  
  `LLM layer15 -> x -> LLM layer16`

- **token Activation** *(intervention applied)*:  
  ```python
  z = SAE.encode(x)
  z = z + a * v
  x_modified = SAE.decode(z)

- **token suppress** *(intervention applied)*:  
  ```python
  z = SAE.encode(x)
  z = z - b * v
  x_modified = SAE.decode(z)

---

### Control for multiple-choice questions

**Original Output:**  
**Q:** Generally, which is smaller, a car or a train?  
**A:** *car.*

---

**Activate "train":**  
**Q:** Generally, which is smaller, a car or a train?  
**A:** *train*

---

**Suppress "car":**  
**Q:** Generally, which is smaller, a car or a train?  
**A:** *a train*


### Diverse Answer Control

#### 🔹 Content Control

**Activate `slowly`:**  
Q: Imagine a traffic scene.  
A: As the sun sets over the city, **a steady stream of cars** passes by, honing through the streets. The vehicles are a steady, rhythmic pattern, weaving through the city **like a slow, deliberate dance**. The cars are a mix of red, green, and yellow, as if on a parade down the road.

**Activate `quickly`:**  
Q: Imagine a traffic scene.  
A: A driver is driving through a **busy intersection**, trying to navigate through a jam of cars. **The traffic is moving at a high speed**, with many vehicles trying to make their way through the narrow lanes. The driver is trying to **make a quick decision** to avoid a collision, but is struggling to make a decision.

#### 🔹 Perspective/Style Control

**Activate `feeling`:**  
Q: Imagine a traffic scene.  
A: A bustling city filled with cars and people as the sun sets casting a golden glow over the city skyline. The traffic is heavy with cars honking and horns as people rushing to their destinations or trying to find their way through the city. **The city is a vibrant and dynamic place with its streets and buildings bustling with life**.

#### 🔹 Lexical Insertion

**Activate `glow`:**  
Q: Imagine a traffic scene.  
A: Traffic patterns are chaotic and unpredictable, with cars and buses moving in waves and lines, and pedestrians and cyclists jostling for space. **The sun is high and casting a golden glow over the landscape, but it's not enough to illuminate the road**. The sky is a shade of orange and pink, with clouds and raindrops dancing in the sunlight.

#### 🔹 Semantic Association

**Activate `day`:**  
Q: Imagine a traffic scene.  
A: In a bustling city, a driver is driving through a congested intersection. The driver is **late for a crucial meeting**, and the traffic is jammed with cars, trucks, and buses. The driver must navigate through the traffic, **trying to find a way to reach their destination before the meeting**.


# Areas for Improvement

- Try advanced autoencoders: JumpReLU, GatedSAE, TopKSAE, Batch Top-K.
- Use more reliable metrics like L0 norm,  CE difference.
- Train on a larger dataset.
- Apply to larger LLMs.
- Explore more diverse and interesting intervention cases.


# Reference
- **Assignment statement**: *assignment-4-brain-surgery.pdf*  
- **Claude research**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)  
- **SAE implementations**:  
  - [Dictionary Learning by saprmarks](https://github.com/saprmarks/dictionary_learning)  
  - [SAELens by jbloomAus](https://github.com/jbloomAus/SAELens/tree/main)  
- **Blog overview**: [SAE Intuitions – A brief introduction to SAE in LLMs](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html)  
- **Recent SAE variants**:  
  - [`GatedSAE`](https://arxiv.org/abs/2404.16014) — Rajamanoharan et al., 2024  
  - [`TopKSAE`](https://arxiv.org/abs/2406.04093) — Gao et al., 2024  
  - [`BatchTopKSAE`](https://arxiv.org/abs/2412.06410) — Bussmann et al., 2024  
  - [`JumpReLU`](https://arxiv.org/abs/2407.14435) — Rajamanoharan et al., 2024




