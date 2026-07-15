import numpy as np
import faiss, torch, gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DOCS = [
    "The Saudi Space Agency (SSA) was established in 2018 by the Council of Ministers to develop and regulate the Kingdom's space sector.",
    "The Saudi Space Agency supports scientific research, innovation, satellite technologies, and international cooperation in space exploration.",
    "Saudi Vision 2030 considers the space sector a strategic field for economic diversification and technological development.",
    "The Saudi Astronaut Program was launched to prepare qualified Saudi astronauts for scientific missions and human spaceflight.",
    "Rayyanah Barnawi became the first Saudi woman to travel to space in 2023 during the Axiom Mission 2 (Ax-2).",
    "Ali AlQarni participated in the Ax-2 mission as a Saudi astronaut representing the Kingdom of Saudi Arabia.",
    "The Ax-2 mission included scientific experiments in health, biology, and microgravity conducted by the Saudi astronauts.",
    "SaudiSat is a series of Saudi satellites developed for communications, Earth observation, scientific research, and technology testing.",
    "Earth observation satellites help monitor agriculture, weather, urban development, and environmental changes.",
    "Satellite communication systems improve connectivity in remote regions across Saudi Arabia.",
    "Space technologies support disaster management, navigation, mapping, and climate monitoring.",
    "The Saudi Space Agency collaborates with local universities to encourage education and research in aerospace engineering.",
    "Saudi Arabia works with international space organizations to exchange expertise and develop future space missions.",
    "The Kingdom aims to develop local talent and create new jobs in the growing space economy.",
    "Artificial intelligence plays an important role in satellite image analysis and space data processing.",
    "Remote sensing technologies help monitor natural resources, deserts, water, and vegetation in Saudi Arabia.",
    "The future goals of the Saudi Space Agency include expanding satellite programs, supporting innovation, and strengthening the national space industry.",
    "Saudi Arabia continues investing in advanced technologies to become a regional leader in the space sector.",
    "The Saudi space sector contributes to scientific research, economic growth, and sustainable development.",
    "The Saudi Space Agency encourages young people to pursue careers in science, engineering, artificial intelligence, and space technology."
    ]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = np.asarray(embedder.encode(DOCS, normalize_embeddings=True), dtype="float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

GEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(GEN_ID)
model = AutoModelForCausalLM.from_pretrained(GEN_ID, torch_dtype=torch.float32)

def retrieve(query, k=3):
    q = np.asarray(embedder.encode([query], normalize_embeddings=True), dtype="float32")
    scores, idxs = index.search(q, k)
    return [(DOCS[i], float(s)) for i, s in zip(idxs[0], scores[0])]

def generate(prompt, max_new_tokens=256):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

def rag_answer(question, k=3, min_score=0.15):
    chunks = retrieve(question, k=k)
    if not chunks or chunks[0][1] < min_score:
        return "I dont know. (No relevant context was found.)", []
    ctx = "\n".join(f"[{i+1}] {t}" for i, (t, _) in enumerate(chunks))
    prompt = ("Answer using ONLY the context below. If the answer is not in the context, "
              "say: I dont know. Cite sources like [1], [2].\n\n"
              f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")
    return generate(prompt), [t for t, _ in chunks]

def chat_fn(question):
    answer, sources = rag_answer(question)
    srcs = "\n".join(f"{i}. {s}" for i, s in enumerate(sources, 1)) or "(none)"
    return answer, srcs

demo = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(label="Ask about the bootcamp"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="Saudi Space AI Assistant",
)

if __name__ == "__main__":
    demo.launch()
