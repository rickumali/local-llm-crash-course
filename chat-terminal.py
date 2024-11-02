from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    if True:
        print(prompt)
    return prompt


while True:
    question = input("Q: ")
    if question.lower() == "quit":
        break
    for word in llm(get_prompt(question), stream=True):
        print(word, end="", flush=True)
    print()
