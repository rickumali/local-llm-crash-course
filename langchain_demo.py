from langchain_community.llms import CTransformers

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_tokens=20,
)

print(llm.invoke("Which city is the capital of India?"))
