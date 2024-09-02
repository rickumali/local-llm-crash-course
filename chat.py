from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

prompt = "Responding only with the correct answer, what bustling city is the home of Red Fort and India Gate?"

print(prompt + llm(prompt))
