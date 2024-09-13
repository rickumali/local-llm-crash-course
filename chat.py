import chainlit as cl
from ctransformers import AutoModelForCausalLM


def check_model_changed(string):
    words = string.split()
    if len(words) >= 2 and words[0] == "Model" and words[1] == "changed":
        return True
    else:
        return False


def get_prompt(instruction: str, history: list[str] = None) -> str:
    if instruction.lower() == "forget everything":
        history.clear()
        prompt = "forget everything"
    elif instruction.lower() == "use orca":
        prompt = "Model changed to orca"
    elif instruction.lower() == "use llama":
        prompt = "Model changed to Llama"
    else:
        system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
        prompt = f"### System:\n{system}\n\n### User:\n"
        if len(history) > 0:
            prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
        prompt += f"{instruction}\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    model = cl.user_session.get("model")
    msg = cl.Message(content='')
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    if prompt == "forget everything":
        response = "Uh oh, I've just forgotten our conversation history."
        await msg.stream_token(response)
    elif check_model_changed(prompt):
        await msg.stream_token(prompt)
    else:
        if model == "orca":
            for word in llm_orca(prompt, stream=True):
                await msg.stream_token(word)
                response += word
        elif model == "llama":
            for word in llm_llama(prompt, stream=True):
                await msg.stream_token(word)
                response += word
        else:
            response = f"Uh oh. I don't know {model}"
            await msg.stream_token(response)
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm_orca
    llm_orca = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
    global llm_llama
    llm_llama = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf")
    cl.user_session.set("model", "orca")
