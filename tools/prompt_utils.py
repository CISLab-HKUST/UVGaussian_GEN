import torch

def get_prompt(prompt, dir, region):
    if dir == 0:
        dir_mes = "front"
    elif dir == 1:
        dir_mes = "side"
    else:
        dir_mes = "back"

    dir_prompt = f"{prompt}, {region}, {dir_mes} view, realistic, HDR, 4k, clear, high quality"

    return dir_prompt


def get_embeddings(guidance, prompt, dir, region):
    text_prompt = get_prompt(prompt, dir, region)

    text_embeds = guidance.get_text_embeds(text_prompt)
    null_embeds = guidance.get_text_embeds("")

    return text_embeds, null_embeds