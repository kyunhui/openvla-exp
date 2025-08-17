import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from PIL import Image

# ===== DummyVLM 정의 =====
class DummyVLM:
    def to(self, device, dtype=None):
        print(f"[DummyVLM] Moving to device: {device}, dtype: {dtype}")
        return self

    def get_prompt_builder(self, system_prompt=None):
        class Builder:
            system_prompt = "Dummy system prompt"

            def add_turn(self, role, message): 
                print(f"[DummyVLM] add_turn: {role}, {message}")
            def get_prompt(self): 
                return "Dummy prompt"
            def get_potential_prompt(self, text): 
                return text
        return Builder()

    def generate(self, image, prompt_text, **kwargs):
        print(f"[DummyVLM] Generating with prompt: {prompt_text}")
        return "Dummy response from VLM"

# ===== GenerateConfig 정의 =====
@dataclass
class GenerateConfig:
    model_path: str
    hf_token: Union[str, Path] = ""  # 토큰 필요 없으므로 빈 문자열

# ===== generate 함수 =====
def generate(cfg: GenerateConfig = GenerateConfig(model_path="dummy_model")) -> None:
    print(f"[INFO] Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")
    dummy_image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    
    user_message = "Hello VLM@"
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vlm = DummyVLM()
    vlm.to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else None)

    vlm_prompt_builder = vlm.get_prompt_builder()
    vlm_prompt_builder.add_turn("user", user_message)
    prompt_text = vlm_prompt_builder.get_prompt()
    
    # Dummy 이미지 생성
    dummy_image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn("user", "Hello VLM!")
    prompt_text = prompt_builder.get_prompt()
    response = vlm.generate(dummy_image, prompt_text)
    
    dummy_image.show()

    print(f"[RESULT] {response}")


if __name__ == "__main__":
    generate()