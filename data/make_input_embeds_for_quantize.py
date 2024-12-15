import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import snapshot_download
import inquirer

class MakeInputEmbeds:
    def __init__(self):
        self.inputs = [
            inquirer.List("model", 
                            message="Which Qwen2VL model would you like to convert?", 
                            choices=["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"], 
                            default="Qwen/Qwen2-VL-2B-Instruct")
        ]
        
        self.model_id = inquirer.prompt(self.inputs)["model"]
        self.model_name = self.model_id.split("/", 1)[1]
        self.path = snapshot_download(self.model_id, local_dir=self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.path, torch_dtype="auto", device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()

        processor = AutoProcessor.from_pretrained(self.path)

        datasets = json.load(open("data/datasets.json", 'r'))
        for data in datasets:
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(data["image_path"], data["image"])
            image = Image.open(imgp)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
            pixel_values = inputs["pixel_values"].type(self.model.visual.get_dtype())
            image_mask = inputs["input_ids"] == self.model.config.image_token_id
            image_embeds = self.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
            inputs_embeds[image_mask] = image_embeds
            print("inputs_embeds", inputs_embeds.shape)
            os.makedirs("data/inputs_embeds/", exist_ok=True)
            np.save("data/inputs_embeds/{}".format(image_name), inputs_embeds.to(dtype=torch.float16).cpu().detach().numpy())
            
        with open('data/inputs.json', 'w') as json_file:
            json_file.write('[\n')
            first = True
            for data in tqdm(datasets):
                input_embed = np.load(os.path.join("data/inputs_embeds", data["image"].split(".")[0]+'.npy'))
                target = data["target"]
                input_dict = {
                    "input_embed": input_embed.tolist(),
                    "target": target
                }
                if not first:
                    json_file.write(',\n')
                else:
                    first = False
                json.dump(input_dict, json_file)
            json_file.write('\n]')

        print("Done")
