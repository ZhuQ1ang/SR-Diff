
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BLIPFeatureExtractor:
    def __init__(self, model_path=r"D:\OneDrive\Desktop\The visual computer\sr-diff\blip_model"):

        self.processor = BlipProcessor.from_pretrained(model_path, use_fast=True)

        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()
        
    def get_cls_embedding(self, image: Image.Image) -> torch.Tensor:

        inputs = self.processor(
            images=image,
            text="",            # prompt，用于启动文本编码
            return_tensors="pt",
            padding="max_length",
            max_length=30,
        ).to(device)

        with torch.no_grad():

            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values
            )
            img_embeds = vision_outputs.last_hidden_state
            img_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=device)


            decoder_outputs = self.model.text_decoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                encoder_hidden_states=img_embeds,
                encoder_attention_mask=img_mask,
                return_dict=True,
                output_hidden_states=True,
            )


            cls_embedding = decoder_outputs.hidden_states[-1][:, 0, :]  


        return cls_embedding.cpu()
    
Blipmodel = BLIPFeatureExtractor()
