import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from rembg import remove

class BannerGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.create_pipe()

    def create_pipe(self):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch.float16
        ).to(self.device)

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", 
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        self.mask_processor = IPAdapterMaskProcessor() 

    def generate_edge_map(self, image : Image.Image):
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 150, 300)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        canny_image = Image.fromarray(edges)
        return canny_image
    
    def create_mask(self, image : Image.Image):
        alpha = image.split()[3]
        mask = Image.new("L", image.size, 0) 
        white = Image.new("L", image.size, 255) 
        mask = Image.composite(white, mask, alpha)
        return mask

    def generate(self, 
                 input_image: Image.Image, 
                 prompt: str, 
                 negative_prompt: str = "",
                 ip_scale: float = 0.7, 
                 control_scale: float = 0.8,
                 guidance_scale: float = 7.5,
                 steps: int = 30):

        self.pipe.set_ip_adapter_scale(ip_scale)

        only_product_img = remove(input_image)
    
        msk = self.create_mask(only_product_img)
        canny_image = self.generate_edge_map(input_image)
        processed_mask = self.mask_processor.preprocess(
            [msk], 
            height=1024, 
            width=1024
        )
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,     
            ip_adapter_image=only_product_img,
            controlnet_conditioning_scale=control_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={"ip_adapter_masks": [processed_mask]}
        ).images[0]
        return image
