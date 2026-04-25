import google.genai as genai
from tqdm import tqdm
from PIL import Image
import random
import os
from dotenv import load_dotenv
import json
import pandas as pd

EVAL_PROMPT = """Ты — эксперт в графическом дизайне и рекламе. Оцени сгенерированный баннер по сравнению с исходным товаром.
Выдай ответ в формате JSON:
{
"object_preservation": 1-10,
"prompt_adherence": 1-10,
"aesthetic_quality": 1-10,
"background_relevance": 1-10,
"reasoning": "краткое пояснение"
}
"""

prompts = ["High-resolution studio photograph, centered on a matte grey background. Even, diffused professional lighting, sharp focus, 8k, hyper-realistic, commercial photography.",
          "A person’s hand holding the item in a modern, naturally lit kitchen setting. Shallow depth of field, warm morning tones, high quality, soft bokeh, authentic atmosphere.",
          "Sleek, minimalist shot resting on a smooth concrete plinth. Monochromatic color palette, soft geometric shadows, sophisticated atmosphere, architectural aesthetic.",
          "Dramatic photography with strong rim lighting. Dark obsidian background, glossy finish reflections, moody and luxurious feel, cinematic contrast.",
          "Nestled among lush green monstera leaves and natural wood textures. Sun-dappled lighting, fresh organic vibe, outdoor garden setting, high detail.",
          "Suspended against a dark, slightly textured background with subtle blue and purple LED light accents. Clean lines, futuristic aesthetic, sharp metallic details.",
          "A carefully curated overhead flat lay photograph. Soft even lighting, structured layout, Instagram aesthetic, lifestyle items surrounding the center.",
          "Energetic, dynamic shot with splashes of water and droplets frozen in time around the base. Bright colors, fast shutter speed, fresh and lively presentation.",
          "Displayed on a reflective black marble surface next to a draped silk cloth. Rich, warm gold accents, elegant bokeh background, upscale boutique feel.",
          "A surreal, creative composition floating among abstract 3D geometric shapes. Soft pastel lighting, dreamlike quality, artistic interpretation, soft shadows."
         ]

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

def evaluate_image(original_img, generated_img, prompt):
    response = client.models.generate_content(
        model='models/gemini-latest-flash',
        contents=[
            eval_prompt,
            original_img, 
            generated_img
        ],
        config={
            'response_mime_type': 'application/json'
        }
    )
    return response.text


# only images should be stored in dirs
def eval_model(orig_images_dir, model_images_dir, output_file='metrics.csv'):
    orig_files = os.listdir(orig_images_dir)
    model_files = os.listdir(model_images_dir)
    eval_result = []
    for i in range(len(files)):
        img1_path = os.path.join(orig_images_dir, orig_files[i])
        img2_path = os.path.join(model_images_dir, model_files[i])
        img1 = Image.open(img1_path).resize((1024, 1024))
        img2 = Image.open(img2_path)
        eval_result.append(evaluate_image(img1, img2, prompts[i % 10]))
    l = []
    for i in eval_result:
        x = json.loads(i)
        l.append(x)
    table = pd.DataFrame(l)
    table.to_csv('metrics.csv', index=True)
    return eval_result