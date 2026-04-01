from model import BannerGenerator
import gradio as gr


gen = BannerGenerator()

def process_banner(image, prompt, neg_prompt, ip_scale, ctrl_scale, guidance, steps):
    result = gen.generate(
        input_image=image,
        prompt=prompt,
        negative_prompt=neg_prompt,
        ip_scale=ip_scale,
        control_scale=ctrl_scale,
        guidance_scale=guidance,
        steps=int(steps)
    )
    return image

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Ad banner Generator (SDLX + ControlNet + Ip-Adapter)")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Product Image")
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="e.g., Professional product photography of a soda can on a marble table..."
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="low quality, blurry, distorted, yellow, gold"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                ip_scale = gr.Slider(0, 1.0, value=0.8, label="IP-Adapter Scale")
                ctrl_scale = gr.Slider(0, 1.0, value=0.8, label="ControlNet Scale")
                guidance = gr.Slider(1.0, 15.0, value=7.5, label="Guidance Scale")
                steps = gr.Number(value=30, label="Steps", precision=0)
            
            run_btn = gr.Button("Generate Banner", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Result")

    run_btn.click(
        fn=process_banner,
        inputs=[input_img, prompt, neg_prompt, ip_scale, ctrl_scale, guidance, steps],
        outputs=[output_img]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)