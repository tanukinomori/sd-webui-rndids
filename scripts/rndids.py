import modules
import gradio
import random

class Script(modules.scripts.Script):

    def __init__(self):
        self.enable = False

    def title(self):
        return "RndIds"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):      
        with gradio.Accordion("RndIds", open=False):
            enable = gradio.Checkbox(value=False, label="Enable RndIds", interactive=True)
        return [enable]

    def process(self, p: modules.processing.StableDiffusionProcessing, enable: bool):

        if not enable:
            return

        try:

            tokenizer = modules.shared.sd_model.cond_stage_model.tokenizer

            random.seed(p.seed)

            stop = tokenizer.encoder['<|startoftext|>']

            ids = []
            for i in range(75):
                ids.append(random.randrange(stop))

            p.all_prompts[0] += "\nBREAK\n" + tokenizer.decode(ids)

        except Exception as e:
            import traceback
            traceback.print_exc()
