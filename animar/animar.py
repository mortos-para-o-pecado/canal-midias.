# Teste de animação real de pinturas (LTX-Video, licença OpenRail-M/Apache) numa GPU T4 gratuita do Google Colab.
# Uso no Colab (uma célula):  !wget -q <url>/animar/animar.py -O animar.py   e depois   %run animar.py
# Gera saida/<nome>.mp4 (4 s, 24 fps) e mostra no notebook. O texto (T5) roda primeiro e é descartado,
# para caber na memória do Colab gratuito.
import os, subprocess, sys, gc, time
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'diffusers>=0.33', 'transformers', 'accelerate', 'sentencepiece', 'imageio', 'imageio-ffmpeg'], check=True)
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer

BASE = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/animar/'
NEG = ('worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, deformed hands, extra fingers, '
       'morphing, melting, text, watermark, camera shake, fast motion')
CENAS = {
    'latour_vela': ('A classical oil painting comes to life: the candle flame flickers softly and its warm light trembles on the young woman\'s face, '
                    'her hand and the old books; she breathes slowly and blinks once; the smoke rises gently. Static camera, very subtle natural motion, '
                    'baroque chiaroscuro oil painting texture preserved.', 512, 672, .5),
    'paulo_valentin': ('A classical oil painting comes to life: an old bearded apostle sits at his desk, he breathes slowly, his head turns slightly '
                       'toward the light, the dust floats in the warm light from the window. Static camera, very subtle natural motion, '
                       'Rembrandt oil painting texture preserved.', 576, 672, .35),
    'pentecostes': ('A classical oil painting comes to life: tongues of fire flicker above the heads of the apostles, the light from the dove above '
                    'glows and pulses softly, robes move slightly. Slow gentle push-in camera, subtle natural motion, El Greco oil painting texture preserved.', 512, 768, .2),
}
os.makedirs('saida', exist_ok=True)
dt = torch.bfloat16

# 1) texto -> embeddings (T5 na GPU), depois libera a memória
tok = T5Tokenizer.from_pretrained('Lightricks/LTX-Video', subfolder='tokenizer')
enc = T5EncoderModel.from_pretrained('Lightricks/LTX-Video', subfolder='text_encoder', torch_dtype=dt, device_map='cuda', low_cpu_mem_usage=True)   # direto para a GPU, pedaço por pedaço (a RAM do Colab gratuito não aguenta tudo de uma vez)
def emb(txt):
    t = tok(txt, padding='max_length', max_length=128, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad(): e = enc(t.input_ids, attention_mask=t.attention_mask)[0]
    return e.to(dt), t.attention_mask
E = {k: emb(v[0]) for k, v in CENAS.items()}; EN = emb(NEG)
del enc; gc.collect(); torch.cuda.empty_cache()

# 2) imagem -> vídeo
pipe = LTXImageToVideoPipeline.from_pretrained('Lightricks/LTX-Video', text_encoder=None, tokenizer=None, torch_dtype=dt)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
def recorta(img, w, h, fy):
    # recorta na proporção do vídeo (sem esticar), com o centro vertical em fy (0 = topo)
    W, H = img.size; alvo = w / h
    if W / H > alvo: nw = int(H * alvo); x0 = (W - nw) // 2; img = img.crop((x0, 0, x0 + nw, H))
    else:
        nh = int(W / alvo); y0 = int(min(max(0, fy * H - nh / 2), H - nh)); img = img.crop((0, y0, W, y0 + nh))
    return img.resize((w, h))
for nome, (txt, w, h, fy) in CENAS.items():
    t0 = time.time()
    img = recorta(load_image(BASE + nome + '.jpg'), w, h, fy)
    (pe, pm), (ne, nm) = E[nome], EN
    v = pipe(image=img, prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
             width=w, height=h, num_frames=97, num_inference_steps=40, guidance_scale=3.0,
             generator=torch.Generator('cuda').manual_seed(7)).frames[0]
    export_to_video(v, f'saida/{nome}.mp4', fps=24)
    print(f'{nome}: pronto em {time.time() - t0:.0f} s', flush=True)

from IPython.display import HTML, display
from base64 import b64encode
for nome in CENAS:
    d = b64encode(open(f'saida/{nome}.mp4', 'rb').read()).decode()
    display(HTML(f'<p>{nome}</p><video width=360 controls loop autoplay muted src="data:video/mp4;base64,{d}"></video>'))
