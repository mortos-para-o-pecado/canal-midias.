# Segunda versão do teste de animação real de pinturas, numa GPU T4 gratuita do Google Colab.
# Mudanças em relação ao animar.py (que deu vídeos quase parados):
#  - modelo LTX-Video 0.9.5 (2B, mesmo tamanho) com o LTXConditionPipeline, bem melhor em imagem -> vídeo;
#  - a imagem de partida passa por uma compressão de vídeo leve (crf 29), como no código oficial da Lightricks:
#    sem isso o modelo tende a "congelar" a pintura;
#  - pedidos descritivos, no estilo recomendado pelo modelo (o que se mexe, como, e a câmera parada);
#  - decode_timestep / decode_noise_scale recomendados para as versões 0.9.1 em diante.
# Uso no Colab (uma célula):  !wget -q <url>/animar/animar2.py -O animar2.py   e depois   %run animar2.py
import os, subprocess, sys, time, json
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'diffusers>=0.33', 'transformers', 'accelerate', 'sentencepiece', 'imageio', 'imageio-ffmpeg'], check=True)
import torch
import torch.nn.functional as F
# A T4 não tem, em bf16, o atalho de memória da atenção: a atenção é feita em fatias de 1024 posições.
_sdpa = F.scaled_dot_product_attention
def _sdpa_fatiado(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    q, k, v = query, key, value; n = q.shape[-2]
    if n <= 1024 or is_causal:
        return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    out = []
    for i in range(0, n, 1024):
        m = attn_mask[..., i:i + 1024, :] if (attn_mask is not None and attn_mask.dim() >= 2 and attn_mask.shape[-2] == n) else attn_mask
        out.append(_sdpa(q[..., i:i + 1024, :], k, v, attn_mask=m, dropout_p=dropout_p, scale=scale, **kw))
    return torch.cat(out, dim=-2)
F.scaled_dot_product_attention = _sdpa_fatiado
from IPython.display import HTML, display
from base64 import b64encode
from PIL import Image
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers.utils import export_to_video, load_image

MODELO = 'Lightricks/LTX-Video-0.9.5'
BASE = 'https://raw.githubusercontent.com/mortos-para-o-pecado/canal-midias./main/animar/'
NEG = 'worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, deformed hands, extra fingers, morphing, text, watermark'
ESTILO = (' The camera remains stationary. The scene keeps the look of a seventeenth-century oil painting, '
          'with deep chiaroscuro shadows, warm candle-like light and visible brushstrokes.')
CENAS = {
    'latour_vela': ('A young woman with long dark hair sits in a dark room beside a single candle, her chin resting on her hand. '
                    'She breathes slowly, her shoulders rising and falling gently, then she slowly lifts her gaze toward the candle flame and blinks. '
                    'The candle flame flickers and sways softly, and a thin wisp of smoke rises and curls upward. '
                    'The warm candlelight trembles across her face, her white blouse and the old books on the table.' + ESTILO, 512, 672, .5),
    'paulo_valentin': ('An old apostle with a white beard and a brown robe sits at a wooden desk in a dim room, his right hand resting on an open book. '
                       'He breathes slowly, lowers his eyes toward the pages and slightly turns his head, deep in thought. '
                       'Warm light falls from the upper left across his face and beard, and fine dust floats slowly in the light.' + ESTILO, 576, 672, .35),
    'pentecostes': ('A crowd of apostles gathered around a woman in blue robes looks upward in awe. Small tongues of fire flicker and dance above their heads, '
                    'and a white dove at the top glows with a soft pulsing light that grows brighter. Their robes move slightly and a few of them raise their hands.'
                    ' The camera slowly pushes in. The scene keeps the look of an El Greco oil painting, with elongated figures, vivid colors and visible brushstrokes.', 512, 768, .2),
}
QUADROS, PASSOS = 97, 40
os.makedirs('saida2', exist_ok=True)
dt = torch.bfloat16

# 1) texto -> embeddings num processo separado (quando ele termina, a GPU fica livre para o vídeo)
TEXTO = r"""
import torch, json, sys
from transformers import T5EncoderModel, T5Tokenizer
dt = torch.bfloat16
M = sys.argv[1]
tok = T5Tokenizer.from_pretrained(M, subfolder='tokenizer')
enc = T5EncoderModel.from_pretrained(M, subfolder='text_encoder', dtype=dt, device_map='cuda', low_cpu_mem_usage=True)
txts = json.load(open('textos2.json')); out = {}
for k, txt in txts.items():
    t = tok(txt, padding='max_length', max_length=256, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad(): e = enc(t.input_ids, attention_mask=t.attention_mask)[0]
    out[k] = (e.to(dt).cpu(), t.attention_mask.cpu())
torch.save(out, 'emb2.pt'); print('textos prontos', flush=True)
"""
json.dump(dict({k: v[0] for k, v in CENAS.items()}, _neg=NEG), open('textos2.json', 'w'))
open('texto2.py', 'w').write(TEXTO)
r = subprocess.run([sys.executable, 'texto2.py', MODELO], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-600:] if r.returncode else '', flush=True)
if r.returncode: raise SystemExit('o texto falhou: reinicie a sessão (Ambiente de execução > Reiniciar sessão) e rode de novo')
EMB = torch.load('emb2.pt')
E = {k: (EMB[k][0].to('cuda'), EMB[k][1].to('cuda')) for k in CENAS}; EN = (EMB['_neg'][0].to('cuda'), EMB['_neg'][1].to('cuda'))

# 2) imagem -> vídeo
pipe = LTXConditionPipeline.from_pretrained(MODELO, text_encoder=None, tokenizer=None, torch_dtype=dt)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()

def recorta(img, w, h, fy):
    # recorta na proporção do vídeo (sem esticar), com o centro vertical em fy (0 = topo)
    W, H = img.size; alvo = w / h
    if W / H > alvo: nw = int(H * alvo); x0 = (W - nw) // 2; img = img.crop((x0, 0, x0 + nw, H))
    else:
        nh = int(W / alvo); y0 = int(min(max(0, fy * H - nh / 2), H - nh)); img = img.crop((0, y0, W, y0 + nh))
    return img.convert('RGB').resize((w, h), Image.LANCZOS)

def comprime(img, nome):
    # passa a imagem por um quadro de vídeo H.264 (crf 29), como o código oficial faz na imagem de partida
    a, b, c = f'saida2/{nome}_in.png', f'saida2/{nome}_in.mp4', f'saida2/{nome}_cond.png'
    img.save(a)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a, '-c:v', 'libx264', '-crf', '29', '-pix_fmt', 'yuv420p', b], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', b, '-frames:v', '1', c], check=True)
    return Image.open(c).convert('RGB')

def mostra(nome):
    d = b64encode(open(f'saida2/{nome}.mp4', 'rb').read()).decode()
    display(HTML(f'<p>{nome}</p><video width=360 controls loop autoplay muted src="data:video/mp4;base64,{d}"></video>'))

for nome, (txt, w, h, fy) in CENAS.items():
    if os.path.exists(f'saida2/{nome}.mp4'): mostra(nome); continue
    t0 = time.time()
    print(nome, '| GPU livre:', round(torch.cuda.mem_get_info()[0] / 2**30, 1), 'GB', flush=True)
    img = comprime(recorta(load_image(BASE + nome + '.jpg'), w, h, fy), nome)
    (pe, pm), (ne, nm) = E[nome], EN
    try:
        v = pipe(conditions=[LTXVideoCondition(image=img, frame_index=0)],
                 prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
                 width=w, height=h, num_frames=QUADROS, frame_rate=24, num_inference_steps=PASSOS, guidance_scale=3.0,
                 decode_timestep=0.05, decode_noise_scale=0.025,
                 generator=torch.Generator('cuda').manual_seed(7)).frames[0]
    except Exception as ex:
        print('ERRO', nome, type(ex).__name__, str(ex)[:300], 'GPU livre:', round(torch.cuda.mem_get_info()[0] / 2**30, 1), 'GB', flush=True)
        torch.cuda.empty_cache(); continue
    export_to_video(v, f'saida2/{nome}.mp4', fps=24)
    print(f'{nome}: pronto em {time.time() - t0:.0f} s', flush=True); mostra(nome)
