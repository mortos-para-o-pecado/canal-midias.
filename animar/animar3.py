# Terceira rodada: meio-termo entre o quase parado (animar.py) e o exagerado (animar2.py, em que o rosto virou
# o de uma atriz de foto e a mão saiu do queixo).
#  - a pintura é presa no primeiro E no último quadro (o vídeo vira um laço que começa e termina no quadro original);
#  - os pedidos só pedem movimentos pequenos (chama, fumaça, respiração, luz);
#  - o pedido negativo afasta "foto", "atriz", "sorriso", "olhar para a câmera", "rosto mudando";
#  - 3 s (73 quadros) e 30 passos, para caber melhor no tempo da placa gratuita.
# Rodar no Colab DEPOIS do animar2.py, na mesma sessão, com:  %run -i animar3.py   (reaproveita o modelo já carregado)
import os, json, subprocess, sys, time
import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image

NEG3 = (NEG + ', photograph, photorealistic, modern person, actress, smiling, open mouth, looking at the camera, '
        'face changing into another person, hand moving away, new objects')
PINTURA = (' The camera remains stationary. This is a seventeenth-century oil painting gently coming alive, not a photograph: '
           'deep chiaroscuro shadows, warm light, visible brushstrokes and canvas texture are preserved in every frame.')
CENAS3 = {
    'latour_vela': ('A young woman with long dark hair sits motionless beside a single candle, her chin resting on her hand, lost in thought; '
                    'only her quiet breathing makes her shoulders rise and fall very slightly. The candle flame flickers and sways, '
                    'a thin wisp of smoke rises and curls, and the warm candlelight trembles softly on her face and on the old books.' + PINTURA,
                    512, 672, .5),
    'paulo_valentin': ('An old apostle with a white beard sits still at his wooden desk, deep in thought, his hand resting on an open book; '
                       'he breathes slowly and his eyelids lower slightly. The warm light from the upper left flickers gently '
                       'and fine dust floats slowly through the light.' + PINTURA, 576, 672, .35),
    'pentecostes': ('Small tongues of fire flicker and dance above the heads of the gathered apostles, and the white dove at the top glows '
                    'with a soft pulsing light. The figures remain still; only their robes move very slightly in a gentle wind. '
                    'The camera slowly pushes in. This is an El Greco oil painting gently coming alive, not a photograph: elongated figures, '
                    'vivid colors and visible brushstrokes are preserved in every frame.', 512, 768, .2),
}
QUADROS3, PASSOS3 = 73, 30
os.makedirs('saida3', exist_ok=True)

# textos novos (o T5 roda num processo à parte e libera a placa ao terminar)
json.dump(dict({k: v[0] for k, v in CENAS3.items()}, _neg=NEG3), open('textos3.json', 'w'))
open('texto3.py', 'w').write(TEXTO.replace('textos2.json', 'textos3.json').replace('emb2.pt', 'emb3.pt'))
r = subprocess.run([sys.executable, 'texto3.py', MODELO], capture_output=True, text=True)
print(r.stdout[-200:], r.stderr[-600:] if r.returncode else '', flush=True)
if r.returncode: raise SystemExit('o texto falhou')
EMB3 = torch.load('emb3.pt')
E3 = {k: (EMB3[k][0].to('cuda'), EMB3[k][1].to('cuda')) for k in CENAS3}; EN3 = (EMB3['_neg'][0].to('cuda'), EMB3['_neg'][1].to('cuda'))

def baixa(nome):
    # a conexão com o GitHub caiu uma vez no meio: tenta de novo antes de desistir
    for t in range(5):
        try: return load_image(BASE + nome + '.jpg')
        except Exception as ex:
            print('  nova tentativa de baixar', nome, type(ex).__name__, flush=True); time.sleep(5)
    raise RuntimeError('não consegui baixar ' + nome)

def mostra3(nome):
    d = b64encode(open(f'saida3/{nome}.mp4', 'rb').read()).decode()
    display(HTML(f'<p>{nome} (v3)</p><video width=360 controls loop autoplay muted src="data:video/mp4;base64,{d}"></video>'))

for nome, (txt, w, h, fy) in CENAS3.items():
    if os.path.exists(f'saida3/{nome}.mp4'): mostra3(nome); continue
    t0 = time.time()
    print(nome, '| GPU livre:', round(torch.cuda.mem_get_info()[0] / 2**30, 1), 'GB', flush=True)
    try:
        img = comprime(recorta(baixa(nome), w, h, fy), nome)
        (pe, pm), (ne, nm) = E3[nome], EN3
        v = pipe(conditions=[LTXVideoCondition(image=img, frame_index=0), LTXVideoCondition(image=img, frame_index=QUADROS3 - 1)],
                 prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
                 width=w, height=h, num_frames=QUADROS3, frame_rate=24, num_inference_steps=PASSOS3, guidance_scale=3.0,
                 decode_timestep=0.05, decode_noise_scale=0.025,
                 generator=torch.Generator('cuda').manual_seed(7)).frames[0]
    except Exception as ex:
        print('ERRO', nome, type(ex).__name__, str(ex)[:300], flush=True); torch.cuda.empty_cache(); continue
    export_to_video(v, f'saida3/{nome}.mp4', fps=24)
    print(f'{nome}: pronto em {time.time() - t0:.0f} s', flush=True); mostra3(nome)
