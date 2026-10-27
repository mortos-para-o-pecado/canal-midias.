# Confere um vídeo gerado: quanto mexeu, onde mexeu e se há tremor; grava uma folha com quadros e o mapa do movimento.
# Uso: python3 olhar.py saida/latour_vela.mp4
import sys
import imageio, numpy as np
from PIL import Image
p = sys.argv[1]
f = np.array(imageio.mimread(p, memtest=False)).astype(np.float32)
n, H, W, _ = f.shape
print(p, 'quadros', n, 'tamanho', W, 'x', H)
d0 = np.abs(f - f[0]).mean(axis=(1, 2, 3))
dp = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))
print('distancia do 1o quadro (a cada 12):', ' '.join(f'{x:.1f}' for x in d0[::12]))
print('passo a passo (media, maximo):', f'{dp.mean():.2f}', f'{dp.max():.2f}', 'no quadro', int(dp.argmax()) + 1)
s = f.std(axis=0).mean(axis=2)   # quanto cada ponto variou ao longo do vídeo
gy, gx = 8, 6
print(f'mapa do movimento ({gy} linhas x {gx} colunas, de cima para baixo):')
for i in range(gy):
    print(' '.join(f'{s[i * H // gy:(i + 1) * H // gy, j * W // gx:(j + 1) * W // gx].mean():4.1f}' for j in range(gx)))
# folha: primeiro, meio, último quadro e o mapa do movimento ampliado
mapa = np.clip(s * 12, 0, 255).astype(np.uint8)
mapa = np.stack([mapa, mapa // 2, 255 - mapa], axis=2) * (mapa[..., None] > 8)
folha = np.concatenate([f[0], f[n // 2], f[-1], mapa.astype(np.float32)], axis=1).astype(np.uint8)
saida = p.rsplit('.', 1)[0] + '_folha.png'
Image.fromarray(folha).save(saida)
print('folha:', saida)
