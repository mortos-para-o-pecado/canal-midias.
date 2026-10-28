# Movimentos locais de uma pintura, sem IA e sem tocar em rosto: chama que tremula, fumaça que ondula,
# poeira que flutua na luz. Cada efeito age só dentro da sua região (com bordas suaves).
import math
import numpy as np, cv2

def _janela(h, w, x0, y0, x1, y1, borda):
    # 1 dentro da região, caindo suavemente até 0 fora dela
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0); dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0)
    return np.exp(-((dx / borda) ** 2 + (dy / borda) ** 2)), xs, ys

class Chama:
    """Chama de vela: balança de lado (mais na ponta), estica e encolhe um pouco, e pisca de brilho."""
    def __init__(self, shape, caixa, base_y, ponta_y, amp=3.0, seed=1):
        h, w = shape[:2]; x0, y0, x1, y1 = caixa
        self.jan, self.xs, self.ys = _janela(h, w, x0, y0, x1, y1, borda=max(6, (x1 - x0) * 0.6))
        self.alt = np.clip((base_y - self.ys) / max(1, base_y - ponta_y), 0, 1.2)   # 0 na base, 1 na ponta
        self.base_y, self.amp = base_y, amp
        self.cx = (x0 + x1) / 2
        r = np.random.default_rng(seed); self.fase = r.uniform(0, 6.28, 4); self.r = r
    def campo(self, t):
        a = self.alt; f = self.fase
        lado = self.amp * a ** 1.6 * (0.6 * np.sin(2 * math.pi * 1.7 * t + f[0] + 5 * a) + 0.4 * np.sin(2 * math.pi * 3.1 * t + f[1] + 9 * a))
        estica = 0.05 * math.sin(2 * math.pi * 2.3 * t + f[2]) + 0.03 * math.sin(2 * math.pi * 5.3 * t + f[3])
        dy = (self.ys - self.base_y) * estica                        # estica a partir da base
        return lado * self.jan, dy * self.jan
    def brilho(self, t):
        return 1 + 0.06 * math.sin(2 * math.pi * 2.9 * t + self.fase[0]) + 0.04 * math.sin(2 * math.pi * 7.1 * t + self.fase[2])

class Fumaca:
    """Fio de fumaça: ondula devagar, mais solto no alto."""
    def __init__(self, shape, caixa, amp=4.0, seed=2):
        h, w = shape[:2]; x0, y0, x1, y1 = caixa
        self.jan, self.xs, self.ys = _janela(h, w, x0, y0, x1, y1, borda=max(8, (x1 - x0) * 0.5))
        self.sol = np.clip((y1 - self.ys) / max(1, y1 - y0), 0, 1)     # 0 embaixo, 1 no alto
        self.amp = amp; self.fase = np.random.default_rng(seed).uniform(0, 6.28, 2)
    def campo(self, t):
        s = self.sol
        dx = self.amp * s * (np.sin(2 * math.pi * 0.35 * t - 7 * s + self.fase[0]) + 0.35 * np.sin(2 * math.pi * 0.8 * t - 13 * s + self.fase[1]))
        return dx * self.jan, -1.5 * s * self.jan * math.sin(2 * math.pi * 0.2 * t)

class Poeira:
    """Partículas de poeira flutuando devagar na luz (só aparecem onde a pintura é clara)."""
    def __init__(self, img, n=70, seed=3, caixa=None):
        h, w = img.shape[:2]; r = np.random.default_rng(seed)
        lum = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (0, 0), 25)
        x0, y0, x1, y1 = caixa or (0, 0, w, h)
        self.p = np.stack([r.uniform(x0, x1, n), r.uniform(y0, y1, n)], 1)
        self.v = np.stack([r.normal(0.0, 2.0, n), r.uniform(-6, -2, n)], 1)       # pixels por segundo, sobe devagar
        self.tam = r.uniform(1.6, 3.6, n) * w / 1000; self.fase = r.uniform(0, 6.28, n)
        self.lum, self.h, self.w = lum, h, w; self.caixa = (x0, y0, x1, y1)
    def aplica(self, q, t):
        out = q.astype(np.float32); x0, y0, x1, y1 = self.caixa
        for (px, py), (vx, vy), s, f in zip(self.p, self.v, self.tam, self.fase):
            x = x0 + (px - x0 + vx * t + 3 * math.sin(0.7 * t + f)) % (x1 - x0)
            y = y0 + (py - y0 + vy * t) % (y1 - y0)
            ix, iy = int(x), int(y)
            if not (0 <= ix < self.w and 0 <= iy < self.h): continue
            a = min(0.85, 1.1 * self.lum[iy, ix]) * (0.55 + 0.45 * math.sin(2.1 * t + f))
            R = max(2, int(s * 3))
            ya, yb, xa, xb = max(0, iy - R), min(self.h, iy + R + 1), max(0, ix - R), min(self.w, ix + R + 1)
            gy, gx = np.mgrid[ya:yb, xa:xb]
            g = np.exp(-(((gx - x) ** 2 + (gy - y) ** 2) / (2 * s * s)))[..., None] * a
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - g) + 255 * g * np.array([1.0, 0.93, 0.8])
        return np.clip(out, 0, 255).astype(np.uint8)

def aplica_campos(q, efeitos, t):
    """Soma os deslocamentos dos efeitos e aplica de uma vez; depois o brilho da chama e a poeira."""
    h, w = q.shape[:2]
    dx = np.zeros((h, w), np.float32); dy = np.zeros((h, w), np.float32); jan_brilho = None; fator = 1.0
    for e in efeitos:
        if hasattr(e, 'campo'):
            a, b = e.campo(t); dx += a; dy += b
        if isinstance(e, Chama): jan_brilho, fator = e.jan, e.brilho(t)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    r = cv2.remap(q, xs - dx, ys - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) if np.abs(dx).max() + np.abs(dy).max() > 0 else q
    if jan_brilho is not None:
        r = np.clip(r.astype(np.float32) * (1 + (fator - 1) * jan_brilho[..., None]), 0, 255).astype(np.uint8)
    for e in efeitos:
        if isinstance(e, Poeira): r = e.aplica(r, t)
    return r
