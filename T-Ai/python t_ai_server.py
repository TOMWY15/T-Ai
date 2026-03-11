"""
T-Ai 1.0 — Server locale
Connette l'interfaccia HTML al modello Python.

INSTALLAZIONE:
    pip install torch flask flask-cors

AVVIO:
    python t_ai_server.py

Poi apri index.html nel browser e seleziona modalita "Locale".
"""

import sys
import os

# ── controlla dipendenze ─────────────────────────────────────────────
missing = []
try:    import torch
except: missing.append("torch")
try:    import flask
except: missing.append("flask")
try:    import flask_cors
except: missing.append("flask-cors")

if missing:
    print(f"\n  ERRORE: librerie mancanti: {', '.join(missing)}")
    print(f"  Esegui:  pip install {' '.join(missing)}\n")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# ════════════════════════════════════════════════════════════════════
#  STESSA ARCHITETTURA di t_ai_model.py
# ════════════════════════════════════════════════════════════════════

MODEL_FILE  = "t_ai_1_0.pt"
MODEL_NAME  = "T-Ai 1.0"

VOCAB_SIZE   = 256
CONTEXT_LEN  = 64
EMBED_DIM    = 128
NUM_HEADS    = 4
NUM_LAYERS   = 3
FF_DIM       = 512
DROPOUT      = 0.1
MAX_TOKENS   = 120
TEMPERATURE  = 0.85
TOP_K        = 30

def encode(text): return [min(ord(c), 255) for c in text]
def decode(tokens): return "".join(chr(t) for t in tokens)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads  = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv  = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(EMBED_DIM, dim=2)
        def sh(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        sc = sc.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        at = self.drop(F.softmax(sc, dim=-1))
        out = (at @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.ff   = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM), nn.GELU(),
            nn.Linear(FF_DIM, EMBED_DIM), nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TAiModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.ln      = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        x    = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=MAX_TOKENS, temp=TEMPERATURE, top_k=TOP_K):
        for _ in range(max_new):
            ctx = idx[:, -CONTEXT_LEN:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt   = torch.multinomial(probs, 1)
            idx   = torch.cat([idx, nxt], dim=1)
        return idx


# ════════════════════════════════════════════════════════════════════
#  CARICA MODELLO
# ════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = None

if os.path.exists(MODEL_FILE):
    model = TAiModel1().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print(f"\n  ✅ {MODEL_NAME} caricato da {MODEL_FILE}")
else:
    print(f"\n  ⚠️  File modello non trovato: {MODEL_FILE}")
    print(f"  Addestra prima il modello con:  python t_ai_model.py")
    print(f"  Il server si avvierà comunque ma risponderà con un messaggio di errore.\n")


# ════════════════════════════════════════════════════════════════════
#  FLASK SERVER
# ════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # permette richieste dall'HTML aperto nel browser


@app.route("/", methods=["GET"])
def index():
    return jsonify({ "model": MODEL_NAME, "status": "online" if model else "no_model" })


@app.route("/chat", methods=["POST"])
def chat():
    if model is None:
        return jsonify({ "reply": f"⚠️ Modello non caricato. Esegui prima: python t_ai_model.py" }), 200

    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({ "reply": "Messaggio vuoto." }), 200

    # Costruisce prompt nel formato di training
    prompt  = f"Utente: {message}\nT-Ai:"
    tokens  = encode(prompt)
    idx     = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(idx)

    generated = decode(out[0].tolist()[len(tokens):])

    # Prende solo la prima riga prima del prossimo "Utente:"
    reply = generated.split("\n")[0].split("Utente:")[0].strip()
    if not reply:
        reply = "Sono T-Ai 1.0, sono qui per aiutarti!"

    return jsonify({ "reply": reply, "model": MODEL_NAME })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model":   MODEL_NAME,
        "loaded":  model is not None,
        "device":  device,
    })


# ════════════════════════════════════════════════════════════════════
#  AVVIO
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
  ╔══════════════════════════════════════════╗
  ║   {MODEL_NAME} — Server locale              ║
  ║   http://localhost:5000                  ║
  ║   Apri index.html nel browser            ║
  ╚══════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=5000, debug=False)