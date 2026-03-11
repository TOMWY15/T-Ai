"""
T-Ai Model 1.0
Transformer Language Model costruito da zero in Python.

INSTALLAZIONE (esegui nel terminale/cmd):
    pip install torch

UTILIZZO:
    python t_ai_model.py          -> addestra e chatta
    python t_ai_model.py --chat   -> solo chat (gia addestrato)
"""

import os
import sys
import json
import math
import argparse

# ── controlla torch ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("\n  ERRORE: PyTorch non installato.")
    print("  Esegui nel terminale:  pip install torch\n")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

MODEL_NAME    = "T-Ai 1.0"
MODEL_FILE    = "t_ai_1_0.pt"

VOCAB_SIZE    = 256
CONTEXT_LEN   = 64
EMBED_DIM     = 128
NUM_HEADS     = 4
NUM_LAYERS    = 3
FF_DIM        = 512
DROPOUT       = 0.1

BATCH_SIZE    = 16
LEARNING_RATE = 3e-4
EPOCHS        = 600
LOG_EVERY     = 60

MAX_TOKENS    = 120
TEMPERATURE   = 0.85
TOP_K         = 30

# ════════════════════════════════════════════════════════════════════
#  TOKENIZER
# ════════════════════════════════════════════════════════════════════

def encode(text):
    return [min(ord(c), 255) for c in text]

def decode(tokens):
    return "".join(chr(t) for t in tokens)

# ════════════════════════════════════════════════════════════════════
#  ARCHITETTURA
# ════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads  = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv      = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj     = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.drop     = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(EMBED_DIM, dim=2)
        def sh(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
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
            nn.Linear(EMBED_DIM, FF_DIM),
            nn.GELU(),
            nn.Linear(FF_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TAiModel1(nn.Module):
    """T-Ai Model 1.0 — Transformer decoder-only"""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.ln      = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

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

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new=MAX_TOKENS, temp=TEMPERATURE, top_k=TOP_K):
        for _ in range(max_new):
            ctx     = idx[:, -CONTEXT_LEN:]
            logits, _ = self(ctx)
            logits  = logits[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs   = F.softmax(logits, dim=-1)
            nxt     = torch.multinomial(probs, 1)
            idx     = torch.cat([idx, nxt], dim=1)
        return idx

# ════════════════════════════════════════════════════════════════════
#  DATI DI TRAINING
# ════════════════════════════════════════════════════════════════════

TRAIN_TEXT = """\
Utente: Ciao!
T-Ai: Ciao! Sono T-Ai 1.0, il tuo assistente intelligente. Come posso aiutarti?

Utente: Chi sei?
T-Ai: Sono T-Ai 1.0, un modello di linguaggio costruito da zero con architettura Transformer.

Utente: Come ti chiami?
T-Ai: Mi chiamo T-Ai 1.0. Sono un assistente AI creato per essere utile e preciso.

Utente: Cosa sai fare?
T-Ai: Posso rispondere a domande, aiutarti a scrivere, spiegare concetti e conversare in italiano.

Utente: Come funzioni?
T-Ai: Sono un Transformer: leggo il testo carattere per carattere e imparo a prevedere il testo successivo.

Utente: Sei intelligente?
T-Ai: Sono T-Ai 1.0, un modello AI che impara dai dati. Faccio del mio meglio per essere utile!

Utente: Qual e la capitale d'Italia?
T-Ai: La capitale d'Italia e Roma, una delle citta piu belle e storiche del mondo.

Utente: Quanto fa 2 piu 2?
T-Ai: 2 piu 2 fa 4!

Utente: Quanto fa 3 per 3?
T-Ai: 3 per 3 fa 9!

Utente: Cos'e Python?
T-Ai: Python e un linguaggio di programmazione semplice e potente, molto usato per l'intelligenza artificiale.

Utente: Cos'e il machine learning?
T-Ai: Il machine learning e una tecnica con cui i computer imparano dai dati senza essere programmati esplicitamente.

Utente: Che tempo fa?
T-Ai: Non ho accesso a dati meteo in tempo reale, ma posso aiutarti con molte altre domande!

Utente: Raccontami una barzelletta.
T-Ai: Perche il computer e andato dal medico? Perche aveva troppi byte di raffreddore!

Utente: Dimmi qualcosa di interessante.
T-Ai: Lo sapevi che il cervello umano ha circa 86 miliardi di neuroni? E la struttura piu complessa che conosciamo!

Utente: Grazie!
T-Ai: Prego! Sono sempre qui se hai bisogno di aiuto.

Utente: Arrivederci.
T-Ai: Arrivederci! E stato un piacere parlare con te. A presto!

Utente: Ciao come stai?
T-Ai: Sto benissimo grazie! Sono pronto ad aiutarti. Tu come stai?

Utente: Bene grazie.
T-Ai: Sono contento! Posso aiutarti con qualcosa?

Utente: Sei meglio di ChatGPT?
T-Ai: Sono T-Ai 1.0, un modello diverso e piu piccolo. Ma sono qui per aiutarti al meglio!

Utente: Hai emozioni?
T-Ai: Non ho emozioni reali, ma sono progettato per essere utile ed empatico nelle conversazioni.

Utente: Cosa e un Transformer?
T-Ai: Il Transformer e un'architettura di rete neurale basata sull'attenzione, usata nei modelli linguistici moderni.

Utente: Come si fa un ciclo in Python?
T-Ai: In Python scrivi: for i in range(10): print(i) — questo stampa i numeri da 0 a 9.

Utente: Ciao T-Ai!
T-Ai: Ciao! Sono T-Ai 1.0. Come posso esserti utile oggi?

""" * 6   # ripete piu volte per avere piu dati


# ════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════

def get_batch(data, device):
    n   = len(data) - CONTEXT_LEN - 1
    if n <= 0:
        raise ValueError("Testo di training troppo corto!")
    idx = torch.randint(0, n, (BATCH_SIZE,))
    x   = torch.stack([data[i : i + CONTEXT_LEN] for i in idx])
    y   = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in idx])
    return x.to(device), y.to(device)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device.upper()}")

    # Prepara dati
    tokens = encode(TRAIN_TEXT)
    data   = torch.tensor(tokens, dtype=torch.long)
    print(f"  Token totali: {len(data):,}")

    # Modello
    model = TAiModel1().to(device)
    print(f"  Parametri: {model.num_params():,}")
    print(f"  Inizio training per {EPOCHS} step...\n")

    opt  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best = float("inf")

    for step in range(1, EPOCHS + 1):
        model.train()
        x, y   = get_batch(data, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            l   = loss.item()
            pct = step / EPOCHS
            bar = ("█" * int(pct * 25)).ljust(25, "░")
            print(f"  [{bar}] step {step:>4}/{EPOCHS}  loss: {l:.4f}")
            if l < best:
                best = l
                torch.save(model.state_dict(), MODEL_FILE)

    print(f"\n  Modello salvato in: {MODEL_FILE}")
    print(f"  Loss migliore: {best:.4f}\n")
    return model


# ════════════════════════════════════════════════════════════════════
#  CHAT
# ════════════════════════════════════════════════════════════════════

def chat(model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica modello se non passato
    if model is None:
        if not os.path.exists(MODEL_FILE):
            print(f"\n  ERRORE: file '{MODEL_FILE}' non trovato.")
            print("  Esegui prima senza --chat per addestrare il modello.\n")
            sys.exit(1)
        model = TAiModel1().to(device)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print(f"\n  Modello {MODEL_NAME} caricato.")

    model.eval()

    print(f"""
  ╔══════════════════════════════════════╗
  ║   {MODEL_NAME} — Chat avviata         ║
  ║   Scrivi 'esci' per uscire           ║
  ╚══════════════════════════════════════╝
""")

    while True:
        try:
            user = input("  Tu  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  T-Ai: Arrivederci!\n")
            break

        if not user:
            continue
        if user.lower() in ("esci", "exit", "quit"):
            print("  T-Ai: Arrivederci!\n")
            break

        prompt  = f"Utente: {user}\nT-Ai:"
        tokens  = encode(prompt)
        idx     = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model.generate(idx)

        generated = decode(out[0].tolist()[len(tokens):])

        # Prende solo la prima riga della risposta
        reply = generated.split("\n")[0].split("Utente:")[0].strip()
        if not reply:
            reply = "Sono T-Ai 1.0, sono qui per aiutarti!"

        print(f"  T-Ai> {reply}\n")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
  ╔══════════════════════════════════════════╗
  ║         {MODEL_NAME} — Language Model       ║
  ║   Transformer costruito da zero          ║
  ╚══════════════════════════════════════════╝
""")

    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="Solo chat (salta training)")
    args = parser.parse_args()

    if args.chat:
        chat()
    else:
        # Training poi chat automaticamente
        model = train()
        chat(model)