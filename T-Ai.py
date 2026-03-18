#!/usr/bin/env python3
"""
╔══════════════════════════════════════╗
║            T-Ai v2.2                 ║
║   AI Assistant · Powered by Gemini   ║
╚══════════════════════════════════════╝
Requires: pip install google-generativeai rich pyperclip
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

# ── dependency check ──────────────────────────────────────────────────────────
MISSING = []
try:
    import google.generativeai as genai
except ImportError:
    MISSING.append("google-generativeai")
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
except ImportError:
    MISSING.append("rich")

# pyperclip opzionale — se manca usa fallback
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

if MISSING:
    print(f"[ERROR] Librerie mancanti: {', '.join(MISSING)}")
    print(f"Installa con:  pip install {' '.join(MISSING)}")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
VERSION      = "2.2.0"
MODEL_NAME   = "gemini-2.5-flash"
CONFIG_FILE  = Path.home() / ".t-ai_config.json"
GEMINI_CHAT  = "https://gemini.google.com/app"

ACCENT = "#00d4aa"
DIM    = "dim white"
ERR    = "bold red"
WARN   = "bold yellow"
OK     = "bold green"

# lingue supportate per syntax highlighting
CODE_LANGS = {
    "python", "python3", "py",
    "javascript", "js", "typescript", "ts",
    "html", "css", "scss",
    "cpp", "c++", "c", "java", "cs", "csharp",
    "bash", "sh", "shell", "powershell", "ps1",
    "json", "yaml", "yml", "toml", "xml",
    "sql", "rust", "go", "kotlin", "swift",
    "php", "ruby", "r", "matlab",
}

console = Console()

# ── config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    default = {
        "system_prompt": (
            "Sei T-Ai, un assistente AI da terminale simile a Claude Code. "
            "Sei preciso, sintetico e utile. Rispondi in italiano o nella lingua "
            "dell'utente. Per il codice usa SEMPRE i blocchi markdown con il nome "
            "del linguaggio (es: ```python, ```cpp, ```html). "
            "Quando lavori con file, sii esplicito sui path."
        ),
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            default.update(saved)
        except Exception:
            pass
    return default

def save_config(cfg: dict):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

# ── API key ───────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    console.print(f"\n[{WARN}]Nessuna API key trovata.[/]")
    console.print(f"[{DIM}]Ottienila gratis → [bold]https://aistudio.google.com/apikey[/][/]\n")
    key = Prompt.ask(f"[{ACCENT}]Incolla la tua GEMINI_API_KEY[/]").strip()
    if key:
        with open(".env", "a") as f:
            f.write(f"GEMINI_API_KEY={key}\n")
        console.print(f"[{OK}]✓ Chiave salvata in .env[/]\n")
    return key

# ── Gemini ────────────────────────────────────────────────────────────────────
def init_model(api_key: str, cfg: dict):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=cfg["system_prompt"],
    )

def chat_with_gemini(chat, user_msg: str) -> str:
    full_text = ""
    try:
        with Live(
            Panel(
                Spinner("dots2", text=f"[{ACCENT}] thinking...[/]"),
                title=f"[{ACCENT}]◆ T-Ai[/]  [dim]{MODEL_NAME}[/]",
                border_style=ACCENT,
            ),
            refresh_per_second=20,
            transient=True,
        ):
            response = chat.send_message(user_msg, stream=True)
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
    except Exception as e:
        full_text = f"[ERRORE] {e}"

    console.print()
    console.print(Panel(
        Markdown(full_text) if full_text.strip() else Text("(nessuna risposta)", style=DIM),
        title=f"[{ACCENT}]◆ T-Ai[/]  [dim]{MODEL_NAME}[/]",
        border_style=ACCENT,
        padding=(0, 2),
    ))

    return full_text

# ── code tools ────────────────────────────────────────────────────────────────
def extract_code_blocks(text: str) -> list:
    pattern = r"```(\w+)?\n(.*?)```"
    blocks  = re.findall(pattern, text, re.DOTALL)
    return [{"lang": (lang or "text").lower(), "code": code.strip()} for lang, code in blocks]

def show_code_blocks(blocks: list):
    """Mostra i blocchi di codice numerati con syntax highlighting."""
    if not blocks:
        return
    console.print()
    for i, b in enumerate(blocks, 1):
        lang  = b["lang"]
        label = lang.upper() if lang != "text" else "TESTO"
        hl    = lang if lang in CODE_LANGS else "text"
        console.print(Panel(
            Syntax(b["code"], hl, theme="monokai", line_numbers=True),
            title=f"[{ACCENT}]Blocco #{i}[/]  [dim]{label}[/]",
            border_style=ACCENT,
            padding=(0, 1),
        ))

def copy_to_clipboard(text: str) -> bool:
    """Copia testo negli appunti. Ritorna True se riuscito."""
    if HAS_CLIPBOARD:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            pass
    # fallback Windows
    try:
        proc = subprocess.run(
            ["clip"], input=text.encode("utf-8"),
            capture_output=True
        )
        return proc.returncode == 0
    except Exception:
        pass
    # fallback macOS
    try:
        proc = subprocess.run(
            ["pbcopy"], input=text.encode("utf-8"),
            capture_output=True
        )
        return proc.returncode == 0
    except Exception:
        pass
    return False

def show_post_message_info(blocks: list):
    """Mostra hint copia e link Gemini dopo ogni messaggio."""
    hints = []

    if blocks:
        langs = list({b["lang"] for b in blocks if b["lang"] not in ("text", "")})
        langs_str = ", ".join(l.upper() for l in langs) if langs else "codice"
        n = len(blocks)
        hints.append(
            f"[{ACCENT}]📋 {n} blocco/i {langs_str}[/] "
            f"[{DIM}]→ usa [bold]/copy {1}[/] per copiare  "
            f"(es: /copy 1, /copy 2…)[/]"
        )

    hints.append(
        f"[{DIM}]💬 Continua su Gemini → [bold {ACCENT}]{GEMINI_CHAT}[/][/]"
    )

    for h in hints:
        console.print(f"  {h}")

def run_code(code: str, lang: str):
    runners = {
        "python":     ["python", "-c"],
        "python3":    ["python", "-c"],
        "py":         ["python", "-c"],
        "bash":       ["bash",   "-c"],
        "sh":         ["bash",   "-c"],
        "js":         ["node",   "-e"],
        "javascript": ["node",   "-e"],
    }
    lang = lang.lower()
    if lang not in runners:
        return "", f"Esecuzione non supportata per '{lang}'. Supportati: python, bash, js.", 1
    try:
        r = subprocess.run(runners[lang] + [code], capture_output=True, text=True, timeout=30)
        return r.stdout, r.stderr, r.returncode
    except FileNotFoundError:
        return "", f"Interprete non trovato.", 1
    except subprocess.TimeoutExpired:
        return "", "Timeout (30s).", 1
    except Exception as e:
        return "", str(e), 1

# ── UI ────────────────────────────────────────────────────────────────────────
def banner():
    art = Text()
    art.append("  ████████╗      █████╗ ██╗\n", style=f"bold {ACCENT}")
    art.append("     ██╔══╝     ██╔══██╗██║\n", style=f"bold {ACCENT}")
    art.append("     ██║  ─────  ███████║██║\n", style=f"bold {ACCENT}")
    art.append("     ██║        ██╔══██║██║\n", style=f"bold {ACCENT}")
    art.append("     ██║        ██║  ██║██║\n", style=f"bold {ACCENT}")
    art.append("     ╚═╝        ╚═╝  ╚═╝╚═╝\n", style=f"bold {ACCENT}")
    console.print()
    console.print(Panel(
        art,
        subtitle=f"[dim]Gemini · {MODEL_NAME} · v{VERSION}[/]",
        border_style=ACCENT,
        padding=(0, 4),
    ))

def help_panel():
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {ACCENT}")
    t.add_column("Comando",      style="bold white", no_wrap=True)
    t.add_column("Descrizione",  style=DIM)
    for cmd, desc in [
        ("/help",           "Mostra questo menu"),
        ("/helpmodel",      "Tutorial: come trovare il modello giusto"),
        ("/new  /clear",    "Nuova conversazione"),
        ("/copy <n>",       "Copia negli appunti il blocco di codice #n"),
        ("/run",            "Esegui l'ultimo blocco di codice (python/bash/js)"),
        ("/file <path>",    "Carica un file come contesto"),
        ("/save <path>",    "Salva l'ultimo codice in un file"),
        ("/system <testo>", "Cambia il system prompt"),
        ("/history",        "Mostra la cronologia della chat"),
        ("/export",         "Esporta la chat in un file Markdown"),
        ("/quit  /exit",    "Esci da T-Ai"),
    ]:
        t.add_row(cmd, desc)
    console.print(Panel(t, title="[bold]Comandi T-Ai[/]", border_style=ACCENT))

def helpmodel_panel():
    content = Text()
    content.append("\nCome trovare il nome del modello giusto per T-Ai\n\n", style=f"bold {ACCENT}")

    content.append("STEP 1 — Ottieni la tua API key\n", style="bold white")
    content.append("  Vai su: ", style=DIM)
    content.append("https://aistudio.google.com/apikey\n", style=f"bold {ACCENT}")
    content.append("  Crea o copia la tua chiave API gratuita.\n\n", style=DIM)

    content.append("STEP 2 — Scopri i modelli disponibili\n", style="bold white")
    content.append("  Crea un file ", style=DIM)
    content.append("lista_modelli.py", style=f"bold {ACCENT}")
    content.append(" con questo contenuto:\n\n", style=DIM)

    code = (
        "import google.generativeai as genai\n"
        'genai.configure(api_key="LA_TUA_API_KEY")\n'
        "for m in genai.list_models():\n"
        '    if "generateContent" in m.supported_generation_methods:\n'
        "        print(m.name)"
    )
    content.append(f"  {code}\n\n", style="bold green")

    content.append("  Avvialo con:  ", style=DIM)
    content.append("python lista_modelli.py\n\n", style=f"bold {ACCENT}")

    content.append("STEP 3 — Scegli il modello\n", style="bold white")
    content.append("  Consigliati:\n", style=DIM)

    models_info = [
        ("gemini-2.5-flash",     "⚡ Veloce, intelligente, gratuito  ← CONSIGLIATO"),
        ("gemini-2.5-pro",       "🧠 Più potente, per task complessi"),
        ("gemini-2.0-flash",     "🔄 Stabile, compatibile"),
        ("gemini-2.0-flash-lite","💨 Leggero, per risposte rapide"),
    ]
    for name, desc in models_info:
        content.append(f"  • ", style=DIM)
        content.append(f"{name:<35}", style=f"bold {ACCENT}")
        content.append(f"{desc}\n", style=DIM)

    content.append("\nSTEP 4 — Imposta il modello in T-Ai\n", style="bold white")
    content.append("  Apri T-Ai.py con un editor di testo e cerca la riga:\n\n", style=DIM)
    content.append('  MODEL_NAME   = "gemini-2.5-flash"\n\n', style="bold green")
    content.append("  Sostituisci il nome con quello che vuoi usare.\n", style=DIM)

    content.append("\n⚠  ERRORE 404?\n", style=WARN)
    content.append(
        "  Significa che il modello non esiste o non è disponibile\n"
        "  sulla tua key. Ri-esegui lista_modelli.py e usa\n"
        "  esattamente uno dei nomi che appaiono nella lista.\n",
        style=DIM
    )

    console.print(Panel(content, title="[bold]📖 Tutorial Modelli[/]", border_style=ACCENT, padding=(0, 2)))

def print_user_msg(msg: str):
    console.print()
    console.print(Panel(
        Text(msg, style="white"),
        title="[bold white]◉ Tu[/]",
        border_style="white",
        padding=(0, 2),
    ))

# ── command handler ───────────────────────────────────────────────────────────
def handle_command(cmd_line: str, state: dict) -> bool:
    parts = cmd_line.strip().split(None, 1)
    cmd   = parts[0].lower()
    arg   = parts[1] if len(parts) > 1 else ""
    cfg   = state["cfg"]

    if cmd in ("/quit", "/exit", "/q"):
        console.print(f"\n[{ACCENT}]Arrivederci! 👋[/]\n")
        return False

    elif cmd == "/help":
        help_panel()

    elif cmd == "/helpmodel":
        helpmodel_panel()

    elif cmd in ("/new", "/clear"):
        state["chat"]      = state["model"].start_chat(history=[])
        state["history"]   = []
        state["last_code"] = None
        state["last_blocks"] = []
        console.clear()
        banner()
        console.print(f"[{OK}]✓ Nuova conversazione avviata.[/]")

    elif cmd == "/copy":
        blocks = state.get("last_blocks", [])
        if not blocks:
            console.print(f"[{WARN}]Nessun blocco di codice disponibile. Chiedi prima del codice a T-Ai.[/]")
        else:
            # se non specificato prende il primo
            idx = 1
            if arg:
                try:
                    idx = int(arg)
                except ValueError:
                    console.print(f"[{WARN}]Uso: /copy <numero>  (es: /copy 1)[/]")
                    return True
            if idx < 1 or idx > len(blocks):
                console.print(f"[{WARN}]Blocco #{idx} non esiste. Disponibili: 1–{len(blocks)}[/]")
            else:
                code = blocks[idx - 1]["code"]
                lang = blocks[idx - 1]["lang"].upper()
                if copy_to_clipboard(code):
                    console.print(f"[{OK}]✓ Blocco #{idx} ({lang}) copiato negli appunti! 📋[/]")
                else:
                    console.print(f"[{WARN}]Clipboard non disponibile. Installa: pip install pyperclip[/]")
                    console.print(f"[{DIM}]Codice del blocco #{idx}:[/]")
                    console.print(Syntax(code, blocks[idx-1]["lang"], theme="monokai"))

    elif cmd == "/run":
        blocks = state.get("last_blocks", [])
        if not blocks:
            console.print(f"[{WARN}]Nessun codice da eseguire.[/]")
        else:
            # esegui il primo blocco eseguibile
            lc = state.get("last_code")
            if not lc:
                lc = blocks[0]
            console.print(Panel(
                Syntax(lc["code"], lc["lang"], theme="monokai", line_numbers=True),
                title=f"[bold]Eseguo[/] [{ACCENT}]{lc['lang'].upper()}[/]",
                border_style=ACCENT,
            ))
            stdout, stderr, rc = run_code(lc["code"], lc["lang"])
            out = stdout or stderr or "(nessun output)"
            console.print(Panel(
                Text(out, style="white"),
                title=f"[{'bold green' if rc == 0 else ERR}]Output (exit {rc})[/]",
                border_style="green" if rc == 0 else "red",
            ))

    elif cmd == "/file":
        if not arg:
            console.print(f"[{WARN}]Uso: /file <percorso>[/]")
        else:
            try:
                content = Path(arg).expanduser().read_text(errors="replace")
                preview = content[:300] + ("…" if len(content) > 300 else "")
                console.print(Panel(Text(preview, style=DIM), title=f"Caricato: {arg}", border_style=ACCENT))
                inject = f"[Contenuto del file `{arg}`]\n```\n{content}\n```"
                state["chat"].send_message(inject)
                state["history"].append({"role": "user", "content": inject})
                console.print(f"[{OK}]✓ File aggiunto al contesto.[/]")
            except Exception as e:
                console.print(f"[{ERR}]Errore lettura file: {e}[/]")

    elif cmd == "/save":
        if not arg:
            console.print(f"[{WARN}]Uso: /save <percorso>[/]")
        elif not state.get("last_code"):
            console.print(f"[{WARN}]Nessun codice da salvare.[/]")
        else:
            try:
                p = Path(arg).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(state["last_code"]["code"])
                console.print(f"[{OK}]✓ Salvato: {p}[/]")
            except Exception as e:
                console.print(f"[{ERR}]Errore salvataggio: {e}[/]")

    elif cmd == "/system":
        if not arg:
            console.print(Panel(Text(cfg["system_prompt"], style=DIM), title="System Prompt attuale", border_style=ACCENT))
        else:
            cfg["system_prompt"] = arg
            save_config(cfg)
            state["model"] = init_model(state["api_key"], cfg)
            state["chat"]  = state["model"].start_chat(history=[])
            state["history"] = []
            console.print(f"[{OK}]✓ System prompt aggiornato. Conversazione resettata.[/]")

    elif cmd == "/history":
        if not state["history"]:
            console.print(f"[{DIM}]Nessuna conversazione in corso.[/]")
        else:
            for m in state["history"]:
                label = "Tu" if m["role"] == "user" else "T-Ai"
                style = "bold white" if m["role"] == "user" else f"bold {ACCENT}"
                preview = m["content"][:100].replace("\n", " ")
                if len(m["content"]) > 100:
                    preview += "…"
                console.print(f"[{style}]{label}:[/] [{DIM}]{preview}[/]")

    elif cmd == "/export":
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"t-ai_chat_{ts}.md")
        lines = [f"# T-Ai Chat — {datetime.now():%d/%m/%Y %H:%M}\n\n"]
        for m in state["history"]:
            role = "**Tu**" if m["role"] == "user" else "**T-Ai**"
            lines.append(f"### {role}\n{m['content']}\n\n---\n\n")
        path.write_text("".join(lines))
        console.print(f"[{OK}]✓ Esportato: [bold]{path}[/][/]")

    else:
        console.print(f"[{WARN}]Comando sconosciuto: {cmd}  ·  usa /help[/]")

    return True

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    cfg     = load_config()
    api_key = get_api_key()

    if not api_key:
        console.print(f"[{ERR}]API key mancante. Impossibile avviare T-Ai.[/]")
        sys.exit(1)

    try:
        model = init_model(api_key, cfg)
        chat  = model.start_chat(history=[])
    except Exception as e:
        console.print(f"[{ERR}]Errore inizializzazione Gemini: {e}[/]")
        sys.exit(1)

    banner()

    clip_str = f"[{OK}]clipboard ✓[/]" if HAS_CLIPBOARD else f"[{DIM}]clipboard (pip install pyperclip)[/]"
    console.print(f"  [{DIM}]modello: [bold]{MODEL_NAME}[/]  ·  {clip_str}  ·  /help per i comandi[/]")
    console.print(Rule(style=DIM))

    state = {
        "cfg":         cfg,
        "api_key":     api_key,
        "model":       model,
        "chat":        chat,
        "history":     [],
        "last_code":   None,
        "last_blocks": [],
    }

    while True:
        try:
            console.print()
            user_input = Prompt.ask(f"[bold {ACCENT}]❯[/]", console=console).strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_command(user_input, state):
                    break
                continue

            print_user_msg(user_input)
            response = chat_with_gemini(state["chat"], user_input)

            state["history"].append({"role": "user",      "content": user_input})
            state["history"].append({"role": "assistant", "content": response})

            # estrai e mostra blocchi codice
            blocks = extract_code_blocks(response)
            state["last_blocks"] = blocks
            if blocks:
                # trova il primo blocco eseguibile
                exec_langs = {"python", "python3", "py", "bash", "sh", "js", "javascript"}
                for b in blocks:
                    if b["lang"] in exec_langs:
                        state["last_code"] = b
                        break
                else:
                    state["last_code"] = blocks[0]

                # mostra i blocchi con syntax highlighting
                show_code_blocks(blocks)

            # sempre: mostra hint copia + link Gemini
            show_post_message_info(blocks)

        except KeyboardInterrupt:
            console.print(f"\n[{DIM}](Ctrl+C — usa /quit per uscire)[/]")
        except EOFError:
            console.print(f"\n[{ACCENT}]Arrivederci! 👋[/]\n")
            break

if __name__ == "__main__":
    main()