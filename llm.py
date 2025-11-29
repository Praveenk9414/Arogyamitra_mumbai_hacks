# llm.py
"""
Robust local Ollama wrapper (CLI-based).
- Finds ollama executable.
- Attempts to run `ollama chat --format json` (streaming) and collects all chunks into one string.
- Falls back to `ollama chat --no-stream` or `ollama run` if needed.
- Returns a final concatenated string (never partial tokens).
- Provides diagnostics when something goes wrong.
"""

import os
import shutil
import subprocess
import time
import json
from typing import Optional, Tuple

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "koesn/llama3-8b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECS", "60"))
OLLAMA_CANDIDATE_PATHS = ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama")


# ---------------- helpers ----------------
def _find_ollama_executable() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    for p in OLLAMA_CANDIDATE_PATHS:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def _build_stream_command(ollama_exe: str, model: str) -> list:
    # prefer streaming JSON chat output
    return [ollama_exe, "chat", "-m", model, "--format", "json"]


def _build_no_stream_chat_command(ollama_exe: str, model: str) -> list:
    # non-streaming chat (returns whole answer)
    return [ollama_exe, "chat", "-m", model, "--no-stream"]


def _build_run_command(ollama_exe: str, model: str) -> list:
    return [ollama_exe, "run", model]


# ---------------- low-level call ----------------
def call_ollama_raw(prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[Optional[str], Optional[str], int]:
    """
    Run Ollama CLI and return (output_text_or_None, stderr_or_None, returncode).
    Strategy:
      1) Try streaming JSON chat and accumulate "response" or "message.content" fields.
      2) Try chat --no-stream and read stdout from communicate.
      3) Try ollama run <model> with prompt on stdin.
    """
    model = model or OLLAMA_MODEL
    timeout = timeout or OLLAMA_TIMEOUT

    ollama_exe = _find_ollama_executable()
    if not ollama_exe:
        return None, "ollama executable not found in PATH or known locations", -1

    strategies = [
        ("stream_json_chat", _build_stream_command(ollama_exe, model)),
        ("no_stream_chat", _build_no_stream_chat_command(ollama_exe, model)),
        ("run", _build_run_command(ollama_exe, model)),
    ]

    last_err = None

    for name, cmd in strategies:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except FileNotFoundError:
            return None, "ollama executable not found (FileNotFoundError)", -3
        except Exception as e:
            last_err = f"starting {name} failed: {e}"
            continue

        try:
            # prepare stdin payload for chat modes (JSON messages)
            if name in ("stream_json_chat", "no_stream_chat"):
                payload = {"messages": [{"role": "user", "content": prompt}]}
                stdin_text = json.dumps(payload) + "\n"
            else:
                # for run, pass prompt raw
                stdin_text = prompt + "\n"

            if name == "stream_json_chat":
                # streaming: read lines until done or timeout
                try:
                    proc.stdin.write(stdin_text)
                    proc.stdin.flush()
                except Exception:
                    pass

                collected_fragments = []
                stderr_collected = ""
                start = time.time()
                # read line-by-line
                while True:
                    # timeout guard
                    if timeout and (time.time() - start) > timeout:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        return None, "timeout", -2

                    line = proc.stdout.readline()
                    if line == "" and proc.poll() is not None:
                        # process finished
                        break
                    if not line:
                        time.sleep(0.01)
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    # try parse JSON chunk
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # non-json line — append raw
                        collected_fragments.append(line)
                        continue

                    # Ollama streaming may use "response" : "text" and "done": bool
                    if isinstance(obj, dict):
                        # Some versions provide `response` directly
                        if "response" in obj and isinstance(obj["response"], str):
                            collected_fragments.append(obj["response"])
                        # Some may include "message" -> {"content": "..."}
                        elif "message" in obj:
                            msg = obj["message"]
                            if isinstance(msg, dict):
                                # attempt to get content
                                if "content" in msg and isinstance(msg["content"], str):
                                    collected_fragments.append(msg["content"])
                                elif "content" in msg and isinstance(msg["content"], dict):
                                    # nested structure
                                    # try common keys
                                    for k in ("text", "parts", "content"):
                                        if k in msg["content"] and isinstance(msg["content"][k], str):
                                            collected_fragments.append(msg["content"][k])
                                            break
                        # Some streams include 'done' flag — if true, we can break
                        if obj.get("done") is True:
                            # read remaining stderr and finish
                            stderr_collected = proc.stderr.read() or ""
                            break
                    else:
                        collected_fragments.append(str(obj))
                # combine collected fragments
                stdout_combined = " ".join([frag for frag in collected_fragments if frag])
                stderr_text = stderr_collected or (proc.stderr.read() or "")
                rc = proc.poll() if proc.poll() is not None else 0
                if stdout_combined:
                    # collapse whitespace
                    return " ".join(stdout_combined.split()), (stderr_text.strip() if stderr_text else None), rc
                else:
                    last_err = stderr_text or "no output from streaming chat"
                    continue

            else:
                # non-streaming strategies: send stdin and wait for communicate
                try:
                    out, err = proc.communicate(stdin_text, timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return None, "timeout", -2

                rc = proc.returncode
                out = out.strip() if out else None
                err = err.strip() if err else None
                if out:
                    return out, err, rc
                else:
                    last_err = err or "no stdout"
                    continue

        except Exception as e:
            try:
                proc.kill()
            except Exception:
                pass
            last_err = f"{name} runtime error: {e}"
            continue

    diag = f"All ollama invocation strategies failed. Last error: {last_err}"
    return None, diag, -9


# ---------------- user-facing call ----------------
def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call the local LLM and return a single final string.
    If the LLM/CLI fails, return a diagnostic multi-line string.
    """
    stdout, stderr, rc = call_ollama_raw(prompt, model=model, timeout=OLLAMA_TIMEOUT)
    if stdout and rc == 0:
        # stdout is already combined/cleaned by call_ollama_raw
        # final cleanup (collapse extra whitespace)
        return " ".join(stdout.split()).strip()

    # failure: build diagnostic
    diag_lines = [
        "LLM (Ollama) unavailable or returned error.",
        f"Model: {model or OLLAMA_MODEL}",
        f"Return code: {rc}",
    ]
    if stderr:
        diag_lines.append("STDERR:")
        diag_lines.append(stderr[:2000])
    if stdout:
        diag_lines.append("STDOUT (partial):")
        diag_lines.append(stdout[:2000])
    diag_lines.append("Fallback: deterministic plan/summary used.")
    diag_lines.append("Truncated prompt (first 500 chars):")
    diag_lines.append((prompt[:500] + "..."))
    return "\n".join(diag_lines)


# ---------------- convenience alias ----------------
def safe_llm_summary(prompt: str, model: Optional[str] = None) -> str:
    """
    Stable alias used across the codebase.
    """
    return call_llm(prompt, model=model)


# If run as script, quick smoke test (does not execute by import)
if __name__ == "__main__":
    test_prompt = "Write a polite one-line dispute email for an overcharge on a hospital bill."
    print(safe_llm_summary(test_prompt)[:1000])
