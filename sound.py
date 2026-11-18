"""Reproduce sonidos de efectos sin bloquear el vídeo (usa `afplay`, incluido en macOS)."""
import shutil
import subprocess
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


class EffectSound:
    def __init__(self, name: str):
        """`name` sin extensión, p. ej. "teleport" → busca assets/teleport.mp3/.wav/.m4a."""
        self.path = next((ASSETS / f"{name}{ext}" for ext in (".mp3", ".MP3", ".wav", ".m4a")
                          if (ASSETS / f"{name}{ext}").exists()), None)
        self.player = shutil.which("afplay")
        self._proc = None
        if self.path is None:
            print(f"[sonido] No hay {name}.mp3 en {ASSETS}. Sin sonido para ese efecto.")
        elif self.player is None:
            print("[sonido] No se encontró 'afplay'. Sin sonido.")
        else:
            print(f"[sonido] Usando {self.path.name}")

    def play(self):
        if self.path is None or self.player is None:
            return
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = subprocess.Popen([self.player, str(self.path)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
