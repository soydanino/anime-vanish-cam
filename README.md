# Teletransportación por gesto para videollamadas

Los dos gestos están activos a la vez (o elige uno con `1` / `2`, ambos con `3`, o `--modo goku|naruto|ambos`):
- **Goku**: dos dedos en la frente ~1 s → teletransporte con líneas tipo cómic. Repite el gesto para volver.
- **Naruto**: junta las palmas frente a la cámara (dedos lado a lado, no entrelazados) → humo y un tronco
  ocupa tu lugar. Separa las manos y vuelve a juntarlas para volver (o pulsa `R`).

Pon dos dedos (índice y medio, rectos y juntos) en la frente, mantén ~1 s y desapareces de la webcam con un efecto
de teletransportación. Repite el gesto para volver. La imagen se envía a una cámara virtual que
puedes elegir en WhatsApp, Zoom, Meet, Teams, etc.

## Instalación (macOS)
```bash
brew install python@3.11 && brew install --cask obs
git clone https://github.com/soydanino/anime-vanish-cam.git ~/teletransportacion-goku
cd ~/teletransportacion-goku
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Configuración (una sola vez)
1. Abre **OBS** → botón **Iniciar cámara virtual** → aprueba la extensión del sistema en
   *Ajustes del Sistema → General → Ítems de inicio y extensiones* (o el aviso que aparezca).
   Luego **Detener cámara virtual** y cierra OBS (no hace falta tenerlo abierto después).
2. La primera vez que ejecutes el programa, macOS pedirá permiso de cámara para la Terminal: acéptalo.

## Uso (antes de entrar a la reunión)
```bash
cd ~/teletransportacion-goku
.venv/bin/python main.py            # cámara 0
.venv/bin/python main.py --camera 1 # otra cámara
```
1. Al arrancar, **sal del encuadre 4 s**; si todavía se te ve, la captura se repite sola para que capture el fondo vacío.
2. Vuelve a sentarte. En la reunión elige la cámara **"OBS Virtual Camera"**.
3. Gesto = desaparecer; gesto otra vez = volver.

Teclas en la vista previa: `1`/`2`/`3` Goku/Naruto/ambos · `R` volver · `B` recapturar fondo · `T` teletransportar sin gesto · `D` diagnóstico del gesto en la terminal · `S` guardar imágenes de diagnóstico · `Q` salir.

Opciones: `--hold 1.0` (segundos del gesto), `--no-virtual` (solo vista previa), `--width/--height/--fps`.

## Consejos
- Buena luz de frente mejora la detección de la mano y el recorte.
- Si mueves la cámara o cambia mucho la luz, pulsa `B` para recapturar el fondo.
- Si el gesto no se detecta, pulsa `D` y mira en la terminal cuántas caras/manos ve y si la pose es correcta.

## Sonido
Los sonidos no se incluyen en el repositorio porque tienen derechos de autor. Si existen
`assets/teleport.mp3` (Goku) y `assets/vanish.mp3` (Naruto), se reproducen por los altavoces; si no, los
efectos funcionan sin sonido. Puedes usar los del repositorio de referencia (ver Créditos) o los tuyos.

El vídeo de la cámara virtual no lleva audio: en la reunión solo se oirá si tu micrófono lo capta.

## Créditos
Los gestos, los efectos (`goku_effect.py`, `naruto_effect.py`, `naruto_detector.py`), el tronco y los sonidos vienen de https://github.com/Nicovic17/vanish_webcam.
