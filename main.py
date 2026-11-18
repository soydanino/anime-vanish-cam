"""Desaparecer de la webcam con un gesto, enviado a una cámara virtual para reuniones.

Basado en los modos Goku y Naruto de https://github.com/Nicovic17/vanish_webcam.

Gestos (por defecto los dos están activos a la vez):
  Goku    dos dedos (índice y medio) en la frente ~1 s → teletransporte. Repite el gesto para volver.
  Naruto  junta las palmas frente a la cámara → humo y tronco. Sepáralas y vuelve a juntarlas para volver.

Teclas en la ventana de vista previa:
  1 / 2 / 3  solo Goku / solo Naruto / ambos (estando visible)
  B  capturar el fondo de nuevo (sal del encuadre durante la cuenta atrás)
  T  desaparecer / volver sin gesto (usa Goku si está activo)
  R  volver
  D  mostrar/ocultar diagnóstico del gesto en la terminal
  S  guardar imágenes de diagnóstico en debug/
  Q  salir (también Esc)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from background import MAX_PERSON_FRACTION, BackgroundCapture, PersonSegmenter, person_fraction
from goku_effect import TELEPORT_SECONDS, GokuGestureGate, GokuPoseRecognizer, GokuTeleport
from naruto_detector import MultiViewHandLandmarker
from naruto_effect import EFFECT_SECONDS as NARUTO_SECONDS
from naruto_effect import load_log, render_substitution, substitution_anchor
from sound import EffectSound

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
INFER_WIDTH = 640
GESTURE_INTERVAL = {"goku": 0.10, "naruto": 0.08}
MODE_NAMES = {"goku": "Goku", "naruto": "Naruto", "ambos": "Goku + Naruto"}
NARUTO_RELEASE_SECONDS = 0.5   # tras desaparecer, separar las manos este tiempo antes de poder volver
WINDOW = "Teletransportacion (vista previa)"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="índice de la webcam (0, 1, ...)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--modo", choices=("ambos", "goku", "naruto"), default="ambos",
                    help="gestos activos al iniciar")
    ap.add_argument("--hold", type=float, default=1.0, help="segundos que hay que mantener el gesto (Goku)")
    ap.add_argument("--no-virtual", action="store_true", help="solo vista previa, sin cámara virtual")
    return ap.parse_args()


def create_recognizer():
    hand_model = str(MODELS_DIR / "hand_landmarker.task")
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )
    fallback_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
    face_options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / "blaze_face_short_range.tflite")),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    return GokuPoseRecognizer(
        vision.HandLandmarker.create_from_options(hand_options),
        vision.FaceDetector.create_from_options(face_options),
        vision.HandLandmarker.create_from_options(fallback_options),
    )


def create_naruto_recognizer():
    hand_model = str(MODELS_DIR / "hand_landmarker.task")
    common = dict(num_hands=2, min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35)
    primary = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO, min_tracking_confidence=0.45, **common))
    alternate = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.IMAGE, **common))
    return MultiViewHandLandmarker(primary, alternate)


def open_virtual_camera(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"[cam virtual] Enviando a: {cam.device}")
        return cam
    except Exception as e:  # noqa: BLE001
        print("[cam virtual] No disponible:", e)
        print("  → Abre OBS una vez, pulsa 'Iniciar cámara virtual', aprueba la extensión del sistema,")
        print("    luego detén la cámara virtual y cierra OBS. Continúo solo con la vista previa.")
        return None


def draw_hud(img, lines, progress):
    y = 30
    for line in lines:
        cv2.putText(img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    if progress > 0:
        cv2.rectangle(img, (15, y), (15 + int((img.shape[1] // 3) * progress), y + 10), (255, 220, 120), -1)


def to_mp(frame, size):
    rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.camera}. Revisa el permiso de cámara de la Terminal.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("La cámara no entrega imágenes.")
    h, w = frame.shape[:2]
    print(f"[webcam] {w}x{h}")

    infer_size = (INFER_WIDTH, int(round(INFER_WIDTH * h / w)))
    recognizer = create_recognizer()
    naruto_recognizer = create_naruto_recognizer()
    log_sprite = load_log()
    segmenter = PersonSegmenter(MODELS_DIR)
    gate = GokuGestureGate(hold_seconds=args.hold)
    capture = BackgroundCapture()
    sounds = {"goku": EffectSound("teleport"), "naruto": EffectSound("vanish")}
    vcam = None if args.no_virtual else open_virtual_camera(w, h, args.fps)

    mode = args.modo                          # gestos activos: "ambos", "goku" o "naruto"
    active = None                             # efecto en curso o con el que desapareciste
    background = None
    vanished_frame = None                     # imagen fija mientras no estás (fondo, o fondo + tronco)
    naruto_mask, naruto_anchor, log_cache = None, None, {}
    naruto_release_start, naruto_released = None, True
    phase = "calibrating"  # calibrating → ready ⇄ (vanishing → vanished → appearing)
    effect, effect_start = None, 0.0
    last_check = {"goku": 0.0, "naruto": 0.0}
    last_diag = 0.0
    diag = False
    t0 = time.monotonic()
    last_ts = -1

    def enabled(kind):
        return mode in (kind, "ambos")

    def next_ts(now):
        nonlocal last_ts
        last_ts = max(int((now - t0) * 1000), last_ts + 1)
        return last_ts

    def start_teleport(frame, now):
        nonlocal phase, effect, effect_start, vanished_frame, active
        appearing = phase == "vanished"
        mask = segmenter.mask(to_mp(frame, infer_size), next_ts(now), w, h)
        effect = GokuTeleport(frame, background, mask)
        effect_start = now
        phase = "appearing" if appearing else "vanishing"
        active = "goku"
        vanished_frame = None
        sounds["goku"].play()
        print("[goku]", "volviendo" if appearing else "saliendo", flush=True)

    def start_naruto(now):
        nonlocal phase, effect_start, naruto_anchor, naruto_mask, vanished_frame, active
        nonlocal naruto_release_start, naruto_released
        phase = "vanishing"
        active = "naruto"
        effect_start = now
        naruto_anchor, naruto_mask, vanished_frame = None, None, None
        naruto_release_start, naruto_released = None, False
        sounds["naruto"].play()
        print("[naruto] sustitución", flush=True)

    def come_back(frame, now):
        nonlocal phase, vanished_frame, naruto_release_start, naruto_released
        if active == "goku":
            start_teleport(frame, now)
            return
        phase = "ready"
        vanished_frame = None
        gate.reset()
        naruto_release_start, naruto_released = None, False
        sounds["naruto"].play()
        print("[naruto] de vuelta", flush=True)

    def gesture_hint():
        parts = []
        if enabled("goku"):
            parts.append("dos dedos en la frente (Goku)")
        if enabled("naruto"):
            parts.append("junta las palmas (Naruto)")
        return " o ".join(parts)

    capture.start(time.monotonic())
    print("[fondo] Sal del encuadre: capturando el fondo en 4 s...", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[webcam] Se perdió la imagen.")
                break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            now = time.monotonic()

            # --- captura y validación del fondo ---
            if capture.capturing:
                candidate = capture.feed(frame, now)
                if candidate is not None:
                    frac = person_fraction(segmenter.mask(to_mp(candidate, infer_size), next_ts(now), w, h))
                    if frac > MAX_PERSON_FRACTION:
                        print(f"[fondo] Todavía se te ve en el fondo ({frac:.0%}). Repito: sal del encuadre.",
                              flush=True)
                        capture.start(now)
                    else:
                        background = candidate
                        phase = "ready"
                        gate.reset()
                        print(f"[fondo] Listo. Para desaparecer: {gesture_hint()}.", flush=True)

            # --- gestos ---
            if not capture.capturing and background is not None and phase in ("ready", "vanished"):
                # Qué detector hace falta: estando visible, los activos; estando desaparecido,
                # solo el del efecto con el que te fuiste (para volver).
                want = [k for k in ("goku", "naruto") if enabled(k)] if phase == "ready" else [active]
                img = None
                for kind in want:
                    if now - last_check[kind] < GESTURE_INTERVAL[kind] or phase not in ("ready", "vanished"):
                        continue
                    last_check[kind] = now
                    img = img if img is not None else to_mp(frame, infer_size)
                    if kind == "goku":
                        pose = recognizer.recognize_pose(img, next_ts(now))
                        if diag and now - last_diag >= 1.0:
                            last_diag = now
                            held = 0.0 if gate.hold_started is None else now - gate.hold_started
                            print(f"[diag goku] caras={recognizer.last_face_count} "
                                  f"manos={recognizer.last_hand_count} pose={pose} "
                                  f"mantenido={held:.2f}/{gate.hold_seconds:g}s", flush=True)
                        if gate.update(pose, now):
                            start_teleport(frame, now)
                    else:
                        joined = naruto_recognizer.recognize_joined_hands(img, next_ts(now))
                        if diag and now - last_diag >= 1.0:
                            last_diag = now
                            print(f"[diag naruto] manos={naruto_recognizer.last_hand_count} juntas={joined}",
                                  flush=True)
                        # Tras cada efecto hay que separar las manos un momento antes de que
                        # vuelvan a contar (si no, al volver con las manos juntas desaparecerías otra vez).
                        if not joined:
                            naruto_release_start = naruto_release_start or now
                            if now - naruto_release_start >= NARUTO_RELEASE_SECONDS:
                                naruto_released = True
                        else:
                            naruto_release_start = None
                            if naruto_released:
                                if phase == "ready":
                                    start_naruto(now)
                                else:
                                    come_back(frame, now)

            # --- imagen de salida ---
            if capture.capturing or background is None:
                out = frame
            elif phase == "vanishing" and active == "naruto":
                age = max(0.04, now - effect_start)
                # La silueta solo hace falta mientras la persona aún se ve (primeros 0.13 s).
                if naruto_mask is None or age < 0.13:
                    naruto_mask = segmenter.mask(to_mp(frame, infer_size), next_ts(now), w, h)
                if naruto_anchor is None:
                    naruto_anchor = substitution_anchor(naruto_mask)
                out = render_substitution(frame, background, naruto_mask, log_sprite, naruto_anchor,
                                          age, log_cache)
                if age >= NARUTO_SECONDS:
                    phase = "vanished"
                    vanished_frame = out
            elif phase in ("vanishing", "appearing"):
                age = now - effect_start
                out = effect.render(background, age, phase == "appearing")
                if age >= TELEPORT_SECONDS:
                    phase = "ready" if phase == "appearing" else "vanished"
                    gate.reset()
            elif phase == "vanished":
                out = vanished_frame if vanished_frame is not None else background
            else:
                out = frame

            if vcam is not None:
                vcam.send(out)

            # --- vista previa (en espejo) ---
            preview = cv2.flip(out, 1)
            if capture.capturing:
                left = capture.seconds_left(now)
                status = f"Sal del encuadre... {left:.0f}" if left > 0.5 else "Capturando fondo..."
            else:
                status = {"ready": "Visible", "vanished": "Desaparecido"}.get(phase, "...")
            holding = 0.0
            if gate.hold_started is not None and gate.armed:
                holding = min(1.0, (now - gate.hold_started) / gate.hold_seconds)
            draw_hud(preview, [f"{MODE_NAMES[mode]} - {status}",
                               "cam virtual: " + ("ON" if vcam else "OFF")
                               + "  |  1 Goku 2 Naruto 3 ambos  B fondo  T efecto  R volver  Q salir"],
                     holding)
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                capture.start(now)
                phase = "calibrating"
                print("[fondo] Sal del encuadre: capturando el fondo en 4 s...", flush=True)
            elif key == ord("t") and background is not None and phase == "ready":
                if enabled("goku"):
                    start_teleport(frame, now)
                else:
                    start_naruto(now)
            elif key in (ord("t"), ord("r")) and phase == "vanished":
                come_back(frame, now)
            elif key in (ord("1"), ord("2"), ord("3")):
                new_mode = {ord("1"): "goku", ord("2"): "naruto", ord("3"): "ambos"}[key]
                if phase in ("ready", "calibrating"):
                    mode = new_mode
                    gate.reset()
                    print(f"[modo] {MODE_NAMES[mode]}", flush=True)
                else:
                    print("[modo] Vuelve a estar visible antes de cambiar de modo.", flush=True)
            elif key == ord("d"):
                diag = not diag
                print(f"[diag] {'activado' if diag else 'desactivado'}", flush=True)
            elif key == ord("s"):
                dump = ROOT / "debug"
                dump.mkdir(exist_ok=True)
                stamp = time.strftime("%H%M%S")
                cv2.imwrite(str(dump / f"{stamp}_camara.png"), frame)
                cv2.imwrite(str(dump / f"{stamp}_salida.png"), out)
                if background is not None:
                    cv2.imwrite(str(dump / f"{stamp}_fondo.png"), background)
                print(f"[debug] Guardado en {dump} ({stamp})", flush=True)
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        recognizer.close()
        naruto_recognizer.close()
        segmenter.close()
        for snd in sounds.values():
            snd.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
