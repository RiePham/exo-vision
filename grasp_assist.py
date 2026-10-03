"""
Step 1 + 2 of the exo-vision pipeline:
  1. Detect cups, bottles and phones on the table and identify them (YOLOv8).
  2. Locate each object in a 2D table frame whose origin (0, 0) is the webcam,
     then ask the user whether they want to grab it.

Coordinate frame (top-down view of the table, units = cm):
    +Y = straight ahead of the webcam (distance along the table)
    +X = to the right of the webcam
    webcam = (0, 0)

Distance is estimated with the pinhole camera model using each known
real size:  distance = focal_px * real_size_cm / size_in_pixels.
Press 'c' with ONE object placed exactly --calib-dist cm in front of the webcam
to calibrate focal_px for your webcam, then pass it with --focal next time.

Keys:  y = yes grab   n = no / skip   r = reset   c = calibrate   q = quit
"""

import argparse
import math

import cv2
import numpy as np
from ultralytics import YOLO

# Objects we care about. size_cm is the real size of the box side named by
# "dim" ("h" = height of the box, "w" = width of the box) - tune for your items.
OBJECTS = {
    "cup":        {"grasp": "cylindrical grasp", "size_cm": 9.5,  "dim": "h"},
    "bottle":     {"grasp": "cylindrical grasp", "size_cm": 20.0, "dim": "h"},
    "cell phone": {"grasp": "precision grip",    "size_cm": 7.5,  "dim": "w"},
}

STABLE_FRAMES = 10      # frames an object must be seen before we ask about it
MAX_MISSED = 15         # frames an object can disappear before it is forgotten
MATCH_DIST_PX = 80      # max centre movement between frames for the same object
SMOOTHING = 0.3         # EMA weight for new position measurements

MAP_RANGE_CM = 100      # how far ahead the top-down map shows
MAP_SIZE_PX = 400


class Track:
    def __init__(self, tid, name, box, pos):
        self.id = tid
        self.name = name
        self.box = box
        self.pos = pos          # (x_cm, y_cm) smoothed
        self.hits = 1
        self.missed = 0

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def stable(self):
        return self.hits >= STABLE_FRAMES


class Tracker:
    """Very small tracker: matches detections to tracks by class + nearest centre."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        unmatched = set(self.tracks)
        for name, box, pos in detections:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, best_d = None, MATCH_DIST_PX
            for tid in unmatched:
                t = self.tracks[tid]
                if t.name != name:
                    continue
                d = math.dist((cx, cy), t.center)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                self.tracks[self.next_id] = Track(self.next_id, name, box, pos)
                self.next_id += 1
            else:
                t = self.tracks[best]
                unmatched.discard(best)
                t.box = box
                t.pos = tuple(SMOOTHING * n + (1 - SMOOTHING) * o
                              for n, o in zip(pos, t.pos))
                t.hits += 1
                t.missed = 0
        for tid in unmatched:
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > MAX_MISSED:
                del self.tracks[tid]

    def visible(self):
        return [t for t in self.tracks.values() if t.missed == 0]


def box_size_px(box, dim):
    x1, y1, x2, y2 = box
    return (y2 - y1) if dim == "h" else (x2 - x1)


def locate(box, name, focal_px, frame_w):
    """Box in pixels -> (x_cm, y_cm) on the table, webcam at (0, 0)."""
    spec = OBJECTS[name]
    size_px = max(box_size_px(box, spec["dim"]), 1)
    dist = focal_px * spec["size_cm"] / size_px          # forward distance
    u = (box[0] + box[2]) / 2
    x = (u - frame_w / 2) * dist / focal_px              # sideways offset
    return (x, dist)


def fmt_pos(pos):
    return f"X={pos[0]:+.1f} cm, Y={pos[1]:.1f} cm"


def on_grab(track):
    """Hook for the next step (sending the target to the exoskeleton)."""
    print(f"[GRAB] {track.name} #{track.id} at ({fmt_pos(track.pos)}) "
          f"-> {OBJECTS[track.name]['grasp']}")


def draw_map(tracks, highlight_id):
    """Top-down 2D view of the table with the webcam at (0, 0)."""
    m = np.full((MAP_SIZE_PX, MAP_SIZE_PX, 3), 30, np.uint8)
    scale = (MAP_SIZE_PX - 40) / MAP_RANGE_CM
    ox, oy = MAP_SIZE_PX // 2, MAP_SIZE_PX - 20       # webcam pixel position

    for d in range(10, MAP_RANGE_CM + 1, 10):          # distance rings
        cv2.circle(m, (ox, oy), int(d * scale), (60, 60, 60), 1)
        cv2.putText(m, f"{d}", (ox + 3, oy - int(d * scale) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    cv2.line(m, (0, oy), (MAP_SIZE_PX, oy), (80, 80, 80), 1)
    cv2.line(m, (ox, 0), (ox, oy), (80, 80, 80), 1)
    cv2.rectangle(m, (ox - 8, oy - 5), (ox + 8, oy + 5), (255, 200, 0), -1)
    cv2.putText(m, "webcam (0,0)", (ox + 12, oy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    for t in tracks:
        px = int(ox + t.pos[0] * scale)
        py = int(oy - t.pos[1] * scale)
        color = (0, 0, 255) if t.id == highlight_id else (0, 255, 0)
        cv2.circle(m, (px, py), 6, color, -1)
        cv2.putText(m, f"{t.name} #{t.id}", (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(m, "Top-down table view (cm)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return m


def draw_banner(frame, text, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main(source, model_name, conf, hfov, focal, calib_dist):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    tracker = Tracker()
    mode = "scan"               # scan -> ask -> grab
    active_id = None            # object being asked about / grabbed
    declined = set()            # objects the user said "no" to
    focal_px = focal

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if focal_px is None:    # estimate from the webcam's field of view
            focal_px = (w / 2) / math.tan(math.radians(hfov) / 2)
            print(f"Using estimated focal length {focal_px:.0f}px "
                  f"(HFOV {hfov} deg). Press 'c' to calibrate.")

        # 1. Detect + identify
        detections = []
        for box in model(frame, verbose=False)[0].boxes:
            name = model.names[int(box.cls[0])]
            if name not in OBJECTS or float(box.conf[0]) < conf:
                continue
            b = tuple(map(int, box.xyxy[0]))
            detections.append((name, b, locate(b, name, focal_px, w)))
        tracker.update(detections)
        visible = tracker.visible()

        # 2. Interaction state machine
        active = tracker.tracks.get(active_id)
        if mode in ("ask", "grab") and active is None:
            if mode == "grab":
                print(f"[LOST] target #{active_id} is no longer visible")
            mode, active_id = "scan", None
        if mode == "scan":
            candidates = [t for t in visible
                          if t.stable and t.id not in declined]
            if candidates:  # ask about the nearest object first
                active = min(candidates, key=lambda t: math.hypot(*t.pos))
                active_id, mode = active.id, "ask"

        # 3. Draw
        for t in visible:
            x1, y1, x2, y2 = t.box
            color = (0, 0, 255) if t.id == active_id else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{t.name} #{t.id} ({t.pos[0]:+.0f}, {t.pos[1]:.0f}) cm",
                        (x1, max(y1 - 8, 48)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if mode == "ask":
            draw_banner(frame, f"Grab the {active.name} at ({fmt_pos(active.pos)})?  "
                               f"[y] yes  [n] no", (0, 255, 255))
        elif mode == "grab":
            draw_banner(frame, f"GRABBING {active.name}: {fmt_pos(active.pos)} -> "
                               f"{OBJECTS[active.name]['grasp']}  [r] reset", (0, 0, 255))
        else:
            draw_banner(frame, "Scanning table...  [c] calibrate  [q] quit", (255, 255, 255))

        table_map = cv2.resize(draw_map(visible, active_id), (h, h))
        cv2.imshow("Exo Vision (webcam | table map)", np.hstack([frame, table_map]))

        # 4. Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("y") and mode == "ask":
            mode = "grab"
            on_grab(active)
        elif key == ord("n") and mode == "ask":
            declined.add(active_id)
            mode, active_id = "scan", None
        elif key == ord("r"):
            declined.clear()
            mode, active_id = "scan", None
        elif key == ord("c"):
            if len(visible) != 1:
                print("[CALIB] place exactly ONE object in view and try again")
            else:
                t = visible[0]
                spec = OBJECTS[t.name]
                focal_px = box_size_px(t.box, spec["dim"]) * calib_dist / spec["size_cm"]
                print(f"[CALIB] focal length = {focal_px:.0f}px "
                      f"(use --focal {focal_px:.0f} next time)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Minimum confidence to accept a detection")
    parser.add_argument("--hfov", type=float, default=60.0,
                        help="Webcam horizontal field of view in degrees (if not calibrated)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Calibrated focal length in pixels")
    parser.add_argument("--calib-dist", type=float, default=30.0,
                        help="Distance (cm) of the object used for 'c' calibration")
    args = parser.parse_args()
    main(args.source, args.model, args.conf, args.hfov, args.focal, args.calib_dist)
