

import argparse
import cv2
from ultralytics import YOLO

TARGET_OBJECTS = {"cup", "bottle", "spoon", "fork", "knife", "bowl", "cell phone"}


GRASP_MAP = {
    "cup":        ["cylindrical grasp", "hold"],
    "bottle":     ["cylindrical grasp", "hold"],
    "spoon":      ["pinch grasp", "hold"],
    "fork":       ["pinch grasp", "hold"],
    "knife":      ["pinch grasp", "hold (caution)"],
    "bowl":       ["spherical/wide grasp", "hold"],
    "cell phone": ["precision grip", "hold"],
}


def suggest_action(class_name):
    return GRASP_MAP.get(class_name, ["grasp (unmapped)"])


def main(source, model_name="yolov8n.pt", conf=0.4):
    model = YOLO(model_name)  # auto-downloads pretrained weights on first run

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name not in TARGET_OBJECTS or confidence < conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            action = suggest_action(class_name)[0]
            label = f"{class_name} {confidence:.2f} -> {action}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Detection -> Grasp Suggestion (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="YOLO weights file (auto-downloaded first run)")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Minimum confidence to show a detection")
    args = parser.parse_args()
    main(args.source, args.model, args.conf)
