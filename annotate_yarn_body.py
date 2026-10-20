"""Small manual polygon annotator for creating main-yarn-body binary masks.

Usage (from the Skein project root):
  python annotate_yarn_body.py --images data/raw_photos --out data/yarn_masks/manual

Controls:
  Left click: add polygon point
  U or Backspace: undo last point
  R: reset current polygon
  C or Enter: close polygon for preview
  S: save the mask and overlay, then move to next image
  N: skip image without saving
  Q or Esc: quit

The mask convention is white (255) = main yarn body; black (0) = background.
Trace the main compact yarn body only. Exclude long loose flyaway fibers and loops
if the measurement definition is the main yarn-body width. Be consistent.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class PolygonAnnotator:
    def __init__(self, images_dir: Path, out_dir: Path, display_width: int = 1200):
        self.images_dir = images_dir
        self.out_dir = out_dir
        self.mask_dir = out_dir / "masks"
        self.overlay_dir = out_dir / "overlays"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.display_width = display_width
        self.paths = sorted(p for p in images_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in EXTS
                            and not p.name.lower().startswith("label_preview"))
        self.index = 0
        self.points: list[tuple[int, int]] = []
        self.closed = False
        self.image = None
        self.scale = 1.0
        self.window = "Skein — outline the MAIN YARN BODY"
        self.banner_h = 58

    def mouse(self, event, x, y, flags, param):
        if self.image is None or event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.closed or y < self.banner_h:
            return
        h, w = self.image.shape[:2]
        scale = min(1.0, self.display_width / w)
        px, py = int(round(x / scale)), int(round((y - self.banner_h) / scale))
        if 0 <= px < w and 0 <= py < h:
            self.points.append((px, py))

    def render(self):
        img = self.image.copy()
        if self.points:
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            if len(self.points) >= 2:
                cv2.polylines(img, [pts], self.closed, (0, 220, 70), 2, cv2.LINE_AA)
            for p in self.points:
                cv2.circle(img, p, max(3, img.shape[1] // 300), (0, 0, 255), -1, cv2.LINE_AA)
            if self.closed and len(self.points) >= 3:
                fill = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(fill, [np.array(self.points, dtype=np.int32)], 255)
                tint = np.zeros_like(img)
                tint[:, :] = (50, 220, 70)
                blend = cv2.addWeighted(img, 0.65, tint, 0.35, 0)
                img[fill > 0] = blend[fill > 0]
                cv2.polylines(img, [pts], True, (0, 220, 70), 2, cv2.LINE_AA)
        scale = min(1.0, self.display_width / img.shape[1])
        if scale != 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        banner = np.full((self.banner_h, img.shape[1], 3), (25, 39, 31), dtype=np.uint8)
        name = self.paths[self.index].name if self.paths else ""
        lines = [f"Image {self.index + 1}/{len(self.paths)}: {name}",
                 "LEFT click=point | C/ENTER=close | U=undo | R=reset | S=save+next | N=skip | Q=quit"]
        y = 23
        for line in lines:
            cv2.putText(banner, line[:150], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 235), 1, cv2.LINE_AA)
            y += 27
        return np.vstack([banner, img])

    def save(self):
        if len(self.points) < 3 or not self.closed:
            print("Not saved: draw at least 3 points and close the polygon with C/Enter first.")
            return False
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points, dtype=np.int32)], 255)
        stem = self.paths[self.index].stem
        mask_path = self.mask_dir / f"{stem}_mask.png"
        overlay_path = self.overlay_dir / f"{stem}_overlay.jpg"
        overlay = self.image.copy()
        tint = np.zeros_like(overlay); tint[:] = (50, 220, 70)
        blend = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
        overlay[mask > 0] = blend[mask > 0]
        cv2.polylines(overlay, [np.array(self.points, dtype=np.int32)], True, (0, 220, 70), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(mask_path), mask):
            print(f"ERROR: could not save {mask_path}")
            return False
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Saved mask: {mask_path}")
        print(f"Saved overlay: {overlay_path}")
        return True

    def run(self):
        if not self.paths:
            print(f"No supported images found in {self.images_dir}")
            return 2
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        while self.index < len(self.paths):
            p = self.paths[self.index]
            self.image = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if self.image is None:
                print(f"Skipping unreadable image: {p}")
                self.index += 1
                continue
            self.points = []
            self.closed = False
            while True:
                cv2.imshow(self.window, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows(); return 0
                if key in (ord('u'), 8, 127):
                    if self.closed:
                        self.closed = False
                    if self.points:
                        self.points.pop()
                elif key == ord('r'):
                    self.points = []; self.closed = False
                elif key in (ord('c'), 10, 13):
                    if len(self.points) >= 3:
                        self.closed = True
                    else:
                        print("Add at least 3 points before closing the polygon.")
                elif key == ord('s'):
                    if self.save():
                        self.index += 1
                        break
                elif key == ord('n'):
                    print(f"Skipped: {p.name}")
                    self.index += 1
                    break
            if self.index >= len(self.paths):
                break
        cv2.destroyAllWindows()
        print("Finished. Review the saved overlays before using masks for training.")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Manually annotate the main yarn body as a binary polygon mask.")
    parser.add_argument("--images", type=Path, default=Path("data/raw_photos"), help="Folder containing yarn images")
    parser.add_argument("--out", type=Path, default=Path("data/yarn_masks/manual"), help="Output folder for masks and overlays")
    parser.add_argument("--display-width", type=int, default=1200, help="Maximum display width in pixels")
    args = parser.parse_args()
    if not args.images.exists():
        print(f"Image folder does not exist: {args.images}", file=sys.stderr)
        return 2
    return PolygonAnnotator(args.images, args.out, args.display_width).run()

if __name__ == "__main__":
    raise SystemExit(main())
