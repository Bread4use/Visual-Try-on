import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from upload_photo import upload_photo
from take_photo import open_camera, save_camera_frame
from train import train_lbph_model
from glasses_filter import (
    load_face_predictor,
    predict_faces,
    process_glasses_frame,
    process_webcam_frame,
)


# ---------- Theme ----------
BG = "#101827"
PANEL = "#172337"
PANEL_2 = "#1D2D45"
TEXT = "#F3F6FC"
MUTED = "#A9B7CA"
ACCENT = "#70A7FF"
GREEN = "#43C59E"
RED = "#F07878"


class GlassesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Glasses Studio")
        self.geometry("1100x760")
        self.minsize(900, 650)
        self.configure(bg=BG)

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True, padx=24, pady=20)

        self.pages = {}
        for page_class in (
            HomePage, UploadPage, TakePhotoPage, TrainPage, GlassesPage
        ):
            page = page_class(self.container, self)
            self.pages[page_class.__name__] = page
            page.place(relwidth=1, relheight=1)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_page("HomePage")

    def show_page(self, name):
        # Stop camera when leaving the capture page.
        current = getattr(self, "current_page", None)
        if current and current != name:
            page = self.pages[current]
            if hasattr(page, "stop_camera"):
                page.stop_camera()
        self.current_page = name
        self.pages[name].tkraise()
        page = self.pages[name]
        if hasattr(page, "on_show"):
            page.on_show()

    def on_close(self):
        for page in self.pages.values():
            if hasattr(page, "stop_camera"):
                page.stop_camera()
        self.destroy()


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller

    def heading(self, title, subtitle):
        tk.Label(self, text=title, bg=BG, fg=TEXT,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w", pady=(8, 4))
        tk.Label(self, text=subtitle, bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 22))

    def card(self, parent=None):
        return tk.Frame(parent or self, bg=PANEL, padx=22, pady=20)

    def label(self, parent, text, **kwargs):
        return tk.Label(parent, text=text, bg=PANEL, fg=TEXT,
                        font=("Segoe UI", 10), **kwargs)

    def entry(self, parent, textvariable=None, width=40):
        return tk.Entry(
            parent, textvariable=textvariable, width=width,
            bg=PANEL_2, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 11),
            highlightthickness=1, highlightbackground="#34445D",
            highlightcolor=ACCENT
        )

    def button(self, parent, text, command, primary=False):
        return tk.Button(
            parent, text=text, command=command,
            bg=ACCENT if primary else PANEL_2,
            fg="#0D1726" if primary else TEXT,
            activebackground="#91BCFF" if primary else "#2A3D59",
            activeforeground=TEXT if not primary else "#0D1726",
            relief="flat", bd=0, padx=18, pady=10,
            cursor="hand2", font=("Segoe UI", 10, "bold")
        )

    def back_button(self):
        self.button(self, "←  Back to dashboard",
                    lambda: self.controller.show_page("HomePage")).pack(anchor="w", pady=(0, 14))


class HomePage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="SMART GLASSES", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(22, 4))
        tk.Label(self, text="Studio", bg=BG, fg=TEXT,
                 font=("Segoe UI", 34, "bold")).pack(anchor="w")
        tk.Label(
            self, text="Manage face photos, train your recognition model, "
                       "and preview glasses placement.",
            bg=BG, fg=MUTED, font=("Segoe UI", 11)
        ).pack(anchor="w", pady=(6, 30))

        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x")
        cards = [
            ("01", "Upload photo", "Add an existing JPG or PNG to a person's dataset.",
             "UploadPage", "↥"),
            ("02", "Take photo", "Capture face photos using your webcam.",
             "TakePhotoPage", "◉"),
            ("03", "Train model", "Detect faces and build the LBPH recognition model.",
             "TrainPage", "◎"),
            ("04", "Glasses preview", "Preview glasses alignment on a selected photo.",
             "GlassesPage", "◇"),
        ]
        for index, (number, title, desc, page, symbol) in enumerate(cards):
            card = tk.Frame(grid, bg=PANEL, padx=22, pady=20)
            card.grid(row=index // 2, column=index % 2, sticky="nsew",
                      padx=(0, 12) if index % 2 == 0 else (12, 0),
                      pady=10)
            tk.Label(card, text=f"{number}   {symbol}", bg=PANEL, fg=ACCENT,
                     font=("Segoe UI", 12, "bold")).pack(anchor="w")
            tk.Label(card, text=title, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(12, 5))
            tk.Label(card, text=desc, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 10), wraplength=350, justify="left").pack(anchor="w")
            self.button(card, "Open page  →",
                        lambda p=page: controller.show_page(p),
                        primary=True).pack(anchor="w", pady=(18, 0))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        dataset_count = sum(
            1 for p in Path("dataset").glob("*/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if Path("dataset").exists() else 0
        people_count = sum(1 for p in Path("dataset").iterdir() if p.is_dir()) \
            if Path("dataset").exists() else 0
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", pady=(24, 0))
        tk.Label(footer, text=f"DATASET     {people_count} people  ·  {dataset_count} photos",
                 bg=BG, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(anchor="w")


class UploadPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.heading("Upload photo", "Add a photo from your computer to the face dataset.")
        self.back_button()
        body = self.card()
        body.pack(fill="x", pady=5)

        self.name_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.preview_frame = tk.Frame(body, bg=PANEL_2, width=260, height=200)
        self.preview_frame.grid(row=0, column=1, rowspan=5, padx=(25, 0), pady=5)
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(
            self.preview_frame, text="Photo preview will appear here",
            bg=PANEL_2, fg=MUTED, wraplength=240
        )
        self.preview_label.pack(fill="both", expand=True)

        self.label(body, "PERSON NAME").grid(row=0, column=0, sticky="w")
        self.entry(body, self.name_var, 34).grid(row=1, column=0, sticky="w", pady=(7, 18))
        self.label(body, "SELECTED PHOTO").grid(row=2, column=0, sticky="w")
        self.entry(body, self.path_var, 34).grid(row=3, column=0, sticky="w", pady=7)
        self.button(body, "Browse files", self.browse).grid(row=4, column=0, sticky="w", pady=8)

        self.status = tk.Label(self, text="Supported formats: JPG, JPEG, PNG",
                               bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.status.pack(anchor="w", pady=12)
        self.button(self, "Save photo to dataset", self.save, primary=True).pack(anchor="w")

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose a face photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if not path:
            return
        self.path_var.set(path)
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((260, 200))
            self.preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview, text="")
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def save(self):
        try:
            saved = upload_photo(self.name_var.get(), self.path_var.get())
            self.status.configure(text=f"Saved: {saved}", fg=GREEN)
            messagebox.showinfo("Photo saved", f"Photo added to:\n{saved}")
        except Exception as exc:
            messagebox.showerror("Upload failed", str(exc))


class TakePhotoPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.camera = None
        self.camera_running = False
        self.saved_count = 0
        self.frame_ref = None
        self.recognizer = None
        self.names_dict = {}
        self.recognition_error = None

        self.heading(
            "Take photos",
            "Capture training photos; show identity predictions when a model is available."
        )
        self.back_button()

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)
        left = self.card(content)
        left.pack(side="left", fill="y", padx=(0, 16))
        right = self.card(content)
        right.pack(side="left", fill="both", expand=True)

        self.label(left, "PERSON NAME").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.entry(left, self.name_var, 25).pack(anchor="w", pady=(7, 16))
        self.button(left, "Start webcam", self.start_camera, primary=True).pack(fill="x", pady=5)
        self.button(left, "Capture photo  (Space)", self.capture).pack(fill="x", pady=5)
        self.button(left, "Stop webcam", self.stop_camera).pack(fill="x", pady=5)
        self.count_label = tk.Label(left, text="Photos captured: 0",
                                    bg=PANEL, fg=GREEN, font=("Segoe UI", 11, "bold"))
        self.count_label.pack(anchor="w", pady=(20, 5))
        tk.Label(left, text="Tip: face the camera and use even lighting.",
                 bg=PANEL, fg=MUTED, wraplength=210, justify="left").pack(anchor="w", pady=8)

        self.video_label = tk.Label(right, text="Webcam is stopped",
                                    bg="#0A111D", fg=MUTED,
                                    font=("Segoe UI", 12), width=70, height=25)
        self.video_label.pack(fill="both", expand=True)
        self.camera_status = tk.Label(self, text="Camera stopped",
                                      bg=BG, fg=MUTED)
        self.camera_status.pack(anchor="w", pady=8)

        self.bind_all("<space>", self._space_capture)

    def _space_capture(self, event):
        if self.controller.current_page == "TakePhotoPage" and self.camera_running:
            self.capture()

    def on_show(self):
        self.focus_set()

    def start_camera(self):
        if self.camera_running:
            return
        if not self.name_var.get().strip():
            messagebox.showwarning("Name required", "Enter a person's name first.")
            return
        try:
            self.recognizer, self.names_dict = load_face_predictor()
            self.recognition_error = None
        except Exception as exc:
            self.recognizer = None
            self.names_dict = {}
            self.recognition_error = str(exc)
        try:
            self.camera = open_camera()
            self.camera_running = True
            if self.recognizer is not None:
                status = "Camera running · face recognition on"
                color = GREEN
            elif self.recognition_error:
                status = f"Camera running · prediction unavailable: {self.recognition_error}"
                color = RED
            else:
                status = "Camera running · train a model to enable predictions"
                color = MUTED
            self.camera_status.configure(text=status, fg=color)
            self.update_frame()
        except Exception as exc:
            if self.camera is not None:
                self.camera.release()
            self.camera = None
            self.camera_running = False
            self.recognizer = None
            self.names_dict = {}
            messagebox.showerror("Webcam error", str(exc))

    def update_frame(self):
        if not self.camera_running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            self.camera_status.configure(text="Could not read webcam frame", fg=RED)
            return
        self.current_frame = frame.copy()
        display = frame.copy()
        if self.recognizer is not None:
            try:
                display = predict_faces(display, self.recognizer, self.names_dict)
            except Exception as exc:
                self.recognition_error = str(exc)
                self.recognizer = None
                self.names_dict = {}
                self.camera_status.configure(
                    text=f"Prediction stopped: {self.recognition_error}", fg=RED
                )
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display)
        image.thumbnail((760, 510))
        self.frame_ref = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.frame_ref, text="")
        self.after(20, self.update_frame)

    def capture(self):
        if not self.camera_running or not hasattr(self, "current_frame"):
            messagebox.showwarning("Camera not running", "Start the webcam first.")
            return
        try:
            saved = save_camera_frame(self.current_frame, self.name_var.get())
            self.saved_count += 1
            self.count_label.configure(text=f"Photos captured: {self.saved_count}")
            self.camera_status.configure(text=f"Saved {saved.name}", fg=GREEN)
        except Exception as exc:
            messagebox.showerror("Capture failed", str(exc))

    def stop_camera(self):
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.recognizer = None
        self.names_dict = {}
        if hasattr(self, "video_label"):
            self.video_label.configure(image="", text="Webcam is stopped")
            self.frame_ref = None
        if hasattr(self, "camera_status"):
            self.camera_status.configure(text="Camera stopped", fg=MUTED)


class TrainPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.messages = queue.Queue()
        self.training = False
        self.heading("Train recognition model", "Build the LBPH face-recognition model from dataset/.")
        self.back_button()

        panel = self.card()
        panel.pack(fill="x", pady=5)
        self.label(panel, "TRAINING OUTPUT").pack(anchor="w")
        self.log = tk.Text(panel, height=12, bg="#0A111D", fg="#C9D7EA",
                           insertbackground=TEXT, relief="flat",
                           font=("Consolas", 10), wrap="word")
        self.log.pack(fill="x", pady=(10, 15))
        self.progress = ttk.Progressbar(panel, mode="indeterminate", length=500)
        self.progress.pack(fill="x", pady=5)
        self.status = tk.Label(self, text="Ready to train.",
                               bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.status.pack(anchor="w", pady=12)
        self.start_button = self.button(
            self, "Start model training", self.start_training, primary=True
        )
        self.start_button.pack(anchor="w")
        self.after(100, self.poll_messages)

    def append_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def start_training(self):
        if self.training:
            return
        self.training = True
        self.start_button.configure(state="disabled")
        self.log.delete("1.0", "end")
        self.status.configure(text="Training in progress...", fg=ACCENT)
        self.progress.start(12)

        def worker():
            try:
                result = train_lbph_model(
                    progress_callback=lambda message: self.messages.put(("log", message))
                )
                self.messages.put(("success", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                else:
                    self.training = False
                    self.progress.stop()
                    self.start_button.configure(state="normal")
                    if kind == "success":
                        self.append_log(f"Model: {payload['model']}")
                        self.append_log(f"Names: {payload['names']}")
                        self.status.configure(
                            text=f"Complete: {payload['samples']} face samples, "
                                 f"{payload['people']} people.", fg=GREEN
                        )
                    else:
                        self.append_log("ERROR: " + payload)
                        self.status.configure(text="Training failed.", fg=RED)
        except queue.Empty:
            pass
        self.after(100, self.poll_messages)


class GlassesPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.image_path = None
        self.processed_image = None
        self.preview_ref = None
        self.camera = None
        self.camera_running = False
        self.glasses_image = None
        self.recognizer = None
        self.names_dict = {}

        self.heading(
            "Glasses preview",
            "Preview glasses on a photo or start the webcam for live filtering and person prediction."
        )
        self.back_button()

        controls = self.card()
        controls.pack(fill="x", pady=(0, 14))
        self.path_label = tk.Label(controls, text="No photo selected",
                                   bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
        self.path_label.pack(side="left", padx=(0, 20))
        self.button(controls, "Choose face photo", self.choose_photo).pack(side="left")
        self.button(controls, "Choose glasses PNG", self.choose_glasses).pack(side="left", padx=8)
        self.start_webcam_button = self.button(
            controls, "Start webcam", self.start_webcam, primary=True
        )
        self.start_webcam_button.pack(side="left", padx=(8, 0))
        self.stop_webcam_button = self.button(
            controls, "Stop webcam", self.stop_camera
        )
        self.stop_webcam_button.pack(side="left", padx=8)
        self.glasses_path = "glasses.png"

        self.preview_panel = tk.Label(
            self, text="Choose a photo to preview it here.",
            bg="#0A111D", fg=MUTED, font=("Segoe UI", 12), height=22
        )
        self.preview_panel.pack(fill="both", expand=True)
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", pady=12)
        self.status = tk.Label(bottom, text="Ready", bg=BG, fg=MUTED)
        self.status.pack(side="left")
        self.button(bottom, "Preview", self.process, primary=True).pack(side="right")
        self.button(bottom, "Save result", self.save_result).pack(side="right", padx=8)
        self.stop_webcam_button.configure(state="disabled")

    def choose_photo(self):
        path = filedialog.askopenfilename(
            title="Choose a face photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            image = cv2.imread(path)
            if image is None:
                messagebox.showerror("Image error", "Could not read the selected image.")
                return

            self.stop_camera()
            self.image_path = path
            self.processed_image = None
            self.path_label.configure(text=Path(path).name)
            self._show_preview(image)
            self.status.configure(text="Photo selected. Click Preview to apply the glasses.")

    def choose_glasses(self):
        path = filedialog.askopenfilename(
            title="Choose transparent glasses PNG",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")]
        )
        if path:
            self.stop_camera()
            self.glasses_path = path
            self.processed_image = None
            self.status.configure(text=f"Glasses overlay: {Path(path).name}")

    def _show_preview(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((1000, 480))
        self.preview_ref = ImageTk.PhotoImage(pil)
        self.preview_panel.configure(image=self.preview_ref, text="")

    def start_webcam(self):
        if self.camera_running:
            return

        glasses_image = cv2.imread(self.glasses_path, cv2.IMREAD_UNCHANGED)
        if (
            glasses_image is None
            or glasses_image.ndim != 3
            or glasses_image.shape[2] != 4
        ):
            messagebox.showwarning(
                "Glasses image required",
                "Choose a transparent glasses PNG before starting the webcam.",
            )
            return

        self.glasses_image = glasses_image
        recognition_error = None
        try:
            self.recognizer, self.names_dict = load_face_predictor()
        except Exception as exc:
            self.recognizer = None
            self.names_dict = {}
            recognition_error = str(exc)

        try:
            self.camera = open_camera()
            self.camera_running = True
            self.start_webcam_button.configure(state="disabled")
            self.stop_webcam_button.configure(state="normal")
            if self.recognizer is not None:
                status = "Webcam running · glasses filter and person prediction on"
                color = GREEN
            elif recognition_error:
                status = f"Webcam running · prediction unavailable: {recognition_error}"
                color = RED
            else:
                status = "Webcam running · train a model to enable person prediction"
                color = MUTED
            self.status.configure(text=status, fg=color)
            self.update_webcam_frame()
        except Exception as exc:
            self.stop_camera()
            messagebox.showerror("Webcam error", str(exc))

    def update_webcam_frame(self):
        if not self.camera_running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            self.status.configure(text="Could not read webcam frame.", fg=RED)
            return

        try:
            self.processed_image = process_webcam_frame(
                frame, self.glasses_image, self.recognizer, self.names_dict
            )
        except Exception as exc:
            self.stop_camera()
            self.status.configure(text=f"Webcam filter failed: {exc}", fg=RED)
            messagebox.showerror("Webcam filter failed", str(exc))
            return

        rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((1000, 480))
        self.preview_ref = ImageTk.PhotoImage(image)
        self.preview_panel.configure(image=self.preview_ref, text="")
        self.after(20, self.update_webcam_frame)

    def stop_camera(self):
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.glasses_image = None
        self.recognizer = None
        self.names_dict = {}
        if hasattr(self, "start_webcam_button"):
            self.start_webcam_button.configure(state="normal")
            self.stop_webcam_button.configure(state="disabled")
        if hasattr(self, "status"):
            self.status.configure(text="Webcam stopped", fg=MUTED)

    def process(self):
        if not self.image_path:
            messagebox.showwarning("Photo required", "Choose a face photo first.")
            return
        self.stop_camera()
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Image error", "Could not read the selected image.")
            return
        try:
            glasses_image = cv2.imread(
                self.glasses_path, cv2.IMREAD_UNCHANGED
            )
            if glasses_image is None:
                raise FileNotFoundError(
                    f"Could not read glasses PNG: {self.glasses_path}"
                )
            self.processed_image, face_count = process_glasses_frame(
                image, glasses_image
            )
            self._show_preview(self.processed_image)
            if face_count:
                self.status.configure(
                    text=f"Detected {face_count} face(s). Preview ready.", fg=GREEN
                )
            else:
                self.status.configure(text="No faces detected. Try a clearer, front-facing photo.",
                                      fg=RED)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))

    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Nothing to save", "Click Preview first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")]
        )
        if path:
            if cv2.imwrite(path, self.processed_image):
                self.status.configure(text=f"Saved preview: {path}", fg=GREEN)
            else:
                messagebox.showerror("Save failed", "Could not save the preview.")


if __name__ == "__main__":
    app = GlassesApp()
    app.mainloop()
