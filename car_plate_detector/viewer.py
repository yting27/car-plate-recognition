"""Tkinter viewer and keyboard navigation."""

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class DetectorViewer:
    """Display detector results and handle keyboard navigation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.action = tk.StringVar(root)
        self.allow_pause = False
        self.photo: ImageTk.PhotoImage | None = None
        self.frame_image: Image.Image | None = None

        root.title("Car Plate Detector")
        width = min(1100, round(root.winfo_screenwidth() * 0.85))
        height = min(800, round(root.winfo_screenheight() * 0.85))
        root.geometry(f"{width}x{height}")
        root.minsize(min(480, width), min(320, height))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind("<Key>", self._on_key)

        self.canvas = tk.Canvas(root, highlightthickness=0, background="black")
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.CENTER)
        self.canvas.bind("<Button-1>", self._focus_canvas)
        self.canvas.bind("<Configure>", self._render_frame)

        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.after_idle(self._focus_on_open)

    def _focus_on_open(self) -> None:
        """Raise the window and focus its canvas once it is ready to display."""
        self.root.update_idletasks()
        self.root.lift()
        self.canvas.focus_force()

    def show(self, annotated: np.ndarray) -> None:
        """Update the annotated image."""
        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.frame_image = Image.fromarray(rgb_image)
        self._render_frame()

    def _render_frame(self, event: tk.Event | None = None) -> None:
        """Fit the current frame to the canvas without changing window size."""
        if self.frame_image is None:
            return
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        image = self.frame_image.copy()
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.coords(self.image_item, width / 2, height / 2)
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def wait_for_action(
        self, timeout_ms: int | None = None, *, allow_pause: bool = False
    ) -> str:
        """Handle UI events until a key is pressed or the next frame is due."""
        self.action.set("")
        self.allow_pause = allow_pause
        timer = None
        if timeout_ms is not None:
            timer = self.root.after(timeout_ms, lambda: self.action.set("tick"))
        try:
            self.root.wait_variable(self.action)
            return self.action.get()
        finally:
            self.allow_pause = False
            if timer is not None:
                self.root.after_cancel(timer)

    def quit(self) -> None:
        self.action.set("quit")

    def close(self) -> None:
        self.root.destroy()

    def _on_key(self, event: tk.Event) -> None:
        if event.keysym in {"Return", "KP_Enter"}:
            self.action.set("advance")
        elif event.keysym == "space" and self.allow_pause:
            self.action.set("pause")

    def _focus_canvas(self, event: tk.Event) -> None:
        del event
        self.canvas.focus_set()
