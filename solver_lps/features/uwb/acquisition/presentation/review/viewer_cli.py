import argparse
import tkinter as tk
from tkinter import ttk

from solver_lps.features.uwb.acquisition.data.review_input import (
    DEFAULT_UWB_REVIEW_LOG_PATH,
    UwbReviewSource,
)
from solver_lps.presentation.navigation import RETURN_HOME


REFRESH_MS = 120


def _distance_summary(anchor_map):
    if not anchor_map:
        return "-"
    parts = []
    for anchor_id in sorted(anchor_map):
        parts.append(f"A{anchor_id}={anchor_map[anchor_id]:.1f}cm")
    return " | ".join(parts)


def _anchor_list(anchor_map):
    if not anchor_map:
        return "-"
    return ", ".join(f"A{anchor_id}" for anchor_id in sorted(anchor_map))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review UWB viewer based on CSV playback.")
    parser.add_argument("--uwb-log", default=DEFAULT_UWB_REVIEW_LOG_PATH)
    parser.add_argument("--window-title", default="UDP Viewer - Review UWB")
    parser.add_argument("--header-title", default="UWB Viewer - review CSV")
    args = parser.parse_args(argv)

    review = UwbReviewSource(args.uwb_log)
    root = tk.Tk()
    root.title(args.window_title)
    root.geometry("1280x720")
    root.configure(bg="#10141c")

    status = tk.StringVar(value="review UWB en attente...")
    rows_by_frame = {}
    navigation_result = {"value": 0}

    header = tk.Label(
        root,
        text=args.header_title,
        bg="#10141c",
        fg="#f2f6ff",
        anchor="w",
        font=("Arial", 16, "bold"),
    )
    header.pack(fill="x", padx=10, pady=(10, 6))

    def return_to_home():
        navigation_result["value"] = RETURN_HOME
        root.destroy()

    tk.Button(
        root,
        text="Retour accueil",
        command=return_to_home,
        bg="#193b55",
        fg="#ffffff",
        activebackground="#245a7e",
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=7,
    ).pack(anchor="e", padx=10, pady=(0, 6))

    controls = tk.Label(
        root,
        text="Space: pause/reprise | Gauche/Droite: +-15 frames | PageUp/PageDown: +-5s",
        bg="#10141c",
        fg="#d0dbeb",
        anchor="w",
        font=("Arial", 11),
    )
    controls.pack(fill="x", padx=10, pady=(0, 8))

    columns = ("frame", "timestamp", "anchors", "distances")
    table = ttk.Treeview(root, columns=columns, show="headings", height=14)
    table.heading("frame", text="Frame")
    table.heading("timestamp", text="Timestamp")
    table.heading("anchors", text="Ancres")
    table.heading("distances", text="Distances")
    table.column("frame", width=90, anchor="center")
    table.column("timestamp", width=120, anchor="center")
    table.column("anchors", width=160, anchor="w")
    table.column("distances", width=820, anchor="w")
    table.pack(fill="both", expand=False, padx=10, pady=(0, 8))

    detail_label = tk.Label(
        root,
        text="Details de la frame",
        bg="#10141c",
        fg="#d8e2f0",
        anchor="w",
        font=("Arial", 11, "bold"),
    )
    detail_label.pack(fill="x", padx=10, pady=(0, 4))

    detail_text = tk.Text(
        root,
        height=12,
        bg="#0b0f15",
        fg="#e8eef8",
        insertbackground="#ffffff",
        wrap="word",
        relief="flat",
        font=("Consolas", 10),
    )
    detail_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))
    detail_text.insert("1.0", "Aucune frame selectionnee.")
    detail_text.config(state="disabled")

    label = tk.Label(root, textvariable=status, anchor="w", bg="#10141c", fg="#aebdd1")
    label.pack(fill="x", padx=10, pady=(0, 10))

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", background="#0f1220", foreground="#edf2fb", fieldbackground="#0f1220", rowheight=24)
    style.configure("Treeview.Heading", background="#1b2432", foreground="#f2f6ff")

    def render_details(_event=None):
        selection = table.selection()
        if not selection:
            return
        frame_key = selection[0]
        frame = rows_by_frame.get(frame_key)
        if frame is None:
            return
        display_frame = frame.get("frame")
        if display_frame is None:
            display_frame = int(frame.get("sequence_index", 0)) + 1
        lines = [
            f"Frame: {display_frame}",
            f"Timestamp: {frame['timestamp_s']:.3f}s",
            f"Ancres: {_anchor_list(frame['distances'])}",
            "Distances:",
        ]
        for anchor_id in sorted(frame["distances"]):
            lines.append(f"  A{anchor_id}: {frame['distances'][anchor_id]:.2f} cm")
        detail_text.config(state="normal")
        detail_text.delete("1.0", tk.END)
        detail_text.insert("1.0", "\n".join(lines))
        detail_text.config(state="disabled")

    def on_close():
        root.destroy()

    def on_key(event):
        if event.keysym == "space":
            review.toggle_pause()
        elif event.keysym == "Left":
            review.seek_frames(-15)
        elif event.keysym == "Right":
            review.seek_frames(15)
        elif event.keysym == "Prior":
            review.seek_relative(-5.0)
        elif event.keysym == "Next":
            review.seek_relative(5.0)

    def tick():
        if not review.frames:
            status.set(f"Aucun CSV review charge | {args.uwb_log}")
            root.after(REFRESH_MS, tick)
            return

        review.frame_index = review.resolve_index()
        current_frame = review.frames[review.frame_index]
        frame_key = str(current_frame["sequence_index"])
        rows_by_frame[frame_key] = current_frame

        values = (
            current_frame["sequence_index"] + 1,
            f"{current_frame['timestamp_s']:.3f}s",
            _anchor_list(current_frame["distances"]),
            _distance_summary(current_frame["distances"]),
        )
        if table.exists(frame_key):
            table.item(frame_key, values=values)
        else:
            table.insert("", tk.END, iid=frame_key, values=values)

        playback = review.playback_state
        state_text = "pause" if playback["paused"] else "lecture"
        status.set(
            f"Review CSV | frame {playback['frame_index'] + 1}/{playback['frame_count']} | "
            f"{playback['position_s']:.2f}s / {playback['duration_s']:.2f}s | {state_text}"
        )

        if table.selection() != (frame_key,):
            table.selection_set(frame_key)
            table.see(frame_key)
            render_details()

        root.after(REFRESH_MS, tick)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Key>", on_key)
    table.bind("<<TreeviewSelect>>", render_details)
    tick()
    root.mainloop()
    return navigation_result["value"]


if __name__ == "__main__":
    main()
