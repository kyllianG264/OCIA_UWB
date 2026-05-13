import argparse
import tkinter as tk

from uwb_udp import UdpDistanceReceiver


CANVAS_SIZE = 560
CENTER_X = CANVAS_SIZE // 2
CENTER_Y = CANVAS_SIZE // 2


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple live UDP distance viewer.")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--max-age", type=float, default=2.0)
    args = parser.parse_args(argv)

    receiver = UdpDistanceReceiver(bind_ip=args.ip, port=args.port, max_age_s=args.max_age)
    root = tk.Tk()
    root.title("UDP Distance Viewer")
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#0f1220", highlightthickness=0)
    canvas.pack()
    status = tk.StringVar(value="waiting for UDP data...")
    label = tk.Label(root, textvariable=status, anchor="w")
    label.pack(fill="x", padx=8, pady=6)

    def on_close():
        receiver.close()
        root.destroy()

    def tick():
        distances = receiver.get_distances(range(1, 9))
        canvas.delete("all")
        canvas.create_oval(CENTER_X - 4, CENTER_Y - 4, CENTER_X + 4, CENTER_Y + 4, fill="#ffffff", outline="")
        scale = 0.35
        palette = ("#50b4ff", "#78ff78", "#ffb050", "#d090ff", "#ff7099", "#66dddd", "#ffffff", "#aaaaaa")
        for index, anchor_id in enumerate(sorted(distances)):
            radius = clamp(distances[anchor_id] * scale, 8, CANVAS_SIZE // 2 - 12)
            color = palette[index % len(palette)]
            canvas.create_oval(
                CENTER_X - radius,
                CENTER_Y - radius,
                CENTER_X + radius,
                CENTER_Y + radius,
                outline=color,
                width=2,
            )
            canvas.create_text(16, 18 + index * 18, text=f"A{anchor_id}: {distances[anchor_id]:.1f} cm", fill=color, anchor="w")
        status.set(receiver.get_status_text(sorted(distances)))
        root.after(50, tick)

    root.protocol("WM_DELETE_WINDOW", on_close)
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
