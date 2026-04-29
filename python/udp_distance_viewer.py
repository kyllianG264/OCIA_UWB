import argparse
import tkinter as tk

from uwb_udp import UdpDistanceReceiver


CANVAS_SIZE = 500
CENTER_X = 250
CENTER_Y = 280


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def main():
    parser = argparse.ArgumentParser(description="Viewer UDP simple pour les distances UWB.")
    parser.add_argument("--ip", default="0.0.0.0", help="IP locale a ecouter.")
    parser.add_argument("--port", type=int, default=4210, help="Port UDP a ecouter.")
    parser.add_argument("--anchor", type=int, default=1, help="Ancre a afficher en grand.")
    args = parser.parse_args()

    receiver = UdpDistanceReceiver(bind_ip=args.ip, port=args.port)
    radius = 50.0

    root = tk.Tk()
    root.title("UWB Visualisation")

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#0f172a", highlightthickness=0)
    canvas.pack()

    halo = canvas.create_oval(200, 200, 300, 300, fill="#1e3a8a", outline="")
    circle = canvas.create_oval(210, 210, 290, 290, fill="#3b82f6", outline="")
    canvas.create_text(250, 25, text="UWB Distance", fill="white", font=("Arial", 16, "bold"))
    main_text = canvas.create_text(250, 55, text=f"D{args.anchor} = -- cm", fill="#94a3b8", font=("Arial", 14))
    status_text = canvas.create_text(250, 85, text=f"UDP {args.ip}:{args.port}", fill="#64748b", font=("Arial", 10))
    list_text = canvas.create_text(250, 455, text="", fill="#cbd5e1", font=("Consolas", 10), justify="center")

    def update():
        nonlocal radius

        receiver.poll()
        distances = receiver.get_distances()
        selected = distances.get(args.anchor)

        if selected is not None:
            radius = clamp(selected * 0.5, 8.0, 220.0)
            canvas.itemconfig(main_text, text=f"D{args.anchor} = {selected:.0f} cm")

        lines = [f"A{anchor_id}: {distance:.0f} cm" for anchor_id, distance in sorted(distances.items())]
        canvas.itemconfig(list_text, text="   ".join(lines[:6]))
        canvas.itemconfig(status_text, text=receiver.get_status_text())

        canvas.coords(halo, CENTER_X - radius - 10, CENTER_Y - radius - 10, CENTER_X + radius + 10, CENTER_Y + radius + 10)
        canvas.coords(circle, CENTER_X - radius, CENTER_Y - radius, CENTER_X + radius, CENTER_Y + radius)
        root.after(20, update)

    def on_close():
        receiver.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update()
    root.mainloop()


if __name__ == "__main__":
    main()
