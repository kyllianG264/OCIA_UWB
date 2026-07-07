import argparse
import tkinter as tk
from tkinter import messagebox, ttk

from solver_lps.features.uwb.acquisition.data.raw_output import RawCaptureWriter, delete_raw_file
from solver_lps.features.uwb.acquisition.domain.udp_reader import UdpDistanceReceiver, parse_message_fields
from solver_lps.presentation.navigation import RETURN_HOME


REFRESH_MS = 120
DEVICE_ID_KEYS = ("mac", "ble", "bt", "device", "dev", "tag", "uid", "serial", "sn", "name")


def _device_identifier(fields, addr):
    for key in DEVICE_ID_KEYS:
        value = fields.get(key)
        if value:
            return str(value)
    return f"{addr[0]}:{addr[1]}"


def _event_name(fields):
    return str(fields.get("event") or fields.get("type") or "DISTANCE").upper()


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
    parser = argparse.ArgumentParser(description="Live multi-UWB UDP viewer without graphs.")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--max-age", type=float, default=2.0)
    parser.add_argument("--capture-output", default=None)
    parser.add_argument("--window-title", default="UDP Viewer - Multi UWB")
    parser.add_argument("--header-title", default="UWB Viewer - sources detectees")
    args = parser.parse_args(argv)

    receiver = UdpDistanceReceiver(bind_ip=args.ip, port=args.port, max_age_s=args.max_age)
    root = tk.Tk()
    root.title(args.window_title)
    root.geometry("1280x720")
    root.configure(bg="#10141c")

    status = tk.StringVar(value="waiting for UWB packets...")
    recording_status = tk.StringVar(value="Enregistrement arrete")
    devices = {}
    capture_writer = RawCaptureWriter(args.capture_output) if args.capture_output else None
    recording = {"active": False}
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

    recording_bar = tk.Frame(root, bg="#10141c")
    recording_bar.pack(fill="x", padx=10, pady=(0, 8))

    def toggle_recording():
        if capture_writer is None:
            recording_status.set("Aucun fichier de sortie configure")
            return
        recording["active"] = not recording["active"]
        if recording["active"]:
            record_button.config(text="Arreter l'enregistrement", bg="#b63a3a")
            recording_status.set(f"REC -> {capture_writer.csv_path}")
        else:
            record_button.config(text="Demarrer l'enregistrement", bg="#287a4b")
            recording_status.set(
                f"Enregistrement arrete | {capture_writer.rows_written} lignes -> {capture_writer.csv_path}"
            )

    record_button = tk.Button(
        recording_bar,
        text="Demarrer l'enregistrement",
        command=toggle_recording,
        bg="#287a4b",
        fg="#ffffff",
        activebackground="#32945c",
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=7,
    )
    record_button.pack(side="left")
    def delete_recording():
        nonlocal capture_writer
        if capture_writer is None:
            recording_status.set("Aucun fichier de sortie configure")
            return
        if not messagebox.askyesno(
            "Supprimer le raw UWB",
            f"Supprimer definitivement ce fichier ?\n\n{capture_writer.csv_path}",
            parent=root,
        ):
            return
        recording["active"] = False
        record_button.config(text="Demarrer l'enregistrement", bg="#287a4b")
        deleted = delete_raw_file(capture_writer.csv_path)
        capture_writer = RawCaptureWriter(capture_writer.csv_path)
        recording_status.set(
            "Raw supprime, prochain enregistrement a la frame 0"
            if deleted
            else "Aucun raw existant, prochain enregistrement a la frame 0"
        )

    tk.Button(
        recording_bar,
        text="Supprimer le raw",
        command=delete_recording,
        bg="#6f2830",
        fg="#ffffff",
        activebackground="#91343f",
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=7,
    ).pack(side="left", padx=(8, 0))
    tk.Label(
        recording_bar,
        textvariable=recording_status,
        bg="#10141c",
        fg="#d0dbeb",
        anchor="w",
    ).pack(side="left", fill="x", expand=True, padx=(12, 0))

    def return_to_home():
        navigation_result["value"] = RETURN_HOME
        receiver.close()
        root.destroy()

    tk.Button(
        recording_bar,
        text="Retour accueil",
        command=return_to_home,
        bg="#193b55",
        fg="#ffffff",
        activebackground="#245a7e",
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=7,
    ).pack(side="right", padx=(8, 0))

    columns = ("device", "ip", "port", "event", "anchors", "distances", "last_seen")
    table = ttk.Treeview(root, columns=columns, show="headings", height=14)
    table.heading("device", text="UWB / Device")
    table.heading("ip", text="IP")
    table.heading("port", text="Port")
    table.heading("event", text="Event")
    table.heading("anchors", text="Ancres")
    table.heading("distances", text="Distances")
    table.heading("last_seen", text="Last Seen")
    table.column("device", width=170, anchor="w")
    table.column("ip", width=140, anchor="w")
    table.column("port", width=70, anchor="center")
    table.column("event", width=90, anchor="center")
    table.column("anchors", width=120, anchor="w")
    table.column("distances", width=470, anchor="w")
    table.column("last_seen", width=90, anchor="center")
    table.pack(fill="both", expand=False, padx=10, pady=(0, 8))

    detail_label = tk.Label(
        root,
        text="Dernier message brut",
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
    detail_text.insert("1.0", "Aucun message selectionne.")
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
        device_key = selection[0]
        device = devices.get(device_key)
        if not device:
            return
        fields = device.get("fields", {})
        lines = [
            f"Device: {device.get('display_id', device_key)}",
            f"IP: {device.get('ip', '-')}:{device.get('port', '-')}",
            f"Event: {device.get('event', '-')}",
            f"Ancres: {_anchor_list(device.get('anchors', {}))}",
            f"Distances: {_distance_summary(device.get('anchors', {}))}",
            "",
            f"Raw: {device.get('message', '')}",
            "",
            "Fields:",
        ]
        for key in sorted(fields):
            lines.append(f"  {key}: {fields[key]}")
        detail_text.config(state="normal")
        detail_text.delete("1.0", tk.END)
        detail_text.insert("1.0", "\n".join(lines))
        detail_text.config(state="disabled")

    def on_close():
        receiver.close()
        root.destroy()

    def tick():
        now = receiver.last_poll_time or 0.0
        for message, addr, parsed in receiver.poll():
            fields = parse_message_fields(message)
            if recording["active"] and capture_writer is not None and parsed:
                capture_writer.append_packet(
                    timestamp_s=receiver.last_poll_time or 0.0,
                    message=message,
                    addr=addr,
                    parsed=parsed,
                    fields=fields,
                )
            device_key = _device_identifier(fields, addr)
            anchors = {int(anchor_id): float(distance_cm) for anchor_id, distance_cm in parsed.items()}
            devices[device_key] = {
                "display_id": device_key,
                "ip": addr[0],
                "port": addr[1],
                "event": _event_name(fields),
                "anchors": anchors,
                "message": message,
                "fields": fields,
                "timestamp": receiver.last_poll_time,
            }

        now = receiver.last_poll_time or now
        stale_keys = []
        for device_key, device in devices.items():
            timestamp = float(device.get("timestamp") or 0.0)
            if now and (now - timestamp) > float(args.max_age):
                stale_keys.append(device_key)
        for device_key in stale_keys:
            devices.pop(device_key, None)
            if table.exists(device_key):
                table.delete(device_key)

        active_devices = sorted(devices.items(), key=lambda item: (item[1].get("ip", ""), item[1].get("port", 0), item[0]))
        for device_key, device in active_devices:
            last_seen = "-" if not now else f"{max(0.0, now - float(device.get('timestamp') or now)):.1f}s"
            values = (
                device.get("display_id", device_key),
                device.get("ip", "-"),
                device.get("port", "-"),
                device.get("event", "-"),
                _anchor_list(device.get("anchors", {})),
                _distance_summary(device.get("anchors", {})),
                last_seen,
            )
            if table.exists(device_key):
                table.item(device_key, values=values)
            else:
                table.insert("", tk.END, iid=device_key, values=values)

        if active_devices:
            anchor_count = sum(len(device.get("anchors", {})) for _, device in active_devices)
            status.set(f"{len(active_devices)} source(s) UWB detectee(s) | {anchor_count} distance(s) visibles | {receiver.bind_ip}:{receiver.port}")
        else:
            status.set(f"Aucune source UWB recente | {receiver.bind_ip}:{receiver.port}")

        if not table.selection():
            children = table.get_children("")
            if children:
                table.selection_set(children[0])
                render_details()

        root.after(REFRESH_MS, tick)

    root.protocol("WM_DELETE_WINDOW", on_close)
    table.bind("<<TreeviewSelect>>", render_details)
    tick()
    root.mainloop()
    return navigation_result["value"]


if __name__ == "__main__":
    main()
