import csv
import tempfile
import unittest
from pathlib import Path

from solver_lps.features.uwb.acquisition.data.raw_output import (
    RawCaptureWriter,
    append_rows,
    build_frame_rows,
    delete_raw_file,
    load_last_frame_index,
)
from solver_lps.features.uwb.acquisition.data.review_input import (
    UwbReviewSource,
    UwbTagReviewSource,
)
from solver_lps.features.uwb.acquisition.data.session_assets import (
    default_uwb_raw_path,
    default_uwb_tag_review_path,
)
from solver_lps.features.uwb.acquisition.domain.udp_reader import UdpReader


class _FakeUdpInput:
    bind_ip = "127.0.0.1"
    port = 4210

    def __init__(self):
        self.payloads = [
            {
                "message": "A1=100",
                "addr": ("127.0.0.1", 9000),
                "received_at": 10.0,
            }
        ]

    def poll_payloads(self):
        payloads, self.payloads = self.payloads, []
        return payloads

    def close(self):
        return None


class AcquisitionContractsTest(unittest.TestCase):
    def test_delete_raw_file_resets_next_capture_to_frame_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "uwb_raw.csv"
            writer = RawCaptureWriter(output_path)
            writer.append_packet(timestamp_s=1.0, message="A1=100", addr=("127.0.0.1", 9000), parsed={1: 100.0})

            self.assertTrue(delete_raw_file(output_path))
            restarted = RawCaptureWriter(output_path)

        self.assertEqual(0, restarted.next_frame_index)
    def test_capture_writer_appends_packets_with_continuous_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "uwb_raw.csv"
            writer = RawCaptureWriter(output_path)
            writer.append_packet(timestamp_s=1.0, message="A1=100", addr=("127.0.0.1", 9000), parsed={1: 100.0})
            writer.append_packet(timestamp_s=2.0, message="A1=110", addr=("127.0.0.1", 9000), parsed={1: 110.0})

            resumed_writer = RawCaptureWriter(output_path)
            resumed_writer.append_packet(timestamp_s=3.0, message="A1=120", addr=("127.0.0.1", 9000), parsed={1: 120.0})
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(["0", "1", "2"], [row["frame"] for row in saved])
        self.assertEqual(2, writer.rows_written)
    def test_paths_follow_neutral_uwb_contract(self):
        self.assertEqual(
            default_uwb_raw_path("basket", "set9"),
            Path("solver_lps/assets/basket/set9/uwb/input/uwb_raw.csv").resolve(),
        )
        self.assertEqual(
            default_uwb_tag_review_path("basket", "set9").name,
            "uwb_tag_review.csv",
        )

    def test_raw_output_appends_normalized_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "uwb_raw.csv"
            rows = build_frame_rows(
                frame_index=4,
                timestamp_s=12.5,
                message="A1=123.4",
                addr=("127.0.0.1", 9000),
                parsed={1: 123.4},
                fields={},
            )
            self.assertEqual(append_rows(output_path, rows), 1)
            self.assertEqual(load_last_frame_index(output_path), 4)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["anchor_id"], "1")
            self.assertEqual(saved[0]["distance_cm"], "123.4")

    def test_review_sources_expose_packet_and_cm_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "uwb_raw.csv"
            raw_path.write_text(
                "frame,timestamp_s,anchor_id,distance_cm\n0,0.0,1,100\n0,0.0,2,200\n",
                encoding="utf-8",
            )
            tag_path = Path(directory) / "uwb_tag_review.csv"
            tag_path.write_text(
                "frame,timestamp_s,x_cm,y_cm,z_cm\n0,0.0,10,20,30\n",
                encoding="utf-8",
            )
            packet = UwbReviewSource(raw_path).get_packet([1, 2])
            self.assertTrue(packet["valid"])
            self.assertEqual(packet["raw"], {1: 100.0, 2: 200.0})
            self.assertEqual(UwbTagReviewSource(tag_path).get_position_at(0.0), (10.0, 20.0, 30.0))

    def test_udp_distances_expire_without_new_packets(self):
        now = [10.0]
        reader = UdpReader(_FakeUdpInput(), max_age_s=2.0, clock=lambda: now[0])
        self.assertEqual(reader.get_distances([1]), {1: 100.0})
        now[0] = 12.1
        self.assertEqual(reader.get_distances([1]), {})
        self.assertNotIn(1, reader.latest)


if __name__ == "__main__":
    unittest.main()
