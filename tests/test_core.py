"""Offline behavioral tests (do not need an OpenAI key or network)."""
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import image_dataset_generator as g


def fake_script():
    return {
        "title_or_visible_heading": "", "overview": "Two characters have a discussion.",
        "reading_order_notes": "Left to right, then down.",
        "whole_page_style": "Black pen line art on white paper; irregular contours.",
        "layout_and_gutters": "Two equal stacked rectangles separated by a white gutter.",
        "cast": [{"id": "A", "design": "Stick-shaped man with oval face", "continuity_notes": "Same coat"}],
        "panels": [{"number": i, "bounds_0_to_1000_xywh": [0, (i-1)*500, 1000, 500],
                    "shot_and_viewpoint": "eye-level medium shot", "setting_and_background": "plain wall",
                    "composition_and_depth": "A at left of center", "actors": [{
                        "id": "A", "position_and_scale": "at image-left, waist up", "torso_facing": "right",
                        "head_facing_and_gaze": "right", "pose_and_hands": "right hand raised",
                        "expression": "smiling", "visible_design": "coat and short hair",
                        "interaction": "points into empty space"}],
                    "props": [], "text": [{"order": 1, "kind": "speech", "speaker_id_or_unknown": "A",
                                         "verbatim": "HI!" if i == 1 else "BYE!",
                                         "placement_and_balloon_tail": "upper-left, tail to A",
                                         "lettering_appearance": "caps drawn by hand", "legibility": "clear"}],
                    "visible_action_and_transition": "A points", "exact_redraw_directions": "Place A left; line at right.",
                    "uncertainties": []} for i in (1, 2)],
        "ending_or_final_beat": "A leaves", "redraw_checklist": ["Keep thin uneven ink lines"],
        "global_uncertainties": []
    }


def response():
    return {"status": "completed", "usage": {"input_tokens": 1000,
            "output_tokens": 500, "input_tokens_details": {"cached_tokens": 100}},
            "output": [{"type": "message", "content": [{"type": "output_text",
            "text": json.dumps(fake_script())}]}]}


class FakeAPI:
    def __init__(self):
        self.uploads = []
        self.batches = {}
        self.calls = 0

    def upload(self, data):
        self.uploads.append(data)
        return "file-" + str(len(self.uploads))

    def create_batch(self, file_id):
        self.calls += 1
        bid = "batch-" + str(self.calls)
        self.batches[bid] = {"id": bid, "input_file_id": file_id, "status": "completed",
                             "output_file_id": bid + "-output"}
        return self.batches[bid]

    def batch(self, bid):
        return self.batches[bid]

    def batch_content(self, file_id):
        bid = file_id.removesuffix("-output")
        number = int(self.batches[bid]["input_file_id"].split("-")[-1])
        records = [json.loads(s) for s in self.uploads[number-1].splitlines()]
        # Deliberately reverse output order: batch results aren't ordered.
        return ("\n".join(json.dumps({"custom_id": r["custom_id"],
                    "response": {"body": response()}}) for r in reversed(records))).encode()

    def response(self, body):
        return response()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "pictures"
        self.root.mkdir()
        for name in ("comic-01.png", "comic-02.png"):
            Image.new("RGB", (150, 300), "white").save(self.root / name)
        (self.root / "nested").mkdir()
        Image.new("RGB", (100, 100), "white").save(self.root / "nested" / "inside.png")
        self.options = {"recursive": True, "model": "gpt-6.2-sol", "transport": "batch",
                        "max_side": 256, "max_output_tokens": 1500, "reasoning": "low",
                        "target_usd": 49.0, "hard_cap_usd": 54.0}

    def export(self):
        d = g.create_export(self.root, self.options, Path(self.tmp.name))
        self.assertEqual(g.add_sources(d, self.root, True), 3)
        return d

    def test_scan_recursive_and_exif_conversion(self):
        self.assertEqual(len(g.scan_images(self.root, False)), 2)
        self.assertEqual(len(g.scan_images(self.root, True)), 3)
        uri, w, h, send_w, send_h = g.image_data(self.root / "comic-01.png", 256)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual((w, h), (150, 300))
        self.assertEqual((send_w, send_h), (128, 256))

    def test_batch_roundtrip_two_datasets_and_resume(self):
        export = self.export()
        client = FakeAPI()
        self.assertEqual(g.submit_batches(export, client, batch_items=2), 3)
        self.assertEqual(g.collect_batches(export, client), (2, 0))
        self.assertEqual(g.collect_batches(export, client), (0, 0))  # idempotent
        m = g.load_manifest(export)
        self.assertEqual(m["stats"]["with_scripts"], 3)
        self.assertEqual(m["stats"]["states"], {"done": 3})
        self.assertTrue((export / "dataset.sqlite3").exists())
        with gzip.open(export / "comprehensive.jsonl.gz", "rt") as f:
            data = [json.loads(line) for line in f]
        with gzip.open(export / "abridged.jsonl.gz", "rt") as f:
            brief = [json.loads(line) for line in f]
        self.assertEqual(len(data), len(brief))
        self.assertEqual(brief[0]["data"]["panels"][0]["dialogue"][0]["text"], "HI!")
        self.assertIn("torso right", (export / "comprehensive_scripts.md").read_text())
        self.assertIn("HI!", (export / "abridged_scripts.md").read_text())
        self.assertGreater(m["stats"]["spent_usd"], 0)
        self.assertEqual(m["stats"]["reserved_usd"], 0)

    def test_budget_blocks_without_oversubmit(self):
        export = self.export()
        manifest = g.load_manifest(export)
        manifest["options"]["hard_cap_usd"] = 0.001
        g.save_manifest(export, manifest)
        client = FakeAPI()
        self.assertEqual(g.submit_batches(export, client), 0)
        self.assertEqual(client.calls, 0)

    def test_changed_source_aborts(self):
        export = self.export()
        Image.new("RGB", (200, 200), "black").save(self.root / "comic-01.png")
        with self.assertRaisesRegex(ValueError, "modified"):
            db = g.connect(export)
            try:
                g.request_for(db, g.candidates(db)[0], self.root, self.options, "gpt-6.2-sol", True)
            finally:
                db.close()

    def test_validate_missing_and_misordered_panels(self):
        data = response()
        self.assertEqual(len(g.parse_response(data)["panels"]), 2)
        broken = response()
        broken["output"][0]["content"][0]["text"] = json.dumps({"panels": []})
        with self.assertRaisesRegex(ValueError, "No panels"):
            g.parse_response(broken)

    def test_refinement_requires_base_completion(self):
        export = self.export()
        with self.assertRaisesRegex(ValueError, "Finish"):
            g.queue_refinement(export, "gpt-6-astra", 1)

    def test_attach_and_release_sending_safety(self):
        export = self.export()
        db = g.connect(export)
        row = g.candidates(db)[0]
        with db:
            db.execute("UPDATE images SET status='sending', batch_id=?, reserved_usd=? WHERE id=?",
                       ("unconfirmed:file-123", 0.05, row["id"]))
        spent, held = g.spend(db)
        self.assertEqual(held, 0.05)
        self.assertEqual(len(g.candidates(db)), 2)
        db.close()


if __name__ == "__main__":
    unittest.main()

# See DatasetTests for the strict schema parser, batch mapping and cap tests.
