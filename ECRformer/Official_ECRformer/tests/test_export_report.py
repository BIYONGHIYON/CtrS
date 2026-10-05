"""Pure helper checks; no inference, training, GPU, or data loading."""
import hashlib
import ast
import io
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

# Extract the exact pure helpers, avoiding optional inference/plot dependencies.
source = Path(__file__).resolve().parents[1] / 'export_test_report.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('choose_examples', 'hash_chunks')]
namespace = {'Path': Path, 'random': random, 'hashlib': hashlib}
exec(compile(tree, str(source), 'exec'), namespace)
exporter = SimpleNamespace(**{name: namespace[name] for name in ('choose_examples', 'hash_chunks')})


class ReportTests(unittest.TestCase):
    def paths(self, groups):
        return [{'S1': f'root/s1_{roi}/sample_{patch}.tif'}
                for roi in range(groups) for patch in range(3)]

    def test_distinct_rois_and_determinism(self):
        paths = self.paths(7)
        chosen, count = exporter.choose_examples(paths, 42)
        self.assertEqual(count, 7)
        self.assertEqual(chosen, exporter.choose_examples(paths, 42)[0])
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len({Path(paths[index]['S1']).parent for index in chosen}), 5)

    def test_small_roi_count_is_not_fabricated(self):
        chosen, count = exporter.choose_examples(self.paths(2), 42)
        self.assertEqual(count, 2)
        self.assertEqual(len(set(chosen)), 5)

    def test_small_dataset(self):
        chosen, count = exporter.choose_examples(self.paths(1), 42)
        self.assertEqual(len(set(chosen)), 3)

    def test_stream_hash(self):
        self.assertEqual(exporter.hash_chunks(io.BytesIO(b'checkpoint')),
                         hashlib.sha256(b'checkpoint').hexdigest())


if __name__ == '__main__':
    unittest.main()
