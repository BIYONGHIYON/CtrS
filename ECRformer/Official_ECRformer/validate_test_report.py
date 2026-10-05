"""Validate a compact exported report without models, datasets, or a GPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def validate(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'report_manifest.json').read_text())
    assert manifest['complete'] is True
    for name, expected in manifest['files_sha256'].items():
        path = folder / name
        assert path.resolve().parent == folder.resolve(), 'Unexpected artifact path'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    summary = json.loads((folder / 'summary.json').read_text())
    with (folder / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary['num_samples'] > 0
    assert len({row['index'] for row in rows}) == len(rows)
    assert len({row['sample_id'] for row in rows}) == len(rows)
    for key, value in summary['best_test'].items():
        mean = sum(float(row[key]) for row in rows) / len(rows)
        assert math.isfinite(mean) and math.isclose(mean, value, rel_tol=1e-9, abs_tol=1e-9), key
        baseline = sum(float(row['cloudy_' + key]) for row in rows) / len(rows)
        assert math.isclose(baseline, summary['cloudy_baseline'][key], rel_tol=1e-9, abs_tol=1e-9)
    selection = json.loads((folder / 'selection.json').read_text())
    selected_metrics = json.loads((folder / 'selected_metrics.json').read_text())
    assert selection['selection_before_evaluation'] is True
    assert selection['indices'] == [item['index'] for item in selected_metrics]
    assert len(selection['indices']) == min(5, len(rows))
    indexed = {int(row['index']): row for row in rows}
    for item in selected_metrics:
        assert item['sample_id'] == indexed[item['index']]['sample_id']
        for key, value in item['best'].items():
            assert math.isclose(value, float(indexed[item['index']][key]), rel_tol=1e-9, abs_tol=1e-9)
        assert all(math.isfinite(value) for value in item['last'].values())
    assert not list(folder.glob('*.ckpt')) and not list(folder.glob('*.npz'))
    print(f"Validated {len(rows)} test samples, {len(selection['indices'])} fixed examples and artifact hashes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    validate(parser.parse_args().folder)
