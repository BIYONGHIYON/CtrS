import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from export_results import choose_samples
from report_graphs import compact_history
from cleanup_experiment import cleanup_plan


class ReportWorkflowTests(unittest.TestCase):
    def test_five_examples_use_unique_scenes_even_with_many_tiles(self):
        selection = choose_samples(75, 4, 20261003)
        self.assertEqual(selection['scene_indices'], [3, 0, 43, 18, 63])
        self.assertEqual(len(set(i//4 for i in selection['sample_indices'])), 5)
        with self.assertRaises(ValueError):
            choose_samples(4, 64, 42)

    def test_cleanup_preserves_weights_metrics_and_other_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for name in ['best.pt','latest.pt','run_config.json','train.log','metrics.json']:
                (run/name).write_text('{}')
            (run/'other_run').mkdir()
            (run/'other_run/train.log').write_text('running')
            self.assertEqual({x.name for x in cleanup_plan(run)}, {'run_config.json','train.log'})
            history = run/'history.jsonl'
            history.write_text('\n'.join(json.dumps(x) for x in [
                {'epoch': 5, 'model_mse': .2}, {'epoch': 5, 'model_mse': .1}, {'epoch': 7,'model_mse': .09}]))
            curves = compact_history(history)
            self.assertEqual([x['epoch'] for x in curves['epochs']], [5,7])
            self.assertEqual(curves['epochs'][0]['model_mse'], .1)


if __name__ == '__main__':
    unittest.main()
