"""Execute preserved source snapshots; never initialize or edit the submodule."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
provenance = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
for filename, key in (("upstream_snapshot.py", "upstream_sha256"),
                      ("adapter_snapshot.py", "adapter_sha256"),
                      ("metrics_snapshot.py", "metrics_sha256")):
    if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != provenance[key]:
        raise ValueError(f"Pinned source snapshot checksum mismatch: {filename}")
spec = importlib.util.spec_from_file_location("variable_scale_upstream", HERE / "upstream_snapshot.py")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
tree = ast.parse((HERE / "adapter_snapshot.py").read_text(encoding="utf-8"))
# Only substitute source discovery. The original restored class is unchanged.
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.FunctionDef) and node.name == "_upstream_model")]
namespace = {"__file__": str(HERE / "adapter_snapshot.py"), "__name__": "variable_scale.reference_adapter",
             "_upstream_model": lambda: upstream.PansharpeningNet}
exec(compile(tree, str(HERE / "adapter_snapshot.py"), "exec"), namespace)
ReferenceX4 = namespace["RestoredPansharpeningNet"]
