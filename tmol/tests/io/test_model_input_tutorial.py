"""Run every cell of the Colab tutorial on each scoring device."""

import json
from pathlib import Path


def test_model_input_notebook(torch_device, monkeypatch):
    root = Path(__file__).parents[3]
    path = root / "notebooks/example_02_model_inputs.ipynb"
    notebook = json.loads(path.read_text())
    monkeypatch.setenv("TMOL_EXAMPLE_SOURCE", str(root))
    monkeypatch.setenv("TMOL_EXAMPLE_DEVICE", str(torch_device))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)

    assert namespace["openfold_pose"].n_poses == 2
    assert set(namespace["rf2_results"]) == {"preserve", "rebuild"}
    assert len(namespace["guidance_energies"]) == 3
