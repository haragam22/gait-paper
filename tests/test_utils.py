"""Small helpers used by notebooks: subset sizing and in-place YAML edits."""
import yaml

from src.utils import set_yaml_scalar, size_subset


def test_size_subset_respects_total_and_per_sweep_budget():
    assert size_subset(10.0, 10, 30) == 360       # 3 h per sweep cap binds
    assert size_subset(10.0, 100, 30) == 36       # total budget binds


def test_set_yaml_scalar_keeps_comments(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text('subsets:\n  attack_n_per_condition: ""   # note\n  pilot_n_per_condition: 10\n')
    set_yaml_scalar(p, "attack_n_per_condition", 40)
    assert yaml.safe_load(p.read_text())["subsets"] == {"attack_n_per_condition": 40, "pilot_n_per_condition": 10}
    assert "# note" in p.read_text()
