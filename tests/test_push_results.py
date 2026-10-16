"""push_results commits only changed result files and pushes them to the remote."""
import subprocess

from src.utils import push_results


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_push_results_pushes_and_skips_when_clean(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    _git("clone", str(remote), str(work), cwd=tmp_path)
    (work / "README").write_text("seed")   # real remote always has main
    _git("add", "README", cwd=work)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "seed", cwd=work)
    _git("push", "origin", "HEAD:main", cwd=work)
    (work / "results").mkdir()
    (work / "results" / "a.csv").write_text("x\n1\n")
    assert push_results(work, "add results", paths=("results",))
    assert "results/a.csv" in _git("ls-tree", "-r", "--name-only", "main", cwd=remote)
    assert not push_results(work, "nothing new", paths=("results",))
