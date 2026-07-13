# Publishing TreeMMM to PyPI

Publishing is Author-only. These steps stage a release on TestPyPI, verify it
in a clean environment, and only then upload the identical artifacts to PyPI.
Never store an API token in this repository; use Twine's prompt, keyring, or an
environment-level secret.

## 1. Verify the release checkout

Confirm that the intended release commit is clean, that `pyproject.toml` and
`treemmm/__init__.py` carry the same version, and that the full test harness is
green. For the current release candidate, both versions must be `0.3.1`.

```bash
python -m pip install --upgrade build twine
pytest -m "not slow" -q
python examples/quickstart_pharma.py
```

## 2. Build and inspect artifacts

Build from the repository root. Upload only the explicitly versioned files so
that unrelated artifacts already present in `dist/` cannot be published.

```bash
python -m build
python -m twine check dist/treemmm-0.3.1*
```

The expected outputs are a wheel and source archive whose names begin with
`treemmm-0.3.1`.

## 3. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/treemmm-0.3.1*
```

Create a fresh virtual environment and install the staged package. TestPyPI
does not mirror every dependency, so retain PyPI as the dependency index.

```bash
python -m venv .venv-testpypi
.venv-testpypi/bin/python -m pip install --upgrade pip
.venv-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  treemmm==0.3.1
```

On Windows, replace `.venv-testpypi/bin/python` with
`.venv-testpypi\\Scripts\\python.exe`.

Run a capability-import smoke test in that environment:

```bash
python -c "import treemmm; from treemmm.mroi import reallocate, reallocate_curve; from treemmm.core.preprocessing.adstock import apply_panel_adstock; from treemmm.demo.datasets.pharma_adstock import generate_pharma_adstock_dataset; from treemmm.demo.datasets.geo_panel import generate_geo_panel_dataset; assert treemmm.__version__ == '0.3.1'"
```

## 4. Upload the same files to PyPI

After the TestPyPI install and smoke test pass, upload the identical artifacts:

```bash
python -m twine upload dist/treemmm-0.3.1*
```

Finally, repeat the clean-environment install from PyPI with
`python -m pip install treemmm==0.3.1` and rerun the capability-import smoke
test. PyPI distributions are immutable; if the upload is wrong, increment the
package version rather than replacing an existing file.
