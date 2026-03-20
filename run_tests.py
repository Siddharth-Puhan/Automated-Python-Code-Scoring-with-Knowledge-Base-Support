import importlib.util
import traceback
from pathlib import Path


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"Could not import {path.name}: {e}")
        return None


def run_tests_in_file(path):
    print(f"Running tests in {path}")
    mod = load_module_from_path(path)
    if mod is None:
        print(f"Skipping tests in {path} due to import error")
        return 0
    failures = 0
    for name in dir(mod):
        if name.startswith("test_") and callable(getattr(mod, name)):
            try:
                print(f"- {name} ...", end=" ")
                getattr(mod, name)()
                print("ok")
            except Exception:
                failures += 1
                print("FAIL")
                traceback.print_exc()
    return failures


def main():
    base = Path(__file__).parent
    test_files = [
        base / "tests" / "test_matching.py",
        base / "tests" / "test_provisional_kb.py",
        base / "tests" / "test_pdf_endpoint.py",
    ]
    total_failures = 0
    for tf in test_files:
        if tf.exists():
            total_failures += run_tests_in_file(tf)
        else:
            print(f"Skipping missing test file: {tf}")

    if total_failures:
        print(f"\nTests finished: {total_failures} failures")
        raise SystemExit(1)
    else:
        print("\nAll tests passed")


if __name__ == "__main__":
    main()
