import sys
import argparse
from textwrap import dedent
import inspect
from pathlib import Path

import numpy as np

from toblerone import scripts, Projector
from toblerone._version import __version__
from toblerone.classes import CommonParser, ImageSpace

suffix = f"""
version {__version__}
(C) Tom Kirk, Quantified Imaging, 2023
"""


def main():
    args = sys.argv

    parser = argparse.ArgumentParser(
        prog="toblerone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=suffix,
        usage="toblerone -command-name <options>",
        description=dedent(
            "TOBLERONE Surface-based analysis tools. Run any command with -h for help."
        ),
    )

    funcs = inspect.getmembers(scripts, inspect.isfunction)
    for fname, _ in funcs:
        if fname.startswith("_"):
            continue
        parser.add_argument(f"-{fname.replace('_', '-')}", action="store_true")

    if len(sys.argv) < 2:
        parser.print_help()
        return
    else:
        args = parser.parse_args(sys.argv[1:2])

    for func_name, flag in vars(args).items():
        if flag:
            func = getattr(scripts, func_name)
            break

    arg_names = inspect.signature(func).parameters.keys()
    parser = CommonParser([*arg_names, "out"])
    func_args = parser.parse_args(sys.argv[2:])
    kwargs = vars(func_args)

    out = Path(kwargs.pop("out"))
    result = func(**kwargs)

    if "projector" in func_name:
        if isinstance(result, Projector):
            if not out.suffix:
                out = out.with_suffix(".h5")
            result.save(out)
        else:
            if not out.suffix:
                out = out.with_suffix(".nii.gz")
            result.to_filename(out)
        return

    spc = ImageSpace(kwargs["ref"])

    if isinstance(result, dict):
        out.mkdir(parents=True, exist_ok=True)
        for k, v in result.items():
            spc.save_image(v, out / f"{k}.nii.gz")
        return

    if isinstance(result, np.ndarray):
        if not out.suffix:
            out = out.with_suffix(".nii.gz")
        spc.save_image(result, out)
        return

    raise RuntimeError("Did not capture command's result")


if __name__ == "__main__":
    main()
