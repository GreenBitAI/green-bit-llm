import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "green_bit_llm"
with open(Path(__file__).parent / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from version import __version__

setup(
    name="green-bit-llm",
    version=__version__,
    description="A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI's LLMs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="team@greenbit.ai",
    author="GreenBitAI Contributors",
    url="https://github.com/GreenBitAI/green-bit-llm",
    license="Apache-2.0",
    install_requires=requirements,
    packages=["green_bit_llm", "green_bit_llm.common", "green_bit_llm.evaluation", "green_bit_llm.inference", "green_bit_llm.sft"],
    python_requires=">=3.9",
)
