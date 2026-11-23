# Running the code

## Google Colab

1. Open a notebook from its chapter page and click **Open in Colab**.
2. Choose *Runtime → Change runtime type → Hardware accelerator*. A T4 runs every notebook; an A100 or H100
   matches the book's reference numbers. Chapter 6 needs its pinned Triton version (see its page).
3. Run the cells from top to bottom. The first cell installs the tested library versions.

## Local

```bash
git clone https://github.com/kaoutar55/High-Performance-Machine-Learning.git
cd High-Performance-Machine-Learning
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip jupyter
jupyter lab
```

Install PyTorch for your CUDA version first (<https://pytorch.org/get-started/locally/>). Each notebook installs
the other libraries it needs. Without a CUDA GPU the timings are not comparable with the book's.

## Case studies on Modal

Chapters 8, 9, 10 and 12 run their case studies as Modal applications (`modal run chapterNN/...`), so that the
same command measures the same configuration on several GPU types. Each chapter page lists its commands, the
GPU types it uses and where it writes its results. Hugging Face models with gated access need a Modal secret named
`huggingface` that holds your token.

## Building this site

```bash
pip install -r docs/requirements.txt
python tools/docs/build_pages.py   # chapter pages and notebooks from the repository
mkdocs serve
```

The site is rebuilt and published from the default branch by `.github/workflows/docs.yml`.
