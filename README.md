# Platitude

Speech analysis tool.
The main target is finding boring parts which mean nothing.
This parts are polite, mandatory for readability, but they are everywhere.

The tool use french president speeches, but it can use anything recurrent.


## Install

```bash
uv sync
source .venv/bin/activate
```

## Use it

Fetch lots of speeches

```bash
python -m platitude.speeches.scrape --output chirac.jsonl --max-pages 1000  --who "Jacques Chirac"
```

Launch Jupyter

```bash
jupyter lab
```

Tinker the notebook :

http://localhost:8888/lab/tree/notebooks/platitude.ipynb
