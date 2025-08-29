# Archerfish Analysis – Experiment 1

This project analyses behavioural data from **Experiment 1** on archerfish.

## Project structure
- `data/` → input Excel file (`archer_dataAnalisy8.xlsx`)
- `src/archerfish/*.py` → modules (I/O, preprocessing, plots & stats for Exp.1)
- `scripts/run_analysis.py` → main script to run the analysis
- `config/config.yaml` → configuration file (data path, mapping colors → target numbers)
- `outputs/` → generated figures, stats and reports

## What is generated
- **Learning curves** per target numerosity (2-green, 3-purple, 4-red) with one line per fish
- **Bar plot** of choices across 9 combinations (numerosity × color)
- **Statistical analysis** of non-numeric predictors (ta, tp, radfix, ch, id) using binomial GEE
- `REPORT.md` summarising outputs

## Usage

1. Create a conda environment and install dependencies:
   ```bash
   conda create -n archerfish python=3.11 -y
   conda activate archerfish
   pip install pandas matplotlib seaborn statsmodels pyyaml jinja2 pingouin
   ```

2. Place your Excel data in `data/archer_dataAnalisy8.xlsx`.

3. Run the analysis:
   ```bash
   python -m scripts.run_analysis --config config/config.yaml --outdir outputs
   ```

4. Results will be saved in `outputs/` (figures, csvs, report).
