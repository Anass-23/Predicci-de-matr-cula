from pathlib import Path
BASE_TMP_PATH = Path(__file__).parent / '../../tmp'
METRICS_TMP_PATH = BASE_TMP_PATH / 'metrics'
PLOTS_TMP_PATH = BASE_TMP_PATH   / 'plots'

if not BASE_TMP_PATH.exists():
    BASE_TMP_PATH.mkdir()
    METRICS_TMP_PATH.mkdir()
    PLOTS_TMP_PATH.mkdir()
    (PLOTS_TMP_PATH / 'trees').mkdir()