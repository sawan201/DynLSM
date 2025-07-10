from line_profiler import load_stats, show_text
import contextlib

# 1. Load your .lprof data
stats = load_stats('main.py.lprof')

# 2. Open a text file and redirect print output into it
with open('profile_results.txt', 'w') as f, contextlib.redirect_stdout(f):
    # show_text takes (timings, unit) and prints the table to stdout
    show_text(stats.timings, stats.unit)

# When the with-block ends, profile_results.txt contains the full report.
