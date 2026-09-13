"""Refresh LESSON.md from the maintained article.md. Charts come from plot.py."""
from pathlib import Path
p=Path(__file__).resolve().parent
(p/'LESSON.md').write_text((p/'article.md').read_text())
print('Refreshed lesson from article.md')
