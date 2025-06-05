from pathlib import Path

urls = Path('0724-urls.txt').read_text().split("\n")
aria2_lines = []
for url in urls:
    if url.startswith('#'):
        aria2_lines.append(url)
    else:
        aria2_lines.append(url)
        p = Path(url.replace('https://olmo-data.org/', '')).parent
        aria2_lines.append(f"\tdir={p}")

aria2_lines = "\n".join(aria2_lines)

Path("0724-urls-aria2.txt").write_text(aria2_lines)