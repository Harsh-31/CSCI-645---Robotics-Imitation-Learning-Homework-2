"""Render report.md -> report.html -> report.pdf (headless Chrome).

Usage: python report/build_pdf.py   (needs `pip install markdown` and Google Chrome)
"""

import pathlib
import re
import subprocess

import markdown

HERE = pathlib.Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.6in 0.7in; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt;
       line-height: 1.42; color: #1a1a1a; max-width: 7.2in; margin: auto; }
h1 { font-size: 19pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt;
     break-after: avoid; }
table { border-collapse: collapse; margin: 6pt 0 8pt; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 6pt; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Menlo, monospace; font-size: 8.6pt; background: #f4f4f4; padding: 0 2px; }
p.fig { text-align: center; margin: 8pt 0 2pt; break-inside: avoid; break-after: avoid; }
p.fig img { max-width: 100%; }
em { color: #333; }
hr { border: none; border-top: 1px solid #ccc; margin: 10pt 0; }
li { margin: 2pt 0; }
"""


def main():
  md = (HERE / "report.md").read_text()
  # Pandoc-style image width ![alt](src){width=55%} -> centred <img>.
  md = re.sub(r"!\[(.*?)\]\((.*?)\)\{width=(\d+)%\}",
              r'<p class="fig"><img src="\2" alt="\1" style="width:\3%"></p>', md)
  body = markdown.markdown(md, extensions=["tables"])
  html = (f"<!doctype html><html><head><meta charset='utf-8'><title>CSCI 645 HW2 Report"
          f"</title><style>{CSS}</style></head><body>{body}</body></html>")
  (HERE / "report.html").write_text(html)
  subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                  "--allow-file-access-from-files",
                  f"--print-to-pdf={HERE / 'report.pdf'}", (HERE / "report.html").as_uri()],
                 check=True, capture_output=True)
  print("wrote", HERE / "report.pdf")


if __name__ == "__main__":
  main()
