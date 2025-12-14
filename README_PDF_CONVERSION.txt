================================================================================
HOW TO CONVERT MARKDOWN TO PDF
================================================================================

File created: section-Obejero-Llatuna.md
Target output: section-Obejero-Llatuna.pdf

================================================================================
OPTION 1: Using Online Converter (Easiest)
================================================================================

1. Visit: https://www.markdowntopdf.com/
   OR: https://md2pdf.netlify.app/
   OR: https://dillinger.io/

2. Upload or paste the content of section-Obejero-Llatuna.md

3. Click "Convert to PDF" or "Export as PDF"

4. Download as: section-Obejero-Llatuna.pdf

================================================================================
OPTION 2: Using Pandoc (Best Quality)
================================================================================

1. Install Pandoc:
   - Windows: Download from https://pandoc.org/installing.html
   - Or use: choco install pandoc

2. Install LaTeX (for better PDF rendering):
   - MiKTeX: https://miktex.org/download
   - Or: TeX Live

3. Run command:
   pandoc section-Obejero-Llatuna.md -o section-Obejero-Llatuna.pdf --pdf-engine=pdflatex

   With table of contents:
   pandoc section-Obejero-Llatuna.md -o section-Obejero-Llatuna.pdf --pdf-engine=pdflatex --toc

================================================================================
OPTION 3: Using VS Code Extension
================================================================================

1. Install VS Code extension: "Markdown PDF" by yzane

2. Open section-Obejero-Llatuna.md in VS Code

3. Right-click in editor → "Markdown PDF: Export (pdf)"

4. File will be saved as section-Obejero-Llatuna.pdf

================================================================================
OPTION 4: Using Python (markdown2pdf)
================================================================================

1. Install package:
   pip install markdown-pdf

2. Run:
   python -m markdown_pdf section-Obejero-Llatuna.md

================================================================================
OPTION 5: Using Chrome/Edge Browser
================================================================================

1. Install extension: "Markdown Viewer" or "Markdown Preview Plus"

2. Open section-Obejero-Llatuna.md in browser

3. Press Ctrl+P (Print)

4. Select "Save as PDF"

5. Save as: section-Obejero-Llatuna.pdf

================================================================================
RECOMMENDED: Pandoc (Option 2)
================================================================================

Pandoc provides the best formatting, professional appearance, and handles 
all markdown features correctly including tables, code blocks, and lists.

Quick Pandoc command with styling:
pandoc section-Obejero-Llatuna.md -o section-Obejero-Llatuna.pdf \
  --pdf-engine=pdflatex \
  --toc \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable documentclass=report

================================================================================
VERIFICATION
================================================================================

After conversion, verify the PDF includes:
✓ All 11 main sections
✓ Table of contents
✓ Code blocks with proper formatting
✓ Tables rendered correctly
✓ Diagrams visible
✓ Appendices included

Total pages should be approximately 30-35 pages.

================================================================================
