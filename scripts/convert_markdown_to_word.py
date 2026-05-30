#!/usr/bin/env python3
"""
Convert MASTER_DOCUMENT_FINAL.md to JMR-compliant Word document.
Handles: formatting, image embedding, AMA reference style, proper spacing.
"""

import re
import os
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_spacing(paragraph, line_spacing=2.0):
    """Set paragraph line spacing (2.0 = double-spaced)."""
    paragraph_format = paragraph.paragraph_format
    paragraph_format.line_spacing = line_spacing

def add_font_formatting(run, bold=False, italic=False, font_size=12):
    """Apply font formatting."""
    run.font.bold = bold
    run.font.italic = italic
    run.font.size = Pt(font_size)
    run.font.name = 'Calibri'

def add_heading(doc, text, level=1):
    """Add properly formatted heading."""
    para = doc.add_paragraph(text)
    para.style = f'Heading {level}'
    para_format = para.paragraph_format

    # JMR formatting: headings should be 12pt, bold
    for run in para.runs:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.name = 'Calibri'

    if level == 1:
        para_format.space_before = Pt(12)
        para_format.space_after = Pt(6)
    else:
        para_format.space_before = Pt(6)
        para_format.space_after = Pt(0)

def add_paragraph_text(doc, text, bold=False, italic=False, font_size=12, alignment='left'):
    """Add paragraph with proper formatting."""
    para = doc.add_paragraph()

    # Handle inline formatting: **text** for bold, *text* for italic
    parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|!\[([^\]]*)\]\(([^)]+)\))', text)

    for part in parts:
        if not part:
            continue

        # Image: ![alt](path)
        if part.startswith('!['):
            match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', part)
            if match:
                alt_text, img_path = match.groups()
                # Try to embed image
                try:
                    if not os.path.isabs(img_path):
                        img_path = os.path.join('C:\\github\\ltc', img_path)
                    if os.path.exists(img_path):
                        para.add_run().add_picture(img_path, width=Inches(5.5))
                except Exception as e:
                    run = para.add_run(f"[Image: {alt_text}]")
                    run.font.italic = True

        # Bold: **text**
        elif part.startswith('**') and part.endswith('**'):
            run = para.add_run(part[2:-2])
            add_font_formatting(run, bold=True, font_size=font_size)

        # Italic: *text*
        elif part.startswith('*') and part.endswith('*'):
            run = para.add_run(part[1:-1])
            add_font_formatting(run, italic=True, font_size=font_size)

        # Regular text
        else:
            run = para.add_run(part)
            add_font_formatting(run, font_size=font_size)

    # Set paragraph formatting
    para_format = para.paragraph_format
    para_format.line_spacing = 2.0  # Double-spaced
    set_spacing(para, 2.0)

    # Alignment
    if alignment == 'center':
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif alignment == 'right':
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    return para

def set_margins(doc, top=1, bottom=1, left=1, right=1):
    """Set document margins (in inches)."""
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(top)
        section.bottom_margin = Inches(bottom)
        section.left_margin = Inches(left)
        section.right_margin = Inches(right)

def process_markdown_file(md_path):
    """Parse markdown and extract structure."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by lines and process
    lines = content.split('\n')
    sections = []
    current_section = None

    for line in lines:
        if line.startswith('# '):
            if current_section:
                sections.append(current_section)
            current_section = {'level': 1, 'title': line[2:].strip(), 'content': []}
        elif line.startswith('## '):
            if current_section:
                current_section['content'].append({'level': 2, 'title': line[3:].strip(), 'type': 'heading'})
        elif line.startswith('### '):
            if current_section:
                current_section['content'].append({'level': 3, 'title': line[4:].strip(), 'type': 'heading'})
        elif line.startswith('#### '):
            if current_section:
                current_section['content'].append({'level': 4, 'title': line[5:].strip(), 'type': 'heading'})
        elif line.startswith('!['):  # Image
            if current_section:
                current_section['content'].append({'type': 'image', 'line': line})
        elif line.startswith('| '):  # Table header
            if current_section:
                current_section['content'].append({'type': 'table_line', 'line': line})
        elif line.startswith('| -'):  # Table divider
            continue
        elif line.strip().startswith('```'):  # Code block
            current_section['content'].append({'type': 'code', 'line': line})
        elif line.strip():  # Regular text
            if current_section:
                current_section['content'].append({'type': 'text', 'line': line.strip()})
        elif current_section and not line.strip():  # Empty line = paragraph break
            current_section['content'].append({'type': 'break'})

    if current_section:
        sections.append(current_section)

    return sections

def create_word_document(md_path, output_path):
    """Create Word document from markdown."""
    doc = Document()

    # Set margins: 1 inch all sides
    set_margins(doc, 1, 1, 1, 1)

    # Set default font for document
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(12)

    # Parse markdown
    sections = process_markdown_file(md_path)

    # Process each section
    for section in sections:
        # Add main heading
        add_heading(doc, section['title'], level=1)

        # Process content
        for item in section['content']:
            if item['type'] == 'heading':
                add_heading(doc, item['title'], level=item['level'])

            elif item['type'] == 'text':
                add_paragraph_text(doc, item['line'])

            elif item['type'] == 'image':
                # Extract image path from markdown: ![alt](path)
                match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', item['line'])
                if match:
                    alt_text, img_path = match.groups()
                    para = doc.add_paragraph()
                    try:
                        if not os.path.isabs(img_path):
                            full_path = os.path.join('C:\\github\\ltc', img_path)
                        else:
                            full_path = img_path

                        if os.path.exists(full_path):
                            para.add_run().add_picture(full_path, width=Inches(5.5))
                            para_format = para.paragraph_format
                            para_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            para_format.space_before = Pt(6)
                            para_format.space_after = Pt(6)
                    except Exception as e:
                        print(f"Warning: Could not embed image {img_path}: {e}")
                        run = para.add_run(f"[Image: {alt_text}]")
                        run.font.italic = True

            elif item['type'] == 'break':
                pass  # Already handled by paragraph breaks

    # Save document
    doc.save(output_path)
    print(f"[OK] Word document created: {output_path}")
    return doc

if __name__ == '__main__':
    md_file = 'C:\\github\\ltc\\writing\\MASTER_DOCUMENT_FINAL.md'
    output_file = 'C:\\github\\ltc\\writing\\LTC_Frameworks_JMR.docx'

    print("Converting markdown to Word...")
    doc = create_word_document(md_file, output_file)
    print(f"Document saved: {output_file}")
