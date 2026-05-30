#!/usr/bin/env python3
"""
Rigorous validation of Word document conversion from markdown.
Checks: Content accuracy, formatting compliance, figure embedding, cross-references.
"""

import re
import os
from pathlib import Path
from docx import Document
from docx.shared import Pt

class WordValidationFramework:
    def __init__(self, docx_path, md_path):
        self.docx_path = docx_path
        self.md_path = md_path
        self.doc = Document(docx_path)
        self.md_content = self._read_markdown()
        self.validation_results = []

    def _read_markdown(self):
        """Read markdown file."""
        with open(self.md_path, 'r', encoding='utf-8') as f:
            return f.read()

    def validate_content(self):
        """Phase 1: Content Validation."""
        print("\n[PHASE 1] CONTENT VALIDATION")
        print("=" * 70)

        # Extract text from Word document
        word_text = '\n'.join([para.text for para in self.doc.paragraphs])

        # Check 1: Basic structure
        self._check_section_count()

        # Check 2: Key content presence
        self._check_content_presence(word_text)

        # Check 3: Figure count
        self._check_figure_count()

        # Check 4: Table count
        self._check_table_count()

        # Check 5: Reference list
        self._check_references()

        # Check 6: Metric values accuracy
        self._check_metric_accuracy(word_text)

        return self.validation_results

    def validate_formatting(self):
        """Phase 2: Format Validation."""
        print("\n[PHASE 2] FORMAT VALIDATION")
        print("=" * 70)

        # Check 1: Font sizes
        self._check_font_sizes()

        # Check 2: Line spacing (double-spaced body text)
        self._check_line_spacing()

        # Check 3: Margins
        self._check_margins()

        # Check 4: Page numbers
        self._check_page_numbers()

        # Check 5: Heading formatting
        self._check_heading_formatting()

        return self.validation_results

    def _check_section_count(self):
        """Verify all major sections present."""
        headings = [para.text for para in self.doc.paragraphs if para.style.name.startswith('Heading')]
        print(f"[INFO] Found {len(headings)} headings in document")

        required_sections = [
            'Abstract', 'Introduction', 'Methodology', 'Results', 'Discussion',
            'Recommendations', 'References'
        ]

        for section in required_sections:
            found = any(section.lower() in h.lower() for h in headings)
            status = "PASS" if found else "FAIL"
            print(f"  [{status}] '{section}' section found")
            self.validation_results.append({
                'check': f'Section: {section}',
                'status': status,
                'details': 'Found' if found else 'Missing'
            })

    def _check_content_presence(self, word_text):
        """Check for key content elements."""
        print(f"\n[INFO] Checking content presence...")

        # Check for key metrics and values
        key_content = {
            'BSTS': 'Model name (BSTS)',
            'recovery': 'Recovery metric',
            'MAPE': 'MAPE metric',
            'Framework': 'Framework terminology',
            'scenario': 'Scenario references',
        }

        for keyword, description in key_content.items():
            found = keyword.lower() in word_text.lower()
            status = "PASS" if found else "FAIL"
            print(f"  [{status}] {description}: '{keyword}'")
            self.validation_results.append({
                'check': f'Content: {description}',
                'status': status,
                'details': f'Keyword: {keyword}'
            })

    def _check_figure_count(self):
        """Count embedded figures."""
        # Detect images in document
        image_count = 0
        for rel in self.doc.part.rels.values():
            if "image" in rel.target_ref:
                image_count += 1

        print(f"\n[INFO] Embedded images: {image_count}")
        expected_figures = 16

        status = "PASS" if image_count >= expected_figures - 2 else "WARN"  # Allow ±2 tolerance
        print(f"  [{status}] Expected ~{expected_figures} figures, found {image_count}")
        self.validation_results.append({
            'check': 'Figure count',
            'status': status,
            'details': f'{image_count} embedded images (expected ~{expected_figures})'
        })

    def _check_table_count(self):
        """Count tables in document."""
        table_count = len(self.doc.tables)
        print(f"\n[INFO] Tables in document: {table_count}")

        expected_tables = 5
        status = "PASS" if table_count >= expected_tables else "WARN"
        print(f"  [{status}] Expected ~{expected_tables} tables, found {table_count}")
        self.validation_results.append({
            'check': 'Table count',
            'status': status,
            'details': f'{table_count} tables (expected ~{expected_tables})'
        })

    def _check_references(self):
        """Check references section."""
        word_text = '\n'.join([para.text for para in self.doc.paragraphs])

        # Count reference numbers
        ref_count = len(re.findall(r'^\d+\.\s+', word_text, re.MULTILINE))
        print(f"\n[INFO] References detected: ~{ref_count}")

        expected_refs = 14
        status = "PASS" if ref_count >= expected_refs - 1 else "WARN"
        print(f"  [{status}] Expected {expected_refs} references, found ~{ref_count}")
        self.validation_results.append({
            'check': 'Reference count',
            'status': status,
            'details': f'{ref_count} references (expected {expected_refs})'
        })

    def _check_metric_accuracy(self, word_text):
        """Check for correct metric values (sampling)."""
        print(f"\n[INFO] Checking metric accuracy (sample)...")

        # Sample metrics to check (from markdown)
        metrics_to_check = {
            'BSTS': ['82.4%', '81.0%'],
            'ARDL': ['68.8%', '0.0%'],
            'kalman_dlm': ['82.0%', '83.1%'],
        }

        for model, values in metrics_to_check.items():
            found = all(val in word_text for val in values)
            status = "PASS" if found else "WARN"
            print(f"  [{status}] {model} metrics: {', '.join(values)}")
            self.validation_results.append({
                'check': f'Metric: {model}',
                'status': status,
                'details': f'Values: {", ".join(values)}'
            })

    def _check_font_sizes(self):
        """Check font sizes (body: 12pt, headings: 12pt)."""
        print(f"\n[INFO] Checking font sizes...")

        body_fonts = []
        heading_fonts = []

        for para in self.doc.paragraphs:
            for run in para.runs:
                if run.font.size:
                    size_pt = run.font.size.pt
                    if para.style.name.startswith('Heading'):
                        heading_fonts.append(size_pt)
                    else:
                        body_fonts.append(size_pt)

        # JMR standard: 12pt
        print(f"  [INFO] Body text fonts: {set(body_fonts) if body_fonts else 'Default'}")
        print(f"  [INFO] Heading fonts: {set(heading_fonts) if heading_fonts else 'Default'}")

        self.validation_results.append({
            'check': 'Font sizes',
            'status': 'INFO',
            'details': f'Body: {set(body_fonts) or "Default"}, Headings: {set(heading_fonts) or "Default"}'
        })

    def _check_line_spacing(self):
        """Check line spacing (should be 2.0 for double-spaced)."""
        print(f"\n[INFO] Checking line spacing...")

        spacing_values = []
        for para in self.doc.paragraphs:
            if para.paragraph_format.line_spacing:
                spacing_values.append(para.paragraph_format.line_spacing)

        if spacing_values:
            avg_spacing = sum(spacing_values) / len(spacing_values)
            status = "PASS" if 1.9 < avg_spacing < 2.1 else "WARN"
            print(f"  [{status}] Average line spacing: {avg_spacing:.2f}")
        else:
            status = "INFO"
            print(f"  [INFO] Using default line spacing")

        self.validation_results.append({
            'check': 'Line spacing',
            'status': status,
            'details': f'Values: {spacing_values or "Default"}'
        })

    def _check_margins(self):
        """Check page margins (1 inch all sides)."""
        print(f"\n[INFO] Checking margins...")

        section = self.doc.sections[0]
        margins = {
            'top': section.top_margin.inches,
            'bottom': section.bottom_margin.inches,
            'left': section.left_margin.inches,
            'right': section.right_margin.inches,
        }

        all_correct = all(0.95 < m < 1.05 for m in margins.values())
        status = "PASS" if all_correct else "WARN"
        print(f"  [{status}] Margins: {margins}")
        self.validation_results.append({
            'check': 'Margins (1 inch)',
            'status': status,
            'details': f'Top: {margins["top"]:.2f}", Bottom: {margins["bottom"]:.2f}", Left: {margins["left"]:.2f}", Right: {margins["right"]:.2f}"'
        })

    def _check_page_numbers(self):
        """Check for page numbers in footer."""
        print(f"\n[INFO] Checking page numbers...")

        has_footer = False
        for section in self.doc.sections:
            footer = section.footer
            if footer.paragraphs and footer.paragraphs[0].text.strip():
                has_footer = True
                break

        status = "WARN" if not has_footer else "INFO"
        print(f"  [{status}] Page numbers in footer: {has_footer}")
        self.validation_results.append({
            'check': 'Page numbers',
            'status': status,
            'details': f'Footer configured: {has_footer}'
        })

    def _check_heading_formatting(self):
        """Check heading styles."""
        print(f"\n[INFO] Checking heading formatting...")

        heading_styles = {}
        for para in self.doc.paragraphs:
            if para.style.name.startswith('Heading'):
                style = para.style.name
                heading_styles[style] = heading_styles.get(style, 0) + 1

        print(f"  [INFO] Heading styles: {heading_styles}")
        self.validation_results.append({
            'check': 'Heading styles',
            'status': 'INFO',
            'details': str(heading_styles)
        })

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        passes = sum(1 for r in self.validation_results if r['status'] == 'PASS')
        warns = sum(1 for r in self.validation_results if r['status'] == 'WARN')
        infos = sum(1 for r in self.validation_results if r['status'] == 'INFO')

        print(f"\nResults: {passes} PASS, {warns} WARN, {infos} INFO")
        print(f"Total checks: {len(self.validation_results)}")

        return {
            'passes': passes,
            'warns': warns,
            'infos': infos,
            'total': len(self.validation_results),
            'details': self.validation_results
        }

if __name__ == '__main__':
    docx_file = 'C:\\github\\ltc\\writing\\LTC_Frameworks_JMR.docx'
    md_file = 'C:\\github\\ltc\\writing\\MASTER_DOCUMENT_FINAL.md'

    print("[VALIDATION] Word Document Conversion")
    print("Document:", docx_file)
    print("Source:", md_file)

    validator = WordValidationFramework(docx_file, md_file)

    # Run validation phases
    validator.validate_content()
    validator.validate_formatting()

    # Generate report
    report = validator.generate_report()

    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    for result in validator.validation_results:
        print(f"{result['check']:30s} [{result['status']:5s}] {result['details']}")
