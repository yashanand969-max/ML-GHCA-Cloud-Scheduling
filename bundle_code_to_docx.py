"""
Generates ML_GHCA_Source_Code_Bundle.docx:
A publication-quality source code appendix compiling all Python modules,
scripts, and configurations from the ML-GHCA Cloud & FMS Scheduling project.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

OUTPUT_FILE = "ML_GHCA_Source_Code_Bundle.docx"

# File manifest grouped into logical categories
FILES_MANIFEST = [
    {
        "category": "1. Ground-Truth FMS Benchmark & Simulation Engine",
        "files": [
            {
                "path": "src/core/scheduler.py",
                "name": "scheduler.py",
                "desc": "Ulusoy et al. (1997) compliant benchmark generator, discrete-event state tracking, Operational Completion Time (OCT), Machine Load Balance (LB), and multi-objective combined cost function.",
            }
        ]
    },
    {
        "category": "2. Metaheuristic Optimization Core",
        "files": [
            {
                "path": "src/core/hill_climbing.py",
                "name": "hill_climbing.py",
                "desc": "Local search algorithm with 1,000 iterations exploring the 2-swap operation neighborhood to eliminate initial sequencing bottlenecks and warm-start the genetic population.",
            },
            {
                "path": "src/core/genetic_algorithm.py",
                "name": "genetic_algorithm.py",
                "desc": "Standard Genetic Hill-Climbing Algorithm (GHCA, 200 generations) and Surrogate-Accelerated Genetic Algorithm (1,200+ generations) with tournament selection, order crossover (OX), swap mutations, elitism, and periodic true-cost calibration.",
            }
        ]
    },
    {
        "category": "3. Upstream Machine Learning Prioritization (The Ablation Baseline)",
        "files": [
            {
                "path": "src/ml/ml_layer.py",
                "name": "ml_layer.py",
                "desc": "6-feature tabular operation extractor, Linear Regression duration predictor, and pre-search operation ordering heuristic (ml_sort and ml_ghca).",
            }
        ]
    },
    {
        "category": "4. 40-Problem Dual-Scale Benchmark Suite & Statistical Analysis",
        "files": [
            {
                "path": "src/experiments/main.py",
                "name": "main.py",
                "desc": "Main evaluation harness executing Baseline, GHCA, Heuristic-GHCA, and ML-GHCA across all 40 dual-scale Ulusoy benchmark problems.",
            },
            {
                "path": "src/experiments/dual_scale_analysis.py",
                "name": "dual_scale_analysis.py",
                "desc": "Decoupled evaluation isolating problem scale (Small Regime: 8 parts vs Large Regime: 18 parts), computing descriptive metrics, Shapiro-Wilk normality tests, paired Wilcoxon signed-rank tests, and rank-biserial effect sizes.",
            },
            {
                "path": "src/experiments/anova_test.py",
                "name": "anova_test.py",
                "desc": "Omnibus Friedman non-parametric test, Two-Way ANOVA variance partitioning (Algorithms vs Problems), and regime-decoupled Wilcoxon hypothesis tests with honest-failure reporting.",
            },
            {
                "path": "src/experiments/weight_sensitivity.py",
                "name": "weight_sensitivity.py",
                "desc": "Hyperparameter sensitivity testing proving method ranking stability across alternative cost weight configurations (0.5/0.5, 0.7/0.3, and 0.9/0.1).",
            }
        ]
    },
    {
        "category": "5. Model Capacity Ablation (Non-Linear Tree Ensemble)",
        "files": [
            {
                "path": "src/ml/phase5_gbdt_ablation.py",
                "name": "phase5_gbdt_ablation.py",
                "desc": "5-fold cross-validation grid search for Gradient Boosted Decision Trees (LightGBM) and 40-problem dual-scale ablation testing whether model capacity improves schedule quality over linear regression.",
            }
        ]
    },
    {
        "category": "6. Microsecond Surrogate Fitness Acceleration Framework",
        "files": [
            {
                "path": "src/ml/surrogate_data_generator.py",
                "name": "surrogate_data_generator.py",
                "desc": "Synthesizes 51,000 sequence permutations across 300 Ulusoy benchmark instances, extracts 28 order-dependent scalar features (71.18 µs extraction speed), and exports problem-grouped train/test splits.",
            },
            {
                "path": "src/ml/surrogate_model.py",
                "name": "surrogate_model.py",
                "desc": "Trains and benchmarks Ridge Regression (alpha=1.0), LightGBM, and MLP regressors; computes Spearman rank correlation (rho = 0.9749) and microsecond inference latency (6.44 µs); serializes model and scaler artifacts.",
            },
            {
                "path": "src/experiments/surrogate_benchmark.py",
                "name": "surrogate_benchmark.py",
                "desc": "Full 40-problem dual-scale benchmark comparing Baseline, GHCA, Heuristic-GHCA, Linear-ML-GHCA, and Surrogate-GHCA (1,200 generations) with paired Wilcoxon signed-rank tests and rank-biserial effect sizes.",
            },
            {
                "path": "src/experiments/surrogate_ablation.py",
                "name": "surrogate_ablation.py",
                "desc": "Ablation experiments evaluating GA generation budget scaling (200 to 1,500 generations), true-cost re-calibration interval sensitivity (25 to 200 generations), and objective weight stability.",
            }
        ]
    },
    {
        "category": "7. Publication Visualizations & Plotting Infrastructure",
        "files": [
            {
                "path": "src/visualization/plot_results.py",
                "name": "plot_results.py",
                "desc": "Generates 300 DPI publication figures: Graph 1 (OCT comparison), Graph 2 (Load balance), Graph 3 (Combined cost), and Graph 4 (Improvement vs GHCA).",
            },
            {
                "path": "src/visualization/plot_convergence.py",
                "name": "plot_convergence.py",
                "desc": "Generates Graph 5 illustrating generation-by-generation evolutionary convergence curves from Hill Climbing seed to Pareto plateau.",
            },
            {
                "path": "src/visualization/plot_dual_scale.py",
                "name": "plot_dual_scale.py",
                "desc": "Generates Graph 6 (Dual-scale KDE distribution overlays) and Graph 7 (Regime comparison boxplots).",
            },
            {
                "path": "src/visualization/plot_phase5_ablation.py",
                "name": "plot_phase5_ablation.py",
                "desc": "Generates Graph 8 (Model capacity actual vs predicted fit parity plots) and Graph 9 (Schedule cost comparisons across regressor capacities).",
            },
            {
                "path": "src/visualization/plot_surrogate_results.py",
                "name": "plot_surrogate_results.py",
                "desc": "Generates Graph 11 (Standard vs Surrogate GA wall-clock convergence), Graph 12 (40-problem surrogate benchmark comparison), and Graph 13 (Paired difference bar chart).",
            }
        ]
    },
    {
        "category": "8. Configuration, Dependencies & Architectural Blueprints",
        "files": [
            {
                "path": "requirements.txt",
                "name": "requirements.txt",
                "desc": "Pinned Python package dependencies for exact reproduction.",
            },
            {
                "path": "docs/surrogate_implementation_plan.py",
                "name": "docs/surrogate_implementation_plan.py",
                "desc": "Phase-wise implementation plan, dataset migration architecture, and engineering blueprints for the surrogate fitness framework.",
            }
        ]
    }
]


def set_cell_background(cell, color_hex):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding of a table cell (in twips: 20 twips = 1 pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def add_callout_box(doc, title, subtitle=None, text_lines=None, border_color="1976D2", bg_color="F0F4F8"):
    """Adds a callout box table with custom background and left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Set custom borders: thick left border, no top/bottom/right border
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(tc_borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_t = p.add_run(title)
    run_t.font.name = "Arial"
    run_t.font.size = Pt(11)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(15, 23, 42)

    if subtitle:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(2)
        p2.paragraph_format.line_spacing = 1.15
        run_s = p2.add_run(subtitle)
        run_s.font.name = "Arial"
        run_s.font.size = Pt(9.5)
        run_s.font.italic = True
        run_s.font.color.rgb = RGBColor(71, 85, 105)

    if text_lines:
        for line in text_lines:
            pl = cell.add_paragraph()
            pl.paragraph_format.space_before = Pt(1)
            pl.paragraph_format.space_after = Pt(1)
            pl.paragraph_format.line_spacing = 1.15
            run_l = pl.add_run(line)
            run_l.font.name = "Arial"
            run_l.font.size = Pt(9.5)
            run_l.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def create_code_bundle():
    print("=" * 80)
    print("INITIALIZING DOCUMENT GENERATION: ML_GHCA_Source_Code_Bundle.docx")
    print("=" * 80)

    doc = docx.Document()

    # Set page setup: Standard Letter, 0.8 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False

        # Add clean header & footer
        hdr_p = section.header.paragraphs[0]
        hdr_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hdr_run = hdr_p.add_run("ML-GHCA Scheduling Framework | Source Code Appendix")
        hdr_run.font.name = "Arial"
        hdr_run.font.size = Pt(8)
        hdr_run.font.color.rgb = RGBColor(148, 163, 184)

        ftr_p = section.footer.paragraphs[0]
        ftr_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ftr_run = ftr_p.add_run("Manipal University Jaipur — Dept of Data Science & Engineering | DSE2270")
        ftr_run.font.name = "Arial"
        ftr_run.font.size = Pt(8)
        ftr_run.font.color.rgb = RGBColor(148, 163, 184)

    # Base styles setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)

    # Custom Code Style
    styles = doc.styles
    code_style_name = 'CodeListingStyle'
    if code_style_name in styles:
        code_style = styles[code_style_name]
    else:
        code_style = styles.add_style(code_style_name, docx.enum.style.WD_STYLE_TYPE.PARAGRAPH)
    code_style.font.name = 'Consolas'
    code_style.font.size = Pt(8.0)
    code_style.paragraph_format.line_spacing = 1.0
    code_style.paragraph_format.space_before = Pt(0)
    code_style.paragraph_format.space_after = Pt(0)
    code_style.paragraph_format.left_indent = Inches(0.15)

    # ------------------------------------------------------------
    # COVER PAGE / TITLE SECTION
    # ------------------------------------------------------------
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(30)
    p_meta.paragraph_format.space_after = Pt(4)
    r_meta = p_meta.add_run("PROJECT BASED LEARNING - 2 (DSE2270) • SOURCE CODE APPENDIX")
    r_meta.font.name = "Arial"
    r_meta.font.size = Pt(10)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(30, 64, 175)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(8)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("ML-GHCA: Cloud and Flexible Manufacturing System (FMS) Scheduling Optimization")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(24)
    r_sub = p_sub.add_run("Complete Source Code Repository, Algorithmic Implementations, Statistical Engines & Surrogate Framework")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Metadata callout
    add_callout_box(
        doc,
        title="ACADEMIC PROJECT INFORMATION",
        subtitle="Department of Data Science and Engineering | Manipal University Jaipur",
        text_lines=[
            "• Course: Project Based Learning - 2 (DSE2270) | B.Tech CSE (Data Science)",
            "• Student Authors:",
            "    1. Annareddy Sai Prathima (Reg No: 2430010176)",
            "    2. Samiksha Saini (Reg No: 2430010177)",
            "    3. Yash Anand (Reg No: 2430010183)",
            "    4. Akshat Jha (Reg No: 2430010186)",
            "• Date: October 2026 | Academic Year: 2026–2027",
            "• Repository: ML-GHCA-Cloud-Scheduling",
            "• Base Research Grounding: Alla et al. (Cogent Engineering, 2024) & Ulusoy et al. (1997)",
        ],
        border_color="2563EB",
        bg_color="F8FAFC"
    )

    # ------------------------------------------------------------
    # MANIFEST / FILE CATALOGUE TABLE
    # ------------------------------------------------------------
    doc.add_paragraph().paragraph_format.space_before = Pt(12)
    p_h = doc.add_paragraph()
    r_h = p_h.add_run("Source Code Inventory & Module Directory")
    r_h.font.name = "Arial"
    r_h.font.size = Pt(14)
    r_h.font.bold = True
    r_h.font.color.rgb = RGBColor(15, 23, 42)

    p_desc = doc.add_paragraph()
    p_desc.paragraph_format.space_after = Pt(10)
    p_desc.add_run("The table below catalogs all 19 core Python modules, configuration scripts, and documentation blueprints contained in this appendix. Total source volume exceeds 4,800 lines of documented, verified scientific code.")

    # Manifest Table
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(1.8), Inches(2.2), Inches(0.9), Inches(1.9)]
    hdr_cells = table.rows[0].cells
    hdr_titles = ["Module File", "Category", "Size / LOC", "Primary Responsibility"]

    for idx, (cell, title, w) in enumerate(zip(hdr_cells, hdr_titles, col_widths)):
        cell.width = w
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx == 2 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(title)
        run.font.name = "Arial"
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    # Collect stats and populate table
    total_loc = 0
    total_bytes = 0

    for cat_data in FILES_MANIFEST:
        cat_name = cat_data["category"]
        for f_data in cat_data["files"]:
            fpath = f_data["path"]
            fname = f_data["name"]
            fdesc = f_data["desc"]

            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    lines = fp.readlines()
                loc = len(lines)
                sz_kb = round(os.path.getsize(fpath) / 1024, 1)
                total_loc += loc
                total_bytes += os.path.getsize(fpath)
            else:
                loc = 0
                sz_kb = 0.0

            row = table.add_row()
            cells = row.cells
            for idx, (cell, w) in enumerate(zip(cells, col_widths)):
                cell.width = w
                set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
                # Alternate shading
                if len(table.rows) % 2 == 0:
                    set_cell_background(cell, "F8FAFC")
                else:
                    set_cell_background(cell, "FFFFFF")

            # Col 0: File Name
            p0 = cells[0].paragraphs[0]
            p0.paragraph_format.space_before = Pt(1)
            p0.paragraph_format.space_after = Pt(1)
            r0 = p0.add_run(fname)
            r0.font.name = "Consolas"
            r0.font.size = Pt(8.5)
            r0.font.bold = True
            r0.font.color.rgb = RGBColor(30, 64, 175)

            # Col 1: Category
            p1 = cells[1].paragraphs[0]
            p1.paragraph_format.space_before = Pt(1)
            p1.paragraph_format.space_after = Pt(1)
            r1 = p1.add_run(cat_name.split(". ")[-1])
            r1.font.name = "Arial"
            r1.font.size = Pt(8)
            r1.font.color.rgb = RGBColor(71, 85, 105)

            # Col 2: Size / LOC
            p2 = cells[2].paragraphs[0]
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p2.paragraph_format.space_before = Pt(1)
            p2.paragraph_format.space_after = Pt(1)
            r2 = p2.add_run(f"{loc} L\n({sz_kb} KB)")
            r2.font.name = "Arial"
            r2.font.size = Pt(8)
            r2.font.color.rgb = RGBColor(30, 41, 59)

            # Col 3: Desc
            p3 = cells[3].paragraphs[0]
            p3.paragraph_format.space_before = Pt(1)
            p3.paragraph_format.space_after = Pt(1)
            r3 = p3.add_run(fdesc[:95] + "..." if len(fdesc) > 95 else fdesc)
            r3.font.name = "Arial"
            r3.font.size = Pt(7.5)
            r3.font.color.rgb = RGBColor(51, 65, 85)

    # Summary row
    row_sum = table.add_row()
    for idx, (cell, w) in enumerate(zip(row_sum.cells, col_widths)):
        cell.width = w
        set_cell_background(cell, "E2E8F0")
        set_cell_margins(cell, top=100, bottom=100, left=90, right=90)
    p_s0 = row_sum.cells[0].paragraphs[0]
    r_s0 = p_s0.add_run("TOTALS (19 Files)")
    r_s0.font.name = "Arial"
    r_s0.font.size = Pt(8.5)
    r_s0.font.bold = True
    r_s0.font.color.rgb = RGBColor(15, 23, 42)

    p_s2 = row_sum.cells[2].paragraphs[0]
    p_s2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_s2 = p_s2.add_run(f"{total_loc:,} Lines\n({round(total_bytes/1024, 1)} KB)")
    r_s2.font.name = "Arial"
    r_s2.font.size = Pt(8.5)
    r_s2.font.bold = True
    r_s2.font.color.rgb = RGBColor(15, 23, 42)

    doc.add_page_break()

    # ------------------------------------------------------------
    # ITERATE THROUGH MODULES & APPEND FORMATTED SOURCE CODE
    # ------------------------------------------------------------
    print(f"Beginning code listing generation across {len(FILES_MANIFEST)} sections...")

    file_counter = 1
    for cat_data in FILES_MANIFEST:
        cat_title = cat_data["category"]

        # Section Header
        p_sec = doc.add_paragraph()
        p_sec.paragraph_format.space_before = Pt(16)
        p_sec.paragraph_format.space_after = Pt(8)
        p_sec.paragraph_format.keep_with_next = True
        r_sec = p_sec.add_run(f"SECTION {cat_title.upper()}")
        r_sec.font.name = "Arial"
        r_sec.font.size = Pt(14)
        r_sec.font.bold = True
        r_sec.font.color.rgb = RGBColor(30, 58, 138)

        for f_data in cat_data["files"]:
            fpath = f_data["path"]
            fname = f_data["name"]
            fdesc = f_data["desc"]

            print(f"[{file_counter}/19] Formatting {fname} ({fpath})...")

            if not os.path.exists(fpath):
                print(f"WARNING: File not found: {fpath}")
                continue

            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                file_lines = fp.readlines()

            loc_count = len(file_lines)
            kb_size = round(os.path.getsize(fpath) / 1024, 2)

            # Sub-heading
            p_subh = doc.add_paragraph()
            p_subh.paragraph_format.space_before = Pt(14)
            p_subh.paragraph_format.space_after = Pt(2)
            p_subh.paragraph_format.keep_with_next = True
            r_num = p_subh.add_run(f"Module {file_counter}: ")
            r_num.font.name = "Arial"
            r_num.font.size = Pt(11)
            r_num.font.bold = True
            r_num.font.color.rgb = RGBColor(71, 85, 105)

            r_fn = p_subh.add_run(fname)
            r_fn.font.name = "Consolas"
            r_fn.font.size = Pt(11)
            r_fn.font.bold = True
            r_fn.font.color.rgb = RGBColor(15, 23, 42)

            # Info badge box
            add_callout_box(
                doc,
                title=f"Source Specification: {fname}",
                subtitle=f"Path: {fpath}  |  Length: {loc_count} lines  |  Size: {kb_size} KB",
                text_lines=[fdesc],
                border_color="64748B",
                bg_color="F1F5F9"
            )

            # Code container header
            p_code_hdr = doc.add_paragraph()
            p_code_hdr.paragraph_format.space_before = Pt(4)
            p_code_hdr.paragraph_format.space_after = Pt(4)
            p_code_hdr.paragraph_format.keep_with_next = True
            r_ch = p_code_hdr.add_run(f"--- BEGIN CODE LISTING: {fname} ({loc_count} LINES) ---")
            r_ch.font.name = "Consolas"
            r_ch.font.size = Pt(8.5)
            r_ch.font.bold = True
            r_ch.font.color.rgb = RGBColor(100, 116, 139)

            # Add lines
            for line_idx, line_raw in enumerate(file_lines, 1):
                clean_line = line_raw.rstrip("\r\n")

                p_line = doc.add_paragraph(style=code_style_name)

                # Line number run (muted gray)
                r_ln = p_line.add_run(f"{line_idx:04d}  ")
                r_ln.font.name = "Consolas"
                r_ln.font.size = Pt(7.5)
                r_ln.font.color.rgb = RGBColor(148, 163, 184)

                # Code content run
                r_code = p_line.add_run(clean_line)
                r_code.font.name = "Consolas"
                r_code.font.size = Pt(8.0)

                # Subtle syntax coloring for comments vs regular code
                stripped = clean_line.strip()
                if stripped.startswith("#"):
                    r_code.font.color.rgb = RGBColor(100, 116, 139)
                    r_code.font.italic = True
                elif stripped.startswith(('def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'else:', 'elif ', 'for ', 'while ')):
                    r_code.font.color.rgb = RGBColor(15, 23, 42)
                    r_code.font.bold = True
                else:
                    r_code.font.color.rgb = RGBColor(30, 41, 59)

            # Code container footer
            p_code_ftr = doc.add_paragraph()
            p_code_ftr.paragraph_format.space_before = Pt(4)
            p_code_ftr.paragraph_format.space_after = Pt(16)
            r_cf = p_code_ftr.add_run(f"--- END CODE LISTING: {fname} ---")
            r_cf.font.name = "Consolas"
            r_cf.font.size = Pt(8.5)
            r_cf.font.bold = True
            r_cf.font.color.rgb = RGBColor(100, 116, 139)

            file_counter += 1
            doc.add_page_break()

    # Save document
    print("=" * 80)
    print(f"SAVING COMPLETE DOCUMENT TO {OUTPUT_FILE}...")
    doc.save(OUTPUT_FILE)
    sz_mb = round(os.path.getsize(OUTPUT_FILE) / (1024 * 1024), 2)
    print(f"SUCCESS: Document created successfully ({sz_mb} MB) at {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 80)


if __name__ == "__main__":
    create_code_bundle()
