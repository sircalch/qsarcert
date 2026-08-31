"""
Interactive HTML report dashboard generator for QSARCert.
"""

import os
import jinja2
from qsarcert.core.scoring import QSARValidationReport

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSARCert OECD Model Validation Report</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --pass-color: #10b981;
            --pass-bg: rgba(16, 185, 129, 0.15);
            --warn-color: #f59e0b;
            --warn-bg: rgba(245, 158, 11, 0.15);
            --fail-color: #ef4444;
            --fail-bg: rgba(239, 68, 68, 0.15);
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem 1rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .title-group h1 { font-size: 2rem; font-weight: 700; }
        .title-group p { color: var(--text-secondary); font-size: 0.95rem; }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1.25rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .badge-pass { background-color: var(--pass-bg); color: var(--pass-color); border: 1px solid var(--pass-color); }
        .badge-warning { background-color: var(--warn-bg); color: var(--warn-color); border: 1px solid var(--warn-color); }
        .badge-fail { background-color: var(--fail-bg); color: var(--fail-color); border: 1px solid var(--fail-color); }

        .grid-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .card-label { font-size: 0.8rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.4rem; }
        .card-value { font-size: 1.4rem; font-weight: 700; color: var(--text-primary); }
        .card-subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem; }

        .section-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; color: var(--accent-blue); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            font-size: 0.95rem;
            background-color: var(--card-bg);
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }
        th, td { padding: 0.85rem 1rem; text-align: left; border-bottom: 1px solid var(--card-border); }
        th { background-color: rgba(255, 255, 255, 0.03); color: var(--text-secondary); font-weight: 600; text-transform: uppercase; font-size: 0.85rem; }
        tr:hover td { background-color: rgba(255, 255, 255, 0.02); }

        .tag { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700; }
        .tag-pass { background-color: var(--pass-bg); color: var(--pass-color); }
        .tag-warning { background-color: var(--warn-bg); color: var(--warn-color); }
        .tag-fail { background-color: var(--fail-bg); color: var(--fail-color); }

        .box { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 0.75rem; padding: 1.25rem; margin-bottom: 2rem; }
        pre { background-color: rgba(0, 0, 0, 0.4); padding: 1rem; border-radius: 0.5rem; color: #38bdf8; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; }
        .btn-copy { background-color: #2563eb; color: white; border: none; padding: 0.4rem 0.8rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.8rem; margin-top: 0.5rem; }
        .btn-copy:hover { background-color: #1d4ed8; }

        footer { text-align: center; font-size: 0.85rem; color: var(--text-secondary); margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--card-border); }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="title-group">
                <h1>QSARCert OECD Model Validation Report</h1>
                <p>{{ report.metadata.endpoint }} &bull; {{ report.metadata.algorithm }} ({{ report.oecd_metrics.n_samples }} compounds)</p>
            </div>
            <div>
                <span class="status-badge badge-{{ report.overall_status.lower() }}">
                    {{ report.overall_status }}
                </span>
            </div>
        </header>

        <div class="grid-cards">
            <div class="card">
                <div class="card-label">External Predictivity (Q&sup2;_ext)</div>
                <div class="card-value">{{ "%.3f"|format(report.oecd_metrics.q2_ext) }}</div>
                <div class="card-subtext">R&sup2; = {{ "%.3f"|format(report.oecd_metrics.r2) }} (MAE = {{ "%.2f"|format(report.oecd_metrics.mae) }})</div>
            </div>
            <div class="card">
                <div class="card-label">Concordance (CCC)</div>
                <div class="card-value">{{ "%.3f"|format(report.oecd_metrics.ccc) }}</div>
                <div class="card-subtext">Slope k = {{ "%.3f"|format(report.oecd_metrics.k_slope) }} / k' = {{ "%.3f"|format(report.oecd_metrics.k_prime_slope) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Applicability Domain</div>
                <div class="card-value">
                    {% if report.applicability_domain %}
                    {{ "%.1f"|format(report.applicability_domain.pct_in_domain) }}%
                    {% else %}
                    N/A
                    {% endif %}
                </div>
                <div class="card-subtext">
                    {% if report.applicability_domain %}
                    Warning leverage h* = {{ "%.3f"|format(report.applicability_domain.warning_leverage) }}
                    {% else %}
                    No feature matrix provided
                    {% endif %}
                </div>
            </div>
            <div class="card">
                <div class="card-label">Y-Randomization (cR&sup2;_p)</div>
                <div class="card-value">
                    {% if report.y_randomization %}
                    {{ "%.3f"|format(report.y_randomization.cr2_p) }}
                    {% else %}
                    N/A
                    {% endif %}
                </div>
                <div class="card-subtext">
                    {% if report.y_randomization %}
                    Scrambled R&sup2; = {{ "%.3f"|format(report.y_randomization.mean_scrambled_r2) }}
                    {% else %}
                    Scrambling skipped
                    {% endif %}
                </div>
            </div>
        </div>

        <h2 class="section-title">OECD Validation Principles & Tropsha Statistical Checks</h2>
        <table>
            <thead>
                <tr>
                    <th>OECD Principle / Metric</th>
                    <th>Measured Value</th>
                    <th>Regulatory Standard</th>
                    <th>Status</th>
                    <th>Diagnostic Interpretation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Principle 4: Goodness-of-Fit (R&sup2;)</strong></td>
                    <td>R&sup2; = {{ "%.3f"|format(report.oecd_metrics.r2) }}</td>
                    <td>R&sup2; &ge; 0.60</td>
                    <td><span class="tag tag-{{ 'pass' if report.oecd_metrics.r2 >= 0.60 else 'warning' }}">{{ 'PASS' if report.oecd_metrics.r2 >= 0.60 else 'WARNING' }}</span></td>
                    <td style="color: var(--text-secondary)">Explains {{ "%.1f"|format(report.oecd_metrics.r2*100) }}% of variance in response</td>
                </tr>
                <tr>
                    <td><strong>Principle 4: External Predictivity (Q&sup2;_ext)</strong></td>
                    <td>Q&sup2;_ext = {{ "%.3f"|format(report.oecd_metrics.q2_ext) }}</td>
                    <td>Q&sup2;_ext &ge; 0.50 (Tropsha)</td>
                    <td><span class="tag tag-{{ report.oecd_metrics.status.lower() }}">{{ report.oecd_metrics.status }}</span></td>
                    <td style="color: var(--text-secondary)">{{ report.oecd_metrics.diagnostic_message }}</td>
                </tr>
                <tr>
                    <td><strong>Principle 4: Concordance Correlation (CCC)</strong></td>
                    <td>CCC = {{ "%.3f"|format(report.oecd_metrics.ccc) }}</td>
                    <td>CCC &ge; 0.85</td>
                    <td><span class="tag tag-{{ 'pass' if report.oecd_metrics.ccc >= 0.85 else 'warning' }}">{{ 'PASS' if report.oecd_metrics.ccc >= 0.85 else 'WARNING' }}</span></td>
                    <td style="color: var(--text-secondary)">Agreement between predicted and experimental scales</td>
                </tr>
                <tr>
                    <td><strong>Principle 4: Modified Roy Metric (r_m&sup2;)</strong></td>
                    <td>r_m&sup2; avg = {{ "%.3f"|format(report.oecd_metrics.r_m_average) }} (&Delta;r_m&sup2; = {{ "%.3f"|format(report.oecd_metrics.delta_r_m_2) }})</td>
                    <td>r_m&sup2; &ge; 0.50, &Delta;r_m&sup2; &le; 0.20</td>
                    <td><span class="tag tag-{{ 'pass' if report.oecd_metrics.r_m_average >= 0.50 and report.oecd_metrics.delta_r_m_2 <= 0.20 else 'warning' }}">{{ 'PASS' if report.oecd_metrics.r_m_average >= 0.50 and report.oecd_metrics.delta_r_m_2 <= 0.20 else 'WARNING' }}</span></td>
                    <td style="color: var(--text-secondary)">Symmetry and penalty-adjusted determination</td>
                </tr>
                {% if report.applicability_domain %}
                <tr>
                    <td><strong>Principle 3: Applicability Domain (Williams Plot)</strong></td>
                    <td>{{ "%.1f"|format(report.applicability_domain.pct_in_domain) }}% in-domain ({{ report.applicability_domain.n_influential_outliers }} outliers)</td>
                    <td>&ge; 85% in-domain, 0 influential outliers</td>
                    <td><span class="tag tag-{{ report.applicability_domain.status.lower() }}">{{ report.applicability_domain.status }}</span></td>
                    <td style="color: var(--text-secondary)">{{ report.applicability_domain.diagnostic_message }}</td>
                </tr>
                {% endif %}
                {% if report.y_randomization %}
                <tr>
                    <td><strong>Principle 4: Robustness (Y-Randomization)</strong></td>
                    <td>cR&sup2;_p = {{ "%.3f"|format(report.y_randomization.cr2_p) }} (Scrambled: {{ "%.3f"|format(report.y_randomization.mean_scrambled_r2) }})</td>
                    <td>cR&sup2;_p &gt; 0.50, Scrambled R&sup2; &le; 0.20</td>
                    <td><span class="tag tag-{{ report.y_randomization.status.lower() }}">{{ report.y_randomization.status }}</span></td>
                    <td style="color: var(--text-secondary)">{{ report.y_randomization.diagnostic_message }}</td>
                </tr>
                {% endif %}
                {% if report.split_leakage %}
                <tr>
                    <td><strong>Split Integrity: Data Leakage Check</strong></td>
                    <td>{{ report.split_leakage.n_exact_duplicates }} duplicates (Mean NN sim: {{ "%.3f"|format(report.split_leakage.mean_nn_similarity) }})</td>
                    <td>0 duplicates</td>
                    <td><span class="tag tag-{{ report.split_leakage.status.lower() }}">{{ report.split_leakage.status }}</span></td>
                    <td style="color: var(--text-secondary)">{{ report.split_leakage.diagnostic_message }}</td>
                </tr>
                {% endif %}
            </tbody>
        </table>

        {% if report.recommendations %}
        <div class="box" style="border-left: 4px solid var(--warn-color);">
            <h3 style="color: var(--warn-color); margin-bottom: 0.5rem;">Diagnostic Recommendations & Model Warnings</h3>
            <ul style="padding-left: 1.25rem;">
                {% for rec in report.recommendations %}
                <li style="margin-bottom: 0.25rem; color: var(--text-secondary);">{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <h2 class="section-title">Computational Methods (Publication Ready)</h2>
        <div class="box">
            <pre id="methodsSnippet">{{ methods_text }}</pre>
            <button class="btn-copy" onclick="copyToClipboard('methodsSnippet')">Copy Methods Snippet</button>
        </div>

        <h2 class="section-title">BibTeX Citation</h2>
        <div class="box">
            <pre id="bibSnippet">{{ citation_bib }}</pre>
            <button class="btn-copy" onclick="copyToClipboard('bibSnippet')">Copy BibTeX</button>
        </div>

        <footer>
            Generated automatically by <strong>QSARCert v1.0.0</strong> &bull; OECD Principles & QSAR Quality Toolkit &bull; Monreal-Hernández, 2026.
        </footer>
    </div>

    <script>
        function copyToClipboard(elementId) {
            const text = document.getElementById(elementId).innerText;
            navigator.clipboard.writeText(text).then(() => {
                alert('Copied to clipboard!');
            }).catch(err => {
                console.error('Error copying: ', err);
            });
        }
    </script>
</body>
</html>
"""


def generate_qsar_html_report(
    report: QSARValidationReport,
    output_path: str,
    methods_text: str = "",
    citation_bib: str = ""
) -> str:
    """
    Renders HTML report template and writes it to disk.
    """
    template = jinja2.Template(HTML_TEMPLATE)
    rendered = template.render(
        report=report,
        methods_text=methods_text,
        citation_bib=citation_bib
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    return output_path
