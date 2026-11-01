# Verification

Checked locally on 30 October 2026 with Python 3.11.9.

The full test suite was rerun on 31 October 2026 before GitHub publication: 111 passed. The browser checks below were recorded on 30 October.

| Check | Observed result |
| :--- | :--- |
| Full test suite | 111 passed, including six Streamlit AppTest flows. |
| Interactive app | Initial and custom questions, third-example selection, benchmark, architecture view, blank input, and backend failure checked. |
| Streamlit browser checks | Ask view fits at 320, 390, 768, and 1440 pixels; mobile benchmark checked. Query and benchmark JSON downloads contain actual results. |
| Deployment dependencies | Pip dry-run resolves `requirements.txt`; no dependencies were changed by that check. |
| Built-in synthetic run | Three questions evaluated; GraphRAG exact match and token F1 both 100%. |
| Responsive layout | Checked at 320, 390, 768, and 1440 pixels in headless Chrome. |
| Spacing and overflow | Page content and comparison columns fit at all four widths. Titles, answers, and evidence IDs remain within their containers. |
| Query interaction | Mouse and Enter expand and collapse the query details. |
| JSON export | Downloaded data matches the complete run envelope. |
| Browser errors | No JavaScript errors observed during these checks. |
| CGC Labs identity | Live reference checked. Report display and body fonts, canvas, and ink match the site. Original logo bytes and proportions are preserved. |
| Offline fallback | At 390 pixels, embedded data and logo remain available; query expansion works with system sans fonts. |
| Packaging | Wheel builds; a project-local installation imports and renders the report. Final wheel includes the current template. |
| Documentation | Relative links resolve; the JSON schema matches the runtime report model. |

The rendering checks cover the layout and interactions listed above. They do not constitute a full accessibility audit. Screen-reader behavior and other browsers were not tested.

Streamlit Cloud, Linux, and the live portfolio iframe remain unverified. See the [deployment and integration guide](STREAMLIT.md) for the hosted checks.

The published [example results](example-results.json) use hash-projection embeddings and heuristic answer extraction. No transformer model or external language-model API was used. The synthetic result does not establish MuSiQue validation or production performance.

Run the test suite with:

```bash
python -m pytest
```

The recorded run disabled unrelated global pytest plugins with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. The project does not require third-party pytest plugins.
