# Streamlit app and portfolio integration

The app presents the GraphRAG pipeline as an interactive engineering demonstration. Its default corpus contains 11 synthetic documents and three example questions. It uses deterministic embeddings and heuristic answer extraction, with the CGC Labs visual identity.

## Run locally

Use Python 3.11. From the repository root, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the address printed by Streamlit. Ask a provided question, compare graph-and-vector retrieval with dense retrieval, and inspect the returned paths and passages. Run the benchmark separately to evaluate all three examples.

No model download, API key, or private dataset is required. The app accepts questions about its fixed demonstration corpus. Its small synthetic benchmark does not establish general question-answering accuracy.

## Runtime behavior

- Questions contain at most 500 characters on one line.
- Each operation builds private temporary stores and closes them before cleanup.
- At most two operations run concurrently. A request waits up to two seconds for a slot.
- Each operation uses a 64 MiB Kuzu buffer pool and two query threads. The buffer size is not total app memory.
- Query results are cached for five minutes, with at most 32 query entries and one benchmark entry. Cached values contain data, not database handles.
- Brand fonts load through official Fontshare stylesheets. System sans fonts remain available offline.

`constraints.txt` records the tested direct dependency versions. The Kuzu and LanceDB minimums support the explicit cleanup APIs used by the demo.

## Deploy to Streamlit Community Cloud

After the reviewed code is published to GitHub:

1. Open Streamlit Community Cloud and create an app.
2. Select the repository and branch containing this project.
3. Set the entry point to `streamlit_app.py`.
4. Select Python 3.11 in Advanced settings.
5. Deploy the app and record its assigned public URL.

Community Cloud takes the repository, branch, and entry point from the deployment form. Its Python selector is in Advanced settings. [Deployment reference](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Keep `requirements.txt` at the repository root beside the entry point. It defines the app's Python dependencies. [Dependency reference](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

The demo needs no secrets. Keep local data stores and environment files out of Git. Treat host-local storage as disposable; the included corpus can be rebuilt after a restart. A hosted app may need time to start before its content appears.

No public deployment URL is configured in this repository. The instructions above do not mean the app has been published.

## Add it to the existing portfolio

The portfolio source uses `components/projects.tsx` for project entries. Its existing `components/streamlit-preview.tsx` component already provides the app dialog, direct link, and close behavior.

Use that component after the public app works. It accepts HTTPS URLs on `*.streamlit.app`, rejects credentials and custom ports, and adds `embed=true` and `embed_options=light_theme`. Streamlit documents these parameters for public apps; private app embedding is not officially supported. [Embedding reference](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app).

The following entry follows the portfolio's existing supporting-project structure. `HOSTED_APP_URL` denotes the actual assigned URL. Define it with the verified deployment URL before adding this entry; it is not a supplied or working address.

```tsx
<article className="supporting-tile">
  <header className="supporting-tile-header">
    <span className="supporting-index" aria-hidden="true">07</span>
    <p className="eyebrow">Retrieval engineering</p>
  </header>
  <h4>Multi-Hop GraphRAG</h4>
  <p>
    I combined graph traversal and vector search, then compared their answers
    with dense retrieval on a small synthetic corpus. The app exposes the
    paths and passages behind each result.
  </p>
  <div className="supporting-tile-actions">
    <StreamlitPreview
      title="Multi-Hop GraphRAG"
      url={HOSTED_APP_URL}
      note="Explore 11 synthetic documents and three example questions. These results are a pipeline demonstration, not a MuSiQue validation score."
    />
  </div>
</article>
```

Place the entry inside the existing `supporting-grid`. `StreamlitPreview` is already imported there. The local project number may need adjustment if other entries have been added.

The current section introduction says the projects use public data. Change that sentence to “I built these independent projects” when adding this synthetic demonstration.

Keep the component's lazy dialog behavior: opening it mounts the iframe; closing it removes the iframe. The portfolio page should not start every Streamlit app on initial load. Its existing direct-app link also gives visitors a way to open the hosted app separately.

## Verify the hosted handoff

- Open the assigned public app URL and run a provided question.
- Confirm that answers and retrieved passages render.
- Run the benchmark and inspect its exports.
- Open the app through the portfolio dialog at desktop and mobile widths.
- Confirm that Close and Escape remove the iframe and restore focus.
- Confirm that the direct app link works independently of the dialog.

Local app checks establish local behavior. The hosted app and the deployed portfolio require their own checks after publication.

Cloud deployment, Linux execution, and the live portfolio embed have not been verified. The local Windows app, question and benchmark downloads, and responsive layouts have been exercised.
