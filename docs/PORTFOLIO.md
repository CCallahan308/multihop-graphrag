# Portfolio copy

## Project card

**Multi-Hop GraphRAG**

Graph and vector retrieval for questions that cross document boundaries.

I built a Streamlit app that links entities across passages, compares graph-and-vector retrieval with dense search, and exposes the evidence behind each answer. The project includes a three-question synthetic benchmark and inspectable result exports.

**Tools:** Python, Streamlit, Kuzu, LanceDB, Pydantic, NumPy, pytest.

**Project type:** Independent engineering demonstration.

The report and project artwork follow the visual identity of my consulting firm, [CGC Labs](https://cgclabs.org/).

## Case study

### The question

How can a retrieval system answer a question when the necessary facts appear in different passages?

A company profile may identify its founder. A separate biography may identify that person's birthplace. Finding a relevant passage is only part of the problem; the system also needs to follow the connection.

### What I built

I separated the passage index from the relationship graph. LanceDB stores passage and entity vectors. Kuzu stores entities, passages, and weighted relationships between them.

At query time, the pipeline links mentions to seed entities, traverses up to three hops, and combines graph candidates with vector results using reciprocal rank fusion. A heuristic extractor ranks answer candidates by the question's terminal relationship before considering path depth.

The default configuration uses rule-based extraction and deterministic hash-projection embeddings. It runs locally without an API key or a model download.

### Engineering decisions

- Deterministic passage IDs connect evidence across both stores.
- Confidence filtering applies to every traversed relationship.
- Explicit hop counts distinguish direct neighbors from deeper targets.
- Regression tests check both multi-hop answers and direct questions with irrelevant path extensions.
- The report presents aggregate metrics alongside each expected and predicted answer.

### Evidence

The built-in demonstration contains 11 documents and three synthetic questions. The graph-and-vector system answered all three correctly, with 100% exact match and token F1 on that small set. The same run's vector baseline scored 0% on those answer metrics.

The repository includes the [run results](example-results.json), [standalone report](example-report.html), and tests for the pipeline's core behavior.

The [Streamlit setup guide](STREAMLIT.md) covers running the interactive app and adding its public deployment to the portfolio's existing app dialog.

### What the result means

The demonstration checks that the implemented retrieval and answer-selection steps work together on the included cases. It does not establish performance on a representative corpus, on the MuSiQue validation split, or under production load.

The next evaluation would need a separate dataset, stronger extraction and embeddings, and analysis of retrieval failures apart from answer-selection failures. The existing faithfulness score is a token-overlap proxy rather than a factuality assessment.

## Short description

Built a Streamlit GraphRAG demonstration with Kuzu and LanceDB, combining three-hop graph traversal, vector search, and inspectable evaluation reports. Added regression coverage for answer selection and confidence filtering.

## Resume wording

List this under independent projects. Choose the version that matches the role.

### Data engineering

**Multi-Hop Retrieval Pipeline | Independent Project**

Python, Kuzu, LanceDB, Streamlit, Pydantic, pytest

- Built a Python ingestion pipeline connecting graph and vector indexes through deterministic passage IDs, typed relationships, and confidence scores.
- Implemented graph traversal up to three hops and reciprocal rank fusion to combine relationship evidence with vector search.
- Added input validation, bounded concurrency, temporary-store cleanup, and 111 passing tests, including six Streamlit app flows.

### Data science

**Comparative Evaluation of Multi-Hop Retrieval | Independent Project**

Python, NumPy, Streamlit, information retrieval

- Compared graph-and-vector retrieval with a dense baseline using retrieval recall, MRR, answer exact match, and token F1.
- Diagnosed entity-ordering and answer-context errors, then added regression tests for the confirmed defects.
- Built an interactive app exposing answers, graph paths, and retrieved passages; documented the limits of a three-question synthetic benchmark.

This demonstrates retrieval implementation and evaluation. It does not establish production impact, trained-model development, or performance on a held-out real-world dataset.
