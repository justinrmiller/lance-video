# video-lance documentation

Detailed reference docs for the `video-lance` pipeline and UI. For a quickstart,
see the top-level [README](../README.md); these pages go deeper.

| Doc | Covers |
|---|---|
| [architecture.md](architecture.md) | Module map, data flow, the LanceDB data model and Blob V1 storage, model/device management, and the config surface. |
| [pipeline.md](pipeline.md) | The ingest pipeline end to end: the stage protocol, each of the eight stages, idempotency, segmentation, accurate clip/frame extraction, embedding, and the embedding-model guard. |
| [search.md](search.md) | The three search modes, reciprocal-rank fusion, score normalization, the FTS + vector indexes, and `ensure_indexes`. |
| [cli-and-ui.md](cli-and-ui.md) | The `video-lance` CLI command/flag reference and a walkthrough of the Gradio app's three tabs. |
| [development.md](development.md) | Dev workflow: environment, tests, tooling gates, coding conventions, and how to extend the pipeline (add a stage, swap a model). |

## Orientation in one paragraph

`video-lance` walks a directory of videos, splits each into overlapping time
**segments**, and for every segment stores: the transcript text for that window,
a text embedding (e5-instruct, 1024-d), a visual embedding of a keyframe
(SigLIP 2, 1152-d), the keyframe JPEG, and a short MP4 clip. Everything lands in
a single **LanceDB** store (three tables: `videos`, `segments`, `_metadata`),
with the JPEG and MP4 kept out-of-line via Lance **Blob V1** columns. Search
runs three ways — `text` (hybrid e5 vector + full-text BM25), `visual` (SigLIP
cross-modal, text-or-image query), and `multi` (a weighted blend of the two) —
fused with reciprocal-rank fusion. A [Typer](https://typer.tiangolo.com/) CLI
(`ingest` / `search` / `info` / `reindex` / `ui`) and a [Gradio](https://gradio.app/)
web app are the two front ends.
