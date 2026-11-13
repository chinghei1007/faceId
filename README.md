# faceId

Local-first face clustering and photo sorting. Import a folder, detect faces, create embeddings, cluster faces into stable `Person 001` identities, and correct assignments through a local web UI.

## Architecture

- **Backend:** FastAPI + SQLite + SQLAlchemy
- **Vision:** InsightFace `buffalo_l` when installed; deterministic mock provider for development/tests
- **Clustering:** HDBSCAN when installed, with a DBSCAN fallback
- **Frontend:** React + Vite
- **Storage:** originals remain in place; the app stores metadata, thumbnails, crops, and embeddings under the configured data directory

The app is designed for private/local use. InsightFace's public pretrained model weights have separate licensing terms from the MIT-licensed library; check the model terms before commercial deployment.

## Quick start

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` and use the Import panel with a local folder path. The API is available at `http://localhost:8000`.

## Docker

```bash
docker compose up --build
```

Set `FACEID_MODEL_PROVIDER=mock` for a dependency-light demo or `insightface` after installing model weights locally. The checked-in example uses `insightface` for real local detection. Set it to `mock` for a dependency-light UI demo. The first InsightFace run downloads the selected model pack locally.

## Processing and training

No training is required for the initial workflow. The system uses a pretrained detector/embedding model and unsupervised clustering. Collect a private validation set and tune thresholds before considering fine-tuning. The provider interface makes a commercially licensed model replaceable without changing the API or UI.

## Labels, corrections, and descriptions

`Person 001` is a stable internal identity. Renaming a person through `PATCH /api/people/{id}` changes the label everywhere without changing its ID. For a one-photo exception, `PATCH /api/faces/{face_id}/label` adds a photo-specific override and pins that face so reclustering will not move it. Merge and split operations also pin affected faces.

The optional `POST /api/photos/{photo_id}/caption` endpoint runs one local caption per photo. To enable it, install `transformers` and `torch`, set `FACEID_CAPTION_PROVIDER=transformers`, and use a local image-to-text model. Captioning is deliberately separate from face clustering so a large import can be indexed first and described later.

## Failed-file retries and student definitions

Jobs record failed file paths and errors. `POST /api/jobs/{id}/retry` with `{"scope":"failed_files"}` retries only those paths; use `{"scope":"batch"}` to retry the full source batch. Successful files are not rescanned by the failed-file retry.

`POST /api/student-definitions/parse` accepts a local PDF path and returns normalized `{class_code, student_number, student_key, name, label}` records plus warnings. Parsing applies Unicode normalization, removes recurring headers, joins Chinese name fragments split by whitespace or page layout, validates class/number formats, and flags names outside the accepted 2-4 Chinese-character range or duplicate student keys.

## Job timelines and definition management

Each batch version is displayed as one Imports/jobs row. Retries are grouped into that row as attempts. Expanding the row loads the job timeline, where every stage records start time, end time, elapsed duration, and any failure message. Individual file failures remain in the batch error record and can be retried without rescanning successful files.

Definition files have their own management view. Importing a contact-sheet PDF extracts its embedded portraits in label order, creates reference embeddings, and compares them against current identity prototypes. Results are shown as `matched`, `ambiguous`, or `unused`; ambiguous and unused definitions remain available for review. Definitions and import batches support soft disable/hide and hard delete.

Faces that cannot be confidently clustered are assigned a new `Person NNN` identity with `needs_review` status rather than an `Unknown` identity. This preserves the incrementing identity requirement while keeping uncertain occurrences visible in the Review queue.

## Folder import UI

The importer accepts either a server-local path or a folder dropped into the browser. The Browse folder control uses directory selection, preserves relative file names, uploads the selected files into a local staging directory, and starts the same processing job. No cloud upload is used; the upload is only to the local FastAPI process.

Definition imports are versioned. The default **Keep as new version** preserves the previous definition set for history and disables nothing. **Replace active version** soft-disables the prior set from the same source path before importing the new one. Person labels are stored on identities, so disabling or deleting a definition does not erase a label; automatic mapping only fills a default `Person NNN` label and never overwrites a label that was already customized.
