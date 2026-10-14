# Publish Touchline on Render's free plan

This package runs Touchline as a public hobby/demo website. Deployment status and
the assigned public URL are shown in your Render dashboard.
It includes the trained in-season model, dated input data, cached live snapshots,
and precomputed season-history predictions. No API token or paid data source is
needed. Do not upload the original workspace: use the release zip, which excludes
credentials, logs, source-conversation files and unrelated experiments.

## Put it online

1. Extract `touchline-ready-to-host.zip` to a new folder. Keep the structure intact.
2. Use the public [Touchline repository](https://github.com/NoxiousHangman/Touchline_Football_Value) and upload the extracted contents,
   including `deploy/assets`. These assets are required; uploading only the HTML
   or Python files will not work. Do not add your `.env` or account tokens.
3. In [Render](https://dashboard.render.com/), choose **New → Blueprint** and
   connect that repository. Render reads the included `render.yaml`.
4. Confirm the service is **Free**, with the Docker runtime, and deploy. No paid
   database, persistent disk or domain is required. Keep paid add-ons disabled;
   review account usage limits rather than enabling chargeable overages.
5. Wait until the build and `/healthz` check pass, then open the assigned
   `https://<your-service>.onrender.com/live` address. This is the shareable website
   link. The loopback `127.0.0.1` address works only on your computer.
6. Check all five league statuses, search for a player, test budget filtering and
   compare two saved players. An initial update may take several minutes.

The release source must sit at the repository root, not inside an extra enclosing
folder. If using **New → Web Service** instead, select Docker, Free, the root
`Dockerfile`, and health-check path `/healthz`. Leave Docker Command empty.

## Updating an existing Render connection

For a Blueprint-managed service, sync the latest `render.yaml`. It explicitly
selects `main`, `./Dockerfile`, the repository-root build context, one Free
instance, `/healthz` and automatic deployments on commits. It binds the server
to `0.0.0.0` and lets Render supply `PORT`.

For a manually created Web Service, check those settings in the dashboard;
committing `render.yaml` does not convert the service into a Blueprint. Retain
the existing service rather than creating a second one. Use **Manual Deploy →
Deploy latest commit** if automatic deployments are disabled. Wait for **Live**,
then check `/healthz`, `/live`, `/about` and `/privacy` on the assigned public URL.
The health endpoint confirms the server is running; each league's freshness
status separately reports whether current statistics are available.

## Free hosting limits

Render's free web service sleeps after 15 minutes without inbound traffic and
may take about a minute to wake. Its local disk changes are lost on sleep/restart
or redeploy. The image includes dated seed snapshots; the server checks freshness
and fetches new data on waking. A seed older than 48 hours is not shown as live.
If providers cannot be reached, affected leagues can remain unavailable until a
successful refresh. Do not use artificial traffic to keep a free instance awake.

The six-hour updater runs **only while the server is awake**. Free hosting cannot
promise uninterrupted scheduled updates or production availability. Shortlists
are saved on each visitor's browser, so server restarts do not delete them.
Models, fee premium, preseason valuations and the fixed GBP conversion do not
automatically retrain or change. Rebuild the bundle to update those inputs.

The service uses one process and four HTTP threads so there is only one updater.
Requests share a 10-second combined snapshot cache. History is precomputed and
live identity/valuation inputs are trimmed to reduce memory. Render Free has
512 MB RAM; check actual memory and provider request volume after deployment.
Do not scale this filesystem-based design to multiple instances without shared
storage and a single coordinated updater.

## Configuration

| Variable | Default / purpose |
| --- | --- |
| `PORT` | Render supplies this; container default is `10000` |
| `HOST` | Docker uses `0.0.0.0`; a direct local run uses `127.0.0.1` |
| `RENDER_EXTERNAL_URL` | Supplied by Render; used for the sitemap and HTTPS headers |
| `PUBLIC_BASE_URL` | Optional bare HTTPS origin for a custom domain |
| `TOUCHLINE_AUTO_UPDATE` | `1`; use `0` only for an offline test |
| `TOUCHLINE_DATA_DIR` | Container's writable packaged input/cache folder |
| `TOUCHLINE_MODELS_DIR` | Container's trusted model folder |
| `TOUCHLINE_HISTORY_PATH` | Packaged historical prediction snapshot |

No environment file is automatically loaded. Set values in the host dashboard.
Never accept model uploads from visitors: joblib files are executable artifacts.
Manual public refresh requests are disabled. The server downloads from its fixed
football sources only; users cannot supply download URLs or filesystem paths.
Raw files, model artifacts, `.env` and repository contents have no HTTP routes.
The host terminates HTTPS. Request body/header sizes and concurrent connections
are bounded; errors return a generic message rather than a traceback.

## Local verification of the same production server

With Docker installed, from the extracted release directory:

```powershell
docker build -t touchline .
docker run --rm -p 127.0.0.1:8766:10000 touchline
```

Open `http://127.0.0.1:8766/live`. The normal local development server can continue
on port 8765. Docker is optional for using the existing project locally.

The release pins the Python dependencies to the training environment. Do not
upgrade scikit-learn independently of the saved model; the application checks
the model's training version before live inference.

## Build another release from the original workspace

This section is for the original training workspace. The public hosting release
does not include the full training data, test suite or `prepare_release.py`.

Use the trusted training environment, with no `TOUCHLINE_HISTORY_PATH` override:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_release.py
.\.venv\Scripts\python.exe scripts/check_release.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The output is `dist/touchline-ready-to-host.zip`. Refresh locally first if you want
fresh seed snapshots. Original timestamps are preserved. The builder does not
retrain, fetch third-party data, publish externally or modify the original model.
`deploy/assets/manifest.json` records SHA-256 hashes verified in the image build.
It also records the season and input cutoff; rebuild for a new season.

## Attribution and publication scope

The site includes visible sources, an About page explaining estimates and a
Privacy page explaining browser storage and hosting logs. Keep them in the public
version. The Transfermarkt dataset publisher declares CC0; its provenance is
included. FPL and ESPN remain third-party sources, and public endpoint access is
not itself a blanket redistribution licence. Check the providers' current terms
for your intended public/commercial use. No endorsement or official partnership
is claimed. No player photographs or club crest rights are assumed.

The fee uplift uses an unverified, selected 25-player reference sample, not a
validated current transfer-fee dataset. It is presented as an illustrative
scenario. Full-season validation is still pending and is not claimed complete.

Blueprint and free-service references checked 14 October 2026:
- [Render free plan and limits](https://render.com/docs/free)
- [Docker deployment](https://render.com/docs/docker)
- [Blueprint configuration](https://render.com/docs/blueprint-spec)
- [Compute plans](https://render.com/docs/compute-plans)
- [Waitress server options](https://docs.pylonsproject.org/projects/waitress/en/stable/arguments.html)
