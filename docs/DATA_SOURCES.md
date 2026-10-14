# Data sources and current readiness

Checked and downloaded on 23 September 2026.

## Football statistics

[football-data.org v4](https://docs.football-data.org/general/v4/index.html) is
available as an alternative statistics source. The user approved using
Transfermarkt match statistics as well as valuations, so the completed real run
does not require a football-data.org token or a cross-provider ID mapping.

Use the visible credit: **Football data provided by the Football-Data.org API**.
See [provider attribution terms](https://www.football-data.org/about).

## Transfermarkt market-value labels

The selected dataset is
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
also published as [Football Data from Transfermarkt on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores).
It is a community-published dataset sourced from Transfermarkt, not an official
Transfermarkt API. Its
[dataset metadata](https://github.com/dcaribou/transfermarkt-datasets/blob/master/data/prep/dataset-metadata.json)
declares CC0/Public Domain. Retain source attribution and snapshot provenance.

Local files in `data/raw/transfermarkt/`:

| File | Observed row count | Purpose |
| --- | ---: | --- |
| `players.csv.gz` | 50,149 | Identity matching using names and birth dates |
| `player_valuations.csv.gz` | 656,301 | Dated market-value targets in EUR |
| `games.csv.gz` | 88,958 | Fixture identities, season/competition dates and coverage checks |
| `appearances.csv.gz` | 1,894,350 | Player-match minutes, goals, assists and cards |
| `download_metadata.json` | — | Download URLs, UTC timestamp, bytes and SHA-256 hashes |

The downloaded valuation dates range from 20 January 2000 to 12 June 2026.
There were no duplicate player/date pairs in this downloaded valuation table.
The publisher reports that updates are paused; do not treat it as a live feed.
No website scraping was used by this project: the downloader retrieved the four
published compressed files. Files are excluded from Git.

Transfermarkt IDs are a different namespace from football-data.org IDs. The
mapping template and review step prevent accidental joins between unrelated
players when combining providers. The real Transfermarkt-only path shares IDs
across its tables and records `player_id_namespace=transfermarkt` throughout.
Namespace checks prevent those IDs from being treated as football-data.org IDs.

Only dated `player_valuations.market_value_in_eur` is used as the target. The
player table's current/highest values, contract information and current club do
not enter model features. A market valuation is not an actual transfer fee.

## What is ready and what remains

The implementation includes preparation, joining, training, testing, plotting
and prediction. `models/transfermarkt-baseline/model.joblib` is trained on real
data, with independent later-season evaluation. Demo model directories remain
explicitly synthetic. See [the real-model report](REAL_MODEL.md) for measured
accuracy, coverage exclusions and limitations.
