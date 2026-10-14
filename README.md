# Touchline

**Find your next signing.** Explore football player values, compare potential targets and build a shortlist across Europe's top five leagues, with every amount shown in pounds.

Touchline is a Python football analysis website with a trained market-value model and an illustrative transfer-fee scenario. This repository contains the deployable website, its model and the packaged data needed to start it.

## Explore the site

Open **`/live`** on your deployed website for the current-season player browser. The **Season history** link opens historical predictions at `/`.

1. Set a transfer budget and age range, then choose a position and sorting order.
2. Select **Find players** and search by player or club. Narrow the results to a league if needed.
3. Open a player to see their value, available league statistics, data timestamp and explanation of the estimate.
4. Save players using the bookmark button. Expand **Saved shortlist** to compare up to three players side by side.
5. Switch between the default **transfer-fee scenario** and the underlying **market-value estimate** to understand the difference.

Shortlists are saved in your browser; they do not require an account and do not sync between devices. Budget ranges filter the selected value view. They are search controls, not confidence intervals.

### League coverage

| League | Current statistics source |
| --- | --- |
| Premier League | Official Fantasy Premier League |
| La Liga | ESPN |
| Bundesliga | ESPN |
| Serie A | ESPN |
| Ligue 1 | ESPN |

Coverage varies by provider and player. Missing statistics stay unavailable rather than being presented as zero. The packaged history contains 4,306 players; current-season totals change as sources update.

## How the values work

The current-season model combines a dated preseason valuation with age, position, league and available match statistics. A model estimate requires an unambiguous player identity and at least 180 verified league minutes. Where an estimate is unavailable, the site may show a clearly labelled dated reference or no price.

The main view applies an approximately **33.3% premium** to the model value or eligible dated reference:

```text
Illustrative transfer fee = base market value × 1.333…
Example: £60m market value → £80m fee scenario
```

That multiplier comes from the median fee-to-value ratio in a selected 25-player reference table. The transactions were not independently verified, and this is not a trained or validated transfer-fee model. Actual negotiations, contracts, injuries and club circumstances can produce very different fees. Values can exceed £200m; there is no artificial £200m ceiling.

All amounts use the fixed presentation conversion **€1 = £0.85950**. This is not a live exchange-rate feed. The experimental current-season model supports September–April; full-season validation remains pending. The site's **Methodology** panel and [About page](web/about.html) explain the limitations.

## Keeping data current

While the server is running, the updater refreshes match-data snapshots older than six hours. Each league shows its own freshness information. Recent cached data can remain available during a provider failure; snapshots older than 48 hours are hidden until a successful refresh.

Updates recompute estimates using new statistics. They **do not** retrain the model or update the historical valuations, fixed exchange rate or fee premium. Those inputs require a reviewed replacement release. The Transfermarkt community dataset is a dated input, not a live valuation feed.

## Run locally

With Docker installed, run from the repository root:

```sh
docker build -t touchline .
docker run --rm -p 127.0.0.1:8766:10000 touchline
```

Open `http://127.0.0.1:8766/live`. No paid API or token is required for the packaged website. Preserve the pinned dependencies when using the supplied model.

## Project layout

| Path | Purpose |
| --- | --- |
| `web/` | Player browser, About and Privacy pages |
| `app.py`, `serve.py` | Prediction service and production entry point |
| `src/` | Collection, preparation, modelling and HTTP modules |
| `deploy/assets/` | Trained model, dated inputs, snapshots and integrity manifest |
| `docs/` | Deployment, source attribution and fee reference |
| `scripts/check_release.py` | Verifies packaged model/data integrity |

Built with Python, requests, pandas, scikit-learn, matplotlib and Waitress, with an HTML/CSS/JavaScript interface. This is the hosting release, not the full research workspace or a complete retraining dataset.

## Sources and attribution

- [Official Fantasy Premier League](https://fantasy.premierleague.com/) supplies Premier League match statistics; fantasy prices and fantasy assists are not used as market values or comparable assists.
- [ESPN](https://www.espn.com/soccer/) supplies public squad and statistics feeds for the other four leagues. Availability is not guaranteed.
- [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) supplies historical statistics, identities and valuations. This is a community dataset, not an official Transfermarkt API. Snapshot provenance is included with the assets.
- The alternative [football-data.org](https://www.football-data.org/) collector uses an environment-variable token. **Football data provided by the Football-Data.org API.** It is not required for the current live website.

Touchline is an independent project, without provider endorsement. Retain source credits and the site's About and Privacy pages. See [data sources](docs/DATA_SOURCES.md) and [privacy information](web/privacy.html) for more detail.
