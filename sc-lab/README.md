# Supply chain lab

Three classic supply chain teaching scenarios as SimPy models, driven by an Excel workbook, replayed on a virtual tabletop that works in a Meta Quest headset or a desktop browser.

| Model | Concept | Variants |
|---|---|---|
| Beer game | Bullwhip effect in a four-echelon chain | human-like ordering, supply line fully counted, shared demand information, halved delays, random demand |
| Inventory policy | Reorder points, safety stock, service measures | (s,Q) with and without safety stock, higher service, periodic (R,S) review, variable lead times |
| Risk pooling | Centralising stock, square-root law | independent demand, correlated demand, higher variability, eight regions |

All parameter values are teaching placeholders. The workbook's source column records where each came from, and the three values for the Beer Game ordering rule are flagged for checking against Sterman (1989) before you quote them.

## Running

```
pip install -r requirements.txt
python run.py inputs/sc_inputs.xlsx --check        # validate only
python run.py inputs/sc_inputs.xlsx                # all variants, about 20 seconds
python run.py inputs/sc_inputs.xlsx --variants beer_base beer_info_sharing
```

Each run writes `outputs/run_<timestamp>/` (per-replication KPIs, summary with 95% confidence intervals, period-by-period data for the replayed replication, a manifest with the input hash, seed, overrides and parameter sources) and refreshes `viewer/traces/`.

## Viewer

Serve the `viewer` folder and open `index.html`:

```
cd viewer
python -m http.server 8000
```

For the headset, upload the project folder to GitHub Pages as before and open `.../sc-lab/viewer/index.html` in the Quest browser. `index.html?trace=beer_info_sharing.json` opens a given scenario.

The network sits on a table you can walk around. Blue trucks carry goods (cargo size shows quantity), white envelopes carry orders, tan crates are stock and red crates are backlog. The board behind the table shows the charts with a time cursor, the model results and prompts for discussion. The viewer only replays; every number comes from the SimPy run.

## Model notes

**Beer game.** Weekly periods. Each echelon receives shipments and orders that have arrived, ships against backlog plus new demand, updates an exponentially smoothed forecast and orders with an anchoring-and-adjustment rule:

order = forecast + alpha_s (desired inventory - net inventory) + alpha_s beta (forecast x lead time - supply line)

Orders and shipments are timed SimPy events. Desired supply line is expressed as forecast x lead time, so the chain starts in exact equilibrium for any delays; this separates Sterman's single desired-stock anchor into its stock and supply line parts. The order variance ratio is the variance of an echelon's weekly orders divided by the variance of customer demand.

**Inventory policy and risk pooling.** Customers arrive one unit at a time at random moments within each day, with normally distributed daily totals. Under (s,Q) the inventory position is checked after every sale, so orders are triggered exactly at the reorder point and the textbook formula s = mu L + z sigma sqrt(L) applies. With once-a-day review instead, stock "undershoots" the reorder point and service falls well below target, which makes a good extension exercise. Unmet demand is backordered. Cycle service is the share of replenishment cycles with no backorders; fill rate is the share of units served at once.

In the risk pooling model both systems face identical demand (common random numbers) and set reorder points for the same z. The central warehouse orders regions x Q, so cycle stock is equal and the saving isolates the safety stock effect.

Check results against theory before class: base inventory gives about 95% cycle service (z = 1.65), no safety stock about 50%, and central safety stock is half the decentralised total with four independent regions.
