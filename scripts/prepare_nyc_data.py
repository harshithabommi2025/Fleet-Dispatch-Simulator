"""Turn an NYC TLC yellow-taxi CSV (2015 or earlier, which include GPS coordinates)
into simulator demand: t_request_s, ox, oy, dx, dy in km.

Download e.g. yellow_tripdata_2015-01.csv from the NYC TLC trip record page
(or Kaggle mirrors), then:

    python scripts/prepare_nyc_data.py yellow_tripdata_2015-01.csv \
        --date 2015-01-15 --start 17:00 --hours 2 --out data/nyc_demand.csv
"""
import argparse
import math
from pathlib import Path

import pandas as pd

# Manhattan bounding box
LAT_MIN, LAT_MAX = 40.70, 40.82
LON_MIN, LON_MAX = -74.02, -73.93
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians((LAT_MIN + LAT_MAX) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--start", default="17:00")
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--sample", type=float, default=1.0, help="keep this fraction of trips")
    ap.add_argument("--out", default="data/nyc_demand.csv")
    a = ap.parse_args()

    cols = ["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude"]
    t0 = pd.Timestamp(f"{a.date} {a.start}")
    t1 = t0 + pd.Timedelta(hours=a.hours)
    chunks = []
    for chunk in pd.read_csv(a.csv, usecols=cols, parse_dates=[cols[0]], chunksize=500_000):
        chunk = chunk[(chunk[cols[0]] >= t0) & (chunk[cols[0]] < t1)]
        for lat, lon in (("pickup_latitude", "pickup_longitude"),
                         ("dropoff_latitude", "dropoff_longitude")):
            chunk = chunk[chunk[lat].between(LAT_MIN, LAT_MAX) & chunk[lon].between(LON_MIN, LON_MAX)]
        chunks.append(chunk)
    df = pd.concat(chunks)
    if a.sample < 1:
        df = df.sample(frac=a.sample, random_state=0)

    out = pd.DataFrame({
        "t_request_s": (df[cols[0]] - t0).dt.total_seconds(),
        "ox": (df.pickup_longitude - LON_MIN) * KM_PER_DEG_LON,
        "oy": (df.pickup_latitude - LAT_MIN) * KM_PER_DEG_LAT,
        "dx": (df.dropoff_longitude - LON_MIN) * KM_PER_DEG_LON,
        "dy": (df.dropoff_latitude - LAT_MIN) * KM_PER_DEG_LAT,
    }).sort_values("t_request_s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out):,} trips to {a.out}")


if __name__ == "__main__":
    main()
