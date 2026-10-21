"""
Synthetic Dataset Generator
============================
Generates 4 inter-connected CSV files that simulate a realistic
product quality data environment:
  - data/raw/sales.csv
  - data/raw/returns.csv
  - data/raw/reviews.csv
  - data/raw/manufacturing.csv

All datasets share `product_id` as the primary key.
Run:  python src/data/generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)

CATEGORIES   = ["Electronics", "Apparel", "Home & Kitchen", "Sports", "Automotive"]
REGIONS      = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
SUPPLIERS    = [f"Supplier_{chr(65+i)}" for i in range(10)]      # Supplier_A … Supplier_J
LINES        = [f"Line_{i:02d}" for i in range(1, 9)]            # Line_01 … Line_08
DEFECT_TYPES = ["Packaging", "Functional", "Cosmetic", "Safety", "Missing Parts"]
RETURN_REASONS = [
    "Defective product", "Not as described", "Changed mind",
    "Wrong item shipped", "Arrived damaged", "Missing accessories",
]

N_PRODUCTS       = 300
N_SALES          = 12_000
N_RETURNS        = 3_500     
N_REVIEWS        = 9_000
N_MANUFACTURING  = 2_500

# ── Positive and negative review templates ────────────────────────────────────
POSITIVE_TEMPLATES = [
    "Absolutely love this product! Works perfectly and exceeded my expectations.",
    "Great quality for the price. Highly recommend to anyone looking for {cat} items.",
    "Fast shipping and the product is exactly as described. Very satisfied!",
    "This is a fantastic {cat} product. Build quality is excellent.",
    "Works like a charm. Customer service was also very helpful.",
    "Five stars! The {cat} product is durable and well-designed.",
    "Impressive performance. Will definitely buy again from this brand.",
    "Outstanding product. My family loves it and it's easy to use.",
]

NEGATIVE_TEMPLATES = [
    "Disappointed with the quality. The product stopped working after two weeks.",
    "Returned this {cat} item because it was defective right out of the box.",
    "Poor packaging led to the product arriving damaged. Very frustrating.",
    "Does not match the description at all. Waste of money.",
    "Had to return it. The {cat} product had missing parts.",
    "Terrible experience. The item broke within days of use.",
    "Not worth the price. Expected much better quality.",
    "Customer service was unhelpful and the product is faulty.",
]

NEUTRAL_TEMPLATES = [
    "It's okay for the price but nothing special.",
    "Average quality. Does the job but could be better.",
    "Decent {cat} product. Some minor issues but overall acceptable.",
    "Not bad, not great. Exactly what you'd expect for this price range.",
    "Works as advertised. Nothing more, nothing less.",
]


def random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_products() -> pd.DataFrame:
    products = []
    for i in range(1, N_PRODUCTS + 1):
        cat = random.choice(CATEGORIES)
        products.append({
            "product_id": f"PROD_{i:04d}",
            "product_name": f"{fake.word().capitalize()} {cat} {fake.word().capitalize()}",
            "category": cat,
            "unit_price": round(random.uniform(10, 500), 2),
            "launch_date": random_date(START_DATE, datetime(2023, 6, 1)).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(products)


def generate_sales(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _ in range(N_SALES):
        prod  = products.sample(1).iloc[0]
        date  = random_date(START_DATE, END_DATE)
        units = random.randint(1, 200)
        rows.append({
            "sale_id":    f"SALE_{len(rows)+1:06d}",
            "product_id": prod["product_id"],
            "category":   prod["category"],
            "date":       date.strftime("%Y-%m-%d"),
            "units_sold": units,
            "revenue":    round(prod["unit_price"] * units, 2),
            "region":     random.choice(REGIONS),
            "discount_pct": round(random.uniform(0, 0.3), 2),
        })
    return pd.DataFrame(rows)


def generate_returns(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _ in range(N_RETURNS):
        prod   = products.sample(1).iloc[0]
        date   = random_date(START_DATE, END_DATE)
        rows.append({
            "return_id":   f"RET_{len(rows)+1:06d}",
            "product_id":  prod["product_id"],
            "category":    prod["category"],
            "date":        date.strftime("%Y-%m-%d"),
            "return_qty":  random.randint(1, 10),
            "reason":      random.choice(RETURN_REASONS),
            "defect_type": random.choice(DEFECT_TYPES + [None, None]),  # some not defective
            "region":      random.choice(REGIONS),
            "refund_amt":  round(prod["unit_price"] * random.randint(1, 10), 2),
        })
    return pd.DataFrame(rows)


def generate_reviews(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _ in range(N_REVIEWS):
        prod   = products.sample(1).iloc[0]
        date   = random_date(START_DATE, END_DATE)
        rating = random.choices([1, 2, 3, 4, 5], weights=[8, 10, 15, 30, 37])[0]

        if rating >= 4:
            tmpl = random.choice(POSITIVE_TEMPLATES)
        elif rating <= 2:
            tmpl = random.choice(NEGATIVE_TEMPLATES)
        else:
            tmpl = random.choice(NEUTRAL_TEMPLATES)

        text = tmpl.replace("{cat}", prod["category"].lower())

        rows.append({
            "review_id":   f"REV_{len(rows)+1:07d}",
            "product_id":  prod["product_id"],
            "category":    prod["category"],
            "date":        date.strftime("%Y-%m-%d"),
            "rating":      rating,
            "review_text": text,
            "verified_purchase": random.choice([True, True, True, False]),
        })
    return pd.DataFrame(rows)


def generate_manufacturing(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Assign each product to a supplier/line (consistent mapping)
    prod_supplier = {p: random.choice(SUPPLIERS) for p in products["product_id"]}
    prod_line     = {p: random.choice(LINES)     for p in products["product_id"]}

    for _ in range(N_MANUFACTURING):
        prod = products.sample(1).iloc[0]
        pid  = prod["product_id"]
        supplier = prod_supplier[pid]
        line     = prod_line[pid]

        # Some suppliers have higher defect rates (realistic)
        base_defect = 0.02 + SUPPLIERS.index(supplier) * 0.008
        defect_rate = max(0, min(1, np.random.normal(base_defect, 0.01)))

        rows.append({
            "batch_id":        f"BATCH_{len(rows)+1:06d}",
            "product_id":      pid,
            "category":        prod["category"],
            "production_date": random_date(START_DATE, END_DATE).strftime("%Y-%m-%d"),
            "supplier":        supplier,
            "production_line": line,
            "batch_size":      random.randint(100, 5000),
            "defect_rate":     round(defect_rate, 4),
            "high_defect_flag": int(defect_rate > 0.05),
            "quality_check_passed": int(defect_rate < 0.08),
            "lead_time_days":  random.randint(3, 60),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🏭  Generating synthetic product quality datasets...")

    products = generate_products()
    products.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    print(f"  ✅ products.csv        → {len(products):,} rows")

    sales = generate_sales(products)
    sales.to_csv(os.path.join(OUTPUT_DIR, "sales.csv"), index=False)
    print(f"  ✅ sales.csv           → {len(sales):,} rows")

    returns = generate_returns(products)
    returns.to_csv(os.path.join(OUTPUT_DIR, "returns.csv"), index=False)
    print(f"  ✅ returns.csv         → {len(returns):,} rows")

    reviews = generate_reviews(products)
    reviews.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)
    print(f"  ✅ reviews.csv         → {len(reviews):,} rows")

    manufacturing = generate_manufacturing(products)
    manufacturing.to_csv(os.path.join(OUTPUT_DIR, "manufacturing.csv"), index=False)
    print(f"  ✅ manufacturing.csv   → {len(manufacturing):,} rows")

    print(f"\n📁 All files saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
