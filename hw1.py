import streamlit as st
import pandas as pd
import numpy as np
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- Generate product names ---
def generate_hardware_product_names():
    categories = {
        'Tools': ['Hammer', 'Screwdriver', 'Wrench', 'Pliers', 'Tape Measure', 'Level', 'Chisel', 'Utility Knife'],
        'Fasteners': ['Nails', 'Screws', 'Bolts', 'Washers', 'Anchors', 'Nuts', 'Rivets', 'Brads'],
        'Electrical': ['Extension Cord', 'Light Bulb', 'Switch', 'Outlet', 'Wire', 'Circuit Breaker', 'Fuse', 'Socket'],
        'Plumbing': ['Pipe', 'Valve', 'Faucet', 'Drain', 'Hose', 'Coupling', 'Elbow', 'Tee'],
        'Paint': ['Paint Brush', 'Roller', "Painter's Tape", 'Paint Tray', 'Sandpaper', 'Primer', 'Paint Can', 'Drop Cloth'],
        'Safety': ['Gloves', 'Goggles', 'Ear Protection', 'Dust Mask', 'Hard Hat', 'Fire Extinguisher', 'First Aid Kit', 'Knee Pads'],
        'Gardening': ['Shovel', 'Rake', 'Hoe', 'Garden Hose', 'Pruners', 'Wheelbarrow', 'Gloves', 'Fertilizer']
    }
    all_items = list(set([item for sublist in categories.values() for item in sublist]))
    extended_items = [f"{item} #{i+1}" for i in range((250 // len(all_items)) + 1) for item in all_items]
    return extended_items[:250]

# --- Generate transactions ---
def generate_sample_data(num_transactions=5000):
    random.seed(42)
    np.random.seed(42)
    items = generate_hardware_product_names()
    data = []
    for txn_id in range(1, num_transactions + 1):
        txn_items = random.sample(items, random.randint(1, 5))
        for item in txn_items:
            data.append({"TransactionID": txn_id, "Item": item})
    return pd.DataFrame(data)

# --- Bundle logic ---
def get_top_bundles(df, min_support=0.001):
    df['TransactionID'] = df['TransactionID'].astype(str)
    df['Item'] = df['Item'].astype(str)
    transactions = df.groupby('TransactionID')['Item'].apply(list)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.dropna(subset=['confidence', 'lift'])
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
    top_rules = rules.sort_values(by=['confidence', 'lift'], ascending=False).head(5)

    bundles = []
    for row in top_rules.itertuples():
        antecedent = list(row.antecedents)[0]
        consequent = list(row.consequents)[0]
        confidence_pct = row.confidence * 100
        lift = row.lift

        explanation = (
            f"Customers who buy **{antecedent}** often also buy **{consequent}** "
            f"(about {confidence_pct:.1f}% of the time). "
        )
        if lift > 1.1:
            explanation += "This is a strong bundle — they go together more often than you'd expect."
        elif lift >= 0.9:
            explanation += "These are frequently bought together, but not strongly related."
        else:
            explanation += "These two are sometimes bought together, but not very strongly."
        bundles.append(explanation)

    return bundles

# --- Streamlit UI ---
st.set_page_config(page_title="🛠️ Hardware Store Bundles", layout="centered")
st.title("🛒 Top 5 Product Bundles")
st.caption("Automatically generated using customer transaction patterns.")

with st.spinner("Analyzing transactions and finding popular item pairs..."):
    df = generate_sample_data()
    top_bundles = get_top_bundles(df)

if not top_bundles:
    st.error("Couldn't find any strong item pairs. Try again later.")
else:
    st.success("Found the top product bundles based on shopping habits!")
    for i, bundle_text in enumerate(top_bundles, 1):
        st.markdown(f"**{i}.** {bundle_text}")
