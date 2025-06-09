import streamlit as st
import pandas as pd
import numpy as np
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- PRODUCT NAME GENERATOR ---
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
    if len(all_items) < 250:
        extended_items = [f"{item} #{i+1}" for i in range((250 // len(all_items)) + 1) for item in all_items]
        return extended_items[:250]
    return all_items[:250]

# --- DATA GENERATOR ---
def generate_sample_data(num_transactions=5000):
    random.seed(42)
    np.random.seed(42)
    hardware_items = generate_hardware_product_names()

    transactions = []
    for txn_id in range(1, num_transactions + 1):
        num_items_in_txn = random.randint(1, 5)
        items_in_txn = random.sample(hardware_items, num_items_in_txn)
        for item in items_in_txn:
            transactions.append({"TransactionID": txn_id, "Item": item})

    return pd.DataFrame(transactions)

# --- BUNDLING LOGIC ---
def run_bundling(df, min_support=0.001):
    df = df.dropna(subset=['TransactionID', 'Item'])
    df['TransactionID'] = df['TransactionID'].astype(str)
    df['Item'] = df['Item'].astype(str)

    transactions = df.groupby('TransactionID')['Item'].apply(list)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return None, None

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.dropna(subset=['confidence', 'lift'])
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
    top_bundles = rules.sort_values(by=['confidence', 'lift'], ascending=False).head(5)

    return frequent_itemsets, top_bundles

# --- STREAMLIT DASHBOARD ---
st.set_page_config(page_title="Hardware Store Bundling", layout="centered")
st.title("🔧 Hardware Store Bundling Analysis")
st.caption("Simulated sales transactions and bundling analysis using Apriori algorithm.")

with st.spinner("Simulating data and analyzing bundles..."):
    df = generate_sample_data()
    frequent_itemsets, top_bundles = run_bundling(df)

if frequent_itemsets is None or top_bundles is None or top_bundles.empty:
    st.error("No bundles found. Try adjusting thresholds or check data generation.")
else:
    st.success("Bundling complete!")

    st.markdown("### 📦 Frequent Itemsets (Top 10)")
    st.dataframe(frequent_itemsets.head(10))

    st.markdown("### 🔗 Top 5 Bundles with Explanation")
    for i, row in enumerate(top_bundles.itertuples(), 1):
        antecedent = list(row.antecedents)[0]
        consequent = list(row.consequents)[0]
        support_pct = row.support * 100
        confidence_pct = row.confidence * 100
        lift = row.lift

        st.markdown(f"**{i}. {antecedent} → {consequent}**")
        st.markdown(f"- **Support**: {support_pct:.2f}% of transactions")
        st.markdown(f"- **Confidence**: If someone buys '{antecedent}', they also buy '{consequent}' {confidence_pct:.2f}% of the time.")
        if lift > 1:
            st.markdown(f"- **Lift**: {lift:.2f} — Customers buy them together more than random chance. ✅")
        elif lift == 1:
            st.markdown(f"- **Lift**: {lift:.2f} — Neutral relationship.")
        else:
            st.markdown(f"- **Lift**: {lift:.2f} — Weak relationship.")

