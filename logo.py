import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- Step 1: Generate Automotive Spare Product Names ---
def generate_automotive_product_names():
    categories = {
        'Engine': ['Spark Plug', 'Oil Filter', 'Timing Belt', 'Radiator', 'Fuel Injector'],
        'Brakes': ['Brake Pad', 'Brake Disc', 'Caliper', 'Brake Fluid'],
        'Suspension': ['Shock Absorber', 'Strut', 'Control Arm', 'Ball Joint'],
        'Electrical': ['Battery', 'Alternator', 'Starter Motor', 'Fuse'],
        'Lights': ['Headlight', 'Taillight', 'Indicator', 'Fog Light'],
        'Transmission': ['Clutch Plate', 'Gearbox', 'Driveshaft', 'Flywheel'],
        'Tyres': ['Tyre', 'Wheel Rim', 'Valve Cap', 'Hubcap']
    }
    all_items = []
    for items in categories.values():
        all_items.extend(items)
    all_items = list(set(all_items))
    if len(all_items) < 250:
        multiplier = (250 // len(all_items)) + 1
        extended_items = [f"{item} #{i+1}" for i in range(multiplier) for item in all_items]
        all_items = extended_items[:250]
    else:
        all_items = all_items[:250]
    return all_items

# --- Step 2: Generate Sample Transaction Data ---
def generate_sample_data(num_transactions=5000):
    random.seed(42)
    np.random.seed(42)
    auto_items = generate_automotive_product_names()

    transactions = []
    for txn_id in range(1, num_transactions + 1):
        num_items_in_txn = random.randint(1, 5)
        items_in_txn = random.sample(auto_items, num_items_in_txn)
        for item in items_in_txn:
            transactions.append({"TransactionID": txn_id, "Item": item})

    df = pd.DataFrame(transactions)
    return df

# --- Step 3: Run Bundling Analysis ---
def run_bundling(df, min_support=0.003):
    df = df.dropna(subset=['TransactionID', 'Item'])
    df['TransactionID'] = df['TransactionID'].astype(str)
    df['Item'] = df['Item'].astype(str)

    transactions = df.groupby('TransactionID')['Item'].apply(list)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return "No bundles found. Try lowering the minimum support."

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
    rules = rules.sort_values(by=['confidence', 'lift'], ascending=False).head(5)

    explanations = []
    for i, row in enumerate(rules.itertuples(), 1):
        item_from = list(row.antecedents)[0]
        item_to = list(row.consequents)[0]
        explanations.append(f"**{i}. If a customer buys _{item_from}_, they are also likely to buy _{item_to}_.**")

    return explanations

# --- Step 4: Streamlit App ---
def main():
    st.set_page_config(page_title="Auto Spares Bundling", page_icon="🧰")

    # Header with logo
    logo = "logo.png"
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo, width=80)
    with col2:
        st.markdown("## Auto Spares Bundling Insights")
    st.markdown("---")

    # Simulate data and run analysis
    st.info("Simulated 5,000 auto spare transactions for bundling analysis.")
    df = generate_sample_data()
    bundles = run_bundling(df)

    # Show results
    st.subheader("Top 5 Product Bundles")
    if isinstance(bundles, list):
        for b in bundles:
            st.markdown(b)
    else:
        st.warning(bundles)

if __name__ == "__main__":
    main()
