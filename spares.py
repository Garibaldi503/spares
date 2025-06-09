import streamlit as st
import pandas as pd
import numpy as np
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def generate_automotive_spares():
    categories = {
        'Engine': ['Oil Filter', 'Air Filter', 'Spark Plug', 'Fuel Injector', 'Timing Belt', 'Piston', 'Crankshaft'],
        'Brakes': ['Brake Pads', 'Brake Disc', 'Brake Caliper', 'Brake Fluid'],
        'Suspension': ['Shock Absorber', 'Strut Mount', 'Control Arm', 'Ball Joint'],
        'Electrical': ['Car Battery', 'Alternator', 'Starter Motor', 'Ignition Coil', 'Headlight Bulb', 'Fuse'],
        'Cooling': ['Radiator', 'Coolant Hose', 'Thermostat', 'Water Pump'],
        'Transmission': ['Clutch Plate', 'Gearbox', 'Transmission Fluid'],
        'Body': ['Side Mirror', 'Headlight Assembly', 'Bumper', 'Wiper Blade'],
        'Tyres': ['Tyre', 'Wheel Rim', 'Valve Cap']
    }
    all_items = list(set([item for sublist in categories.values() for item in sublist]))
    multiplier = (250 // len(all_items)) + 1
    extended_items = [f"{item} #{i+1}" for i in range(multiplier) for item in all_items]
    return extended_items[:250]

def generate_sample_data(num_transactions=5000):
    random.seed(42)
    np.random.seed(42)

    automotive_items = generate_automotive_spares()

    transactions = []
    for txn_id in range(1, num_transactions + 1):
        num_items_in_txn = random.randint(1, 5)
        items_in_txn = random.sample(automotive_items, num_items_in_txn)
        for item in items_in_txn:
            transactions.append({"TransactionID": txn_id, "Item": item})

    df = pd.DataFrame(transactions)
    return df

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
        st.write("No popular item combinations found. Try adjusting the settings.")
        return

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules['confidence'] = pd.to_numeric(rules['confidence'], errors='coerce')
    rules['lift'] = pd.to_numeric(rules['lift'], errors='coerce')
    rules = rules.dropna(subset=['confidence', 'lift'])

    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]
    top_bundles = rules.sort_values(by=['confidence', 'lift'], ascending=False).head(5)

    st.markdown("## Suggested Automotive Spare Part Combos")
    st.markdown("These combinations are based on common purchasing behavior among customers:")

    for i, row in enumerate(top_bundles.itertuples(), 1):
        item_a = list(row.antecedents)[0]
        item_b = list(row.consequents)[0]
        times_together = row.support * 100
        likely_together = row.confidence * 100
        higher_than_random = row.lift

        st.markdown(f"### {i}. {item_a} + {item_b}")
        st.write(f"Customers who bought **{item_a}** often also bought **{item_b}**.")
        st.write(f"- They were found together in about {times_together:.1f}% of all sales.")
        st.write(f"- When someone picks **{item_a}**, there's a high chance they'll add **{item_b}** to their purchase.")
        if higher_than_random > 1:
            st.write(f"- This combination happens more often than you'd expect by chance — it's a smart bundle idea!")

def main():
    st.title("Automotive Spare Parts Bundling Ideas")
    df_generated = generate_sample_data()
    run_bundling(df_generated)

if __name__ == "__main__":
    main()
