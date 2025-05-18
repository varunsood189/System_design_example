import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta

# Constants for simulation
num_rows = 1500
merchants = ['Amazon', 'Walmart', 'AppleStore', 'Ebay', 'Target']
transaction_types = ['purchase', 'refund']
locations = ['US', 'UK', 'IN', 'DE', 'FR']  # Added location list

# Generate random user_ids
user_ids = [str(uuid.uuid4()) for _ in range(100)]

# Helper function to generate timestamps
def random_timestamp(start, end):
    delta = end - start
    int_delta = delta.total_seconds()
    random_second = random.uniform(0, int_delta)
    return start + timedelta(seconds=random_second)

# Generate data rows
data = []
start_time = datetime.utcnow() - timedelta(days=7)
end_time = datetime.utcnow()

for _ in range(num_rows):
    user_id = random.choice(user_ids)
    amount = round(random.uniform(10, 2000), 2)
    merchant = random.choice(merchants)
    txn_type = random.choice(transaction_types)
    txn_time = random_timestamp(start_time, end_time)
    location = random.choice(locations)  # Pick random location
    # Simulated user features
    avg_txn_amt = round(random.uniform(50, 800), 2)
    var_txn_amt = round(random.uniform(0, 200), 2)
    unique_merchants = random.randint(1, 5)
    txn_freq_1hr = random.randint(1, 10)
    # Fraud label with some heuristic
    is_fraud = 1 if amount > 1500 or (txn_freq_1hr > 7 and amount > 500) else 0
    
    data.append({
        'transaction_id': str(uuid.uuid4()),
        'user_id': user_id,
        'amount': amount,
        'merchant': merchant,
        'transaction_type': txn_type,
        'timestamp': txn_time.isoformat() + 'Z',
        'location': location,  # Include location
        'time_since_last_txn': round(random.uniform(10, 10000), 2),
        'avg_txn_amt': avg_txn_amt,
        'var_txn_amt': var_txn_amt,
        'unique_merchants': unique_merchants,
        'txn_freq_1hr': txn_freq_1hr,
        'is_fraud': is_fraud
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "fraud_data.csv"
df.to_csv(csv_path, index=False)

csv_path

