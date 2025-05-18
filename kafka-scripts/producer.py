from confluent_kafka import Producer
import json
import time
import random
import uuid
from datetime import datetime

# Kafka config
conf = {
    'bootstrap.servers': 'localhost:9092'
}

producer = Producer(conf)

topic = 'transactions'

# Sample merchants and users for simulation
merchants = ['Amazon', 'Walmart', 'AppleStore', 'Ebay', 'Target']
users = [str(uuid.uuid4()) for _ in range(10)]  # 10 random users

def generate_transaction():
    transaction = {
        'transaction_id': str(uuid.uuid4()),
        'user_id': random.choice(users),
        'amount': round(random.uniform(10, 1000), 2),
        'merchant': random.choice(merchants),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'location': random.choice(['US', 'UK', 'IN', 'DE', 'FR']),
        'transaction_type': random.choice(['purchase', 'refund']),
    }
    return transaction

def delivery_report(err, msg):
    if err is not None:
        print(f'Delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_transactions(n=10, delay=1):
    for _ in range(n):
        txn = generate_transaction()
        producer.produce(topic, key=txn['transaction_id'], value=json.dumps(txn), callback=delivery_report)
        producer.poll(0)  # Trigger delivery callback
        time.sleep(delay)
    producer.flush()

if __name__ == '__main__':
    produce_transactions(n=20, delay=0.5)

