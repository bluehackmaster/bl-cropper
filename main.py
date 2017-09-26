import uuid

from google.cloud import bigquery

def query():
    client = bigquery.Client.from_service_account_json(
        'BlueLens-d8117bd9e6b1.json')

    query = 'SELECT * FROM stylelens.stylenanda LIMIT 10;'

    query_job = client.run_async_query(str(uuid.uuid4()), query)

    query_job.begin()
    query_job.result()  # Wait for job to complete.

    # Print the results.
    destination_table = query_job.destination
    destination_table.reload()
    for row in destination_table.fetch_data():
        print(row)

if __name__ == '__main__':
    query()