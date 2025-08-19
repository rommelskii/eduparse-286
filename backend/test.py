from session import Database

DB_URL = 'mongodb://localhost:27017/'
COLLECTION_NAME = 'eduparse'


d = Database(DB_URL, COLLECTION_NAME)

arr = d.fetchSessionByName("rommel")

print(arr)

d.removeSessionByName("rommel", "Mitochondrial structure")

arr = d.fetchSessionByName("rommel")

print(arr)

