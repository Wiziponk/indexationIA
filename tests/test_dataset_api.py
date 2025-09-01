from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, Base, engine
from app.models import Dataset

client = TestClient(app)

# Prepare database with a sample dataset
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    db.add(Dataset(uid='u1', raw_path='r.parquet', emb_path='e.npy', config={}))
    db.commit()

def test_list_datasets():
    r = client.get('/api/datasets')
    assert r.status_code == 200
    assert any(d['uid'] == 'u1' for d in r.json())

def test_update_dataset():
    r = client.put('/api/datasets/u1', json={'label': 'hello'})
    assert r.status_code == 200
    assert r.json()['label'] == 'hello'

def test_delete_dataset():
    r = client.delete('/api/datasets/u1')
    assert r.status_code == 204
    r = client.get('/api/datasets')
    assert all(d['uid'] != 'u1' for d in r.json())
