from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session
from app.main import app
from app.db import engine
from app.models import Dataset

client = TestClient(app)

# Prepare database with a sample dataset
SQLModel.metadata.drop_all(bind=engine)
SQLModel.metadata.create_all(bind=engine)
with Session(engine) as session:
    session.add(Dataset(uid='u1', raw_path='r.parquet', emb_path='e.npy', config={}))
    session.commit()

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
