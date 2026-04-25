"""Run demo server using TestClient to exercise endpoints without external uvicorn processes."""
import os
from demo_server import app
from fastapi.testclient import TestClient

# Ensure artifacts written to local folder; set ARTIFACT_BASE_URI for returned URIs
os.environ.setdefault("ARTIFACT_BASE_URI", "http://localhost:8001/explain")
# Do not set METADATA_INGESTOR_URL to avoid failing posts; recorder will skip POST
os.environ.setdefault("METADATA_INGESTOR_URL", "")

client = TestClient(app)

resp = client.post("/query", json={"query": "what invention done by albert einstein"})
print("status", resp.status_code)
print(resp.json())

# list artifacts created
import glob
print('\nArtifacts:')
for p in glob.glob('artifacts/**/*', recursive=True):
    print(p)
