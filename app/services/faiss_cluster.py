import faiss
import numpy as np
from collections import Counter
from app.utils.supabase_client import get_supabase_client

EMBED_DIM = 384  # Adjust if your model is different
SIM_THRESHOLD = 0.7  # Cosine similarity threshold for new cluster

class FaissClusterManager:
    def __init__(self):
        self.index = None
        self.embeddings = []
        self.article_ids = []
        self.cluster_ids = []
        self.supabase = get_supabase_client()
        self._load()

    def _load(self):
        # Load all embeddings and cluster_ids from Supabase
        res = self.supabase.table("articles").select("id,embedding,cluster_id,tags").not_.is_("embedding", "null").execute()
        self.embeddings = []
        self.article_ids = []
        self.cluster_ids = []
        self.tags = {}
        for row in res.data or []:
            if row["embedding"]:
                self.embeddings.append(row["embedding"])
                self.article_ids.append(row["id"])
                self.cluster_ids.append(row.get("cluster_id") or "0")
                self.tags[row["id"]] = row.get("tags", [])
        if self.embeddings:
            arr = np.array(self.embeddings, dtype=np.float32)
            faiss.normalize_L2(arr)
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.index.add(arr)

    def assign(self, embedding: np.ndarray) -> str:
        if self.index is None or self.index.ntotal == 0:
            return "0"
        emb = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, 1)
        sim = float(D[0][0])
        idx = int(I[0][0])
        if sim < SIM_THRESHOLD:
            # New cluster
            new_id = str(max([int(cid) for cid in self.cluster_ids if cid.isdigit()] + [0]) + 1)
            return new_id
        return self.cluster_ids[idx]

    def label(self, cluster_id: str) -> str:
        # Get top tags for this cluster
        tag_list = []
        for i, cid in enumerate(self.cluster_ids):
            if cid == cluster_id:
                tag_list.extend(self.tags.get(self.article_ids[i], []))
        if not tag_list:
            return "Unlabeled"
        top = [t for t, _ in Counter(tag_list).most_common(3)]
        return " / ".join(top)

faiss_manager = FaissClusterManager()

def assign_cluster(embedding):
    return faiss_manager.assign(np.array(embedding, dtype=np.float32))

def label_cluster(cluster_id):
    label = faiss_manager.label(cluster_id)
    desc = f"Cluster {cluster_id}: {label}"
    return label, desc

def reload_faiss():
    faiss_manager._load() 