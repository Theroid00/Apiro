import logging

from apiro.corpus.embedder import Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repair_corpus")

def main():
    embedder = Embedder()
    collection = embedder._collection
    count = collection.count()
    logger.info(f"Total documents in collection: {count}")
    
    batch_size = 5000
    updated_count = 0
    
    for offset in range(0, count, batch_size):
        logger.info(f"Fetching batch at offset {offset} (progress: {offset/count:.1%})...")
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )
        
        ids = batch.get("ids", [])
        metadatas = batch.get("metadatas", [])
        
        if not ids:
            break
            
        update_ids = []
        update_metadatas = []
        
        for i, meta in enumerate(metadatas):
            doc_id = ids[i]
            needs_update = False
            
            # If meta is None, initialize it as empty dict
            if meta is None:
                meta = {}
                needs_update = True
            
            # Check if evidence_level is missing or None
            if "evidence_level" not in meta or meta["evidence_level"] is None:
                source = str(meta.get("source_db", "")).lower()
                if "pubmed" in source:
                    meta["evidence_level"] = 2
                elif "omim" in source:
                    meta["evidence_level"] = 3
                elif "clinvar" in source:
                    meta["evidence_level"] = 1
                elif "openfda" in source:
                    meta["evidence_level"] = 2
                else:
                    meta["evidence_level"] = 2 # default fallback
                needs_update = True
                
            if needs_update:
                update_ids.append(doc_id)
                update_metadatas.append(meta)
                
        if update_ids:
            logger.info(f"Updating {len(update_ids)} documents in this batch...")
            # We only need to update the metadata, not documents or embeddings
            collection.update(
                ids=update_ids,
                metadatas=update_metadatas
            )
            updated_count += len(update_ids)
            
    logger.info(f"Repair complete! Updated {updated_count} documents.")

if __name__ == "__main__":
    main()
