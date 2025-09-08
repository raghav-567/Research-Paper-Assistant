import numpy as np
from collections import defaultdict
class RAGPipeline:
    def __init__(self, search_agent, extraction_agent, summarizer_agent, index, id_to_metadata):
        self.search_agent = search_agent
        self.extraction_agent = extraction_agent
        self.summarizer_agent = summarizer_agent
        self.index = index 
        self.id_to_metadata = id_to_metadata  # chunk-level
        self.paper_metadata = {}  # paper-level

    def build_index(self, chunks, paper_info=None):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.summarizer_agent.embedding_model.encode(texts, convert_to_numpy=True)

        self.index.add(embeddings)

        start_idx = len(self.id_to_metadata)
        for i, chunk in enumerate(chunks):
            self.id_to_metadata[start_idx + i] = {
                "text": chunk["text"],
                "paper_id": chunk["metadata"]["paper_id"]
            }

        # Store paper-level metadata if provided
        if paper_info:
            self.paper_metadata[paper_info["id"]] = paper_info

    def query(self, query, top_k_chunks=200, top_k_papers=3, chunks_per_paper=3):
        query_emb = self.summarizer_agent.embedding_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k_chunks) 
        results = [(i, float(D[0][rank])) for rank, i in enumerate(I[0])]

        paper_scores = defaultdict(list)
        paper_chunks = defaultdict(list)

        for idx, score in results:
            if idx == -1:
                continue 
            chunk_info = self.id_to_metadata[idx]
            paper_id = chunk_info["paper_id"]
            text_chunk = chunk_info["text"]

            paper_scores[paper_id].append(score)
            paper_chunks[paper_id].append((score, text_chunk))

        ranked_papers = sorted(
            paper_scores.items(),
            key=lambda x: max(x[1]),
            reverse=True
        )[:top_k_papers]

        papers, summaries = [], []

        for paper_id, _ in ranked_papers:
            sorted_chunks = sorted(paper_chunks[paper_id], key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for _, chunk in sorted_chunks[:chunks_per_paper]]
            combined_text = " ".join(top_chunks)

            summary = self.summarizer_agent.summarizer(
                combined_text[:1000],
                max_length=150,
                min_length=60,
                do_sample=False
            )[0]['summary_text']

            paper_info = self.paper_metadata.get(
                paper_id,
                {"id": paper_id, "title": f"Paper {paper_id}", "authors": ["Unknown"]}
            )

            papers.append(paper_info)
            summaries.append(summary)

        return papers, summaries
