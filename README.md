# News-Classification-Based-on-LLMs-Python-and-RAG
We developed an LLM chatbot for automated news text processing and classification, and categorize the classification (Business, Politics, Entertainment, Sport, Technology) using context-aware keyword extraction. We improve the open-source LLM's performance by RAG.

Before running the codes, some modules must be installed first: torch, transformers, threading, faiss, pymupdf, pandas and sklearn. And the Hugging Face token must be input to run the chatbot. You can change the categories by prompts.

File Name and Explanation:
(1) bbc_text_cls.csv and NewsCategorizer.csv: files contain thousands of news text for retrieval-augmented generation
(2) LLM-Project-github.py: The codes for running the main program
