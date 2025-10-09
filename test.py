import google.generativeai as genai

# 🔑 Paste your Gemini API key here
API_KEY = "AIzaSyBeLf_qcAilC-_Ljqk5Pqd9R86cmsesmZ8"

# Configure the Gemini API
genai.configure(api_key=API_KEY)

# Load the Gemini model (you can use gemini-1.5-flash or gemini-1.5-pro)
model = genai.GenerativeModel("gemini-2.5-flash")

# Example papers to summarize
papers = [
    {
        "title": "Optimization for Machine Learning",
        "authors": "Elad Hazan",
        "text": """Optimization lies at the core of machine learning. 
        This book provides a unified framework connecting convex optimization, online learning, 
        and stochastic methods with applications to modern ML problems."""
    },
    {
        "title": "Minimax deviation strategies for machine learning and recognition with short learning samples",
        "authors": "Michail Schlesinger, Evgeniy Vodolazskiy",
        "text": """This paper introduces a minimax approach for classification with small datasets, 
        ensuring robustness against sample uncertainty and offering a framework for recognition 
        under limited training conditions."""
    },
    {
        "title": "An Optimal Control View of Adversarial Machine Learning",
        "authors": "Xiaojin Zhu",
        "text": """This paper formulates adversarial machine learning through the lens of optimal control, 
        providing insights into dynamic interactions between models and adversaries."""
    }
]

# Generate summaries
for paper in papers:
    print(f"## {paper['title']}")
    print(f"**Authors**: {paper['authors']}")
    try:
        prompt = f"Summarize this paper in 3–5 bullet points:\n\n{paper['text']}"
        response = model.generate_content(prompt)
        print(f"**Summary**:\n{response.text}\n")
    except Exception as e:
        print(f"**Summary**: Error generating summary. ({e})\n")
