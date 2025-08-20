<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  
</head>
<body>

<h1>Customer Support Chatbot with RAG & Emotion Intensity Detection</h1>

<hr />

<p>This project builds an AI-powered customer support chatbot with these features:</p>

<ul>
  <li><strong>Retrieval-Augmented Generation (RAG):</strong> Finds the best answers from a knowledge base of past questions and answers. This ensures accurate and relevant replies.</li>
  <li><strong>Emotion Intensity Detection:</strong> Analyzes messages to detect key emotions like anger, joy, sadness, fear, and surprise, and measures how strong they are.</li>
  <li><strong>Alert System:</strong> When emotions like anger or frustration are very strong, it triggers an alert and sends an email to a human support agent using contacts stored in the database.</li>
  <li><strong>Seamless Integration:</strong> Both RAG and emotion detection work together via one backend API. The chatbot gives helpful answers with emotional context.</li>
  <li><strong>User-Friendly Interface:</strong> A Streamlit frontend where users chat naturally, and both the bot’s reply and detected emotions are shown clearly.</li>
</ul>


<h2>Why Emotion Detection in Customer Support?</h2>

<p>In customer support, understanding not just <em>what</em> a customer says, but <em>how</em> they feel, is critical. Emotions such as frustration, anger, or sadness can signal urgent issues or dissatisfied customers needing immediate attention. Detecting emotions in real-time helps businesses:</p>

<ul>
  <li><strong>Improve customer satisfaction:</strong> By identifying unhappy or frustrated customers early, support agents can prioritize and tailor their responses to calm and solve issues faster.</li>
  <li><strong>Enhance agent efficiency:</strong> Emotion insights allow automatic alerts for high-intensity emotional messages, enabling timely human intervention or escalation.</li>
  <li><strong>Gain actionable insights:</strong> Aggregated emotion data reveals trends about product issues or service quality, guiding improvements.</li>
  <li><strong>Personalize user experience:</strong> Emotion-aware chatbots can adjust tone or responses to better empathize with customers, creating a more human-like support experience.</li>
  <li><strong>Enable alert systems:</strong> Automated alerts notify human agents when customer emotions reach critical intensity, ensuring urgent issues get immediate attention.</li>
</ul>

<hr />

<h2>Project Structure</h2>
<pre><code>customer_support/
├── backend/
│   ├── rag_module.py          # RAG retrieval logic
│   ├── emotion_module.py      # Emotion and intensity detection
│   ├── main.py                # FastAPI backend server
│   ├── preprocess_data.py     # Data preprocessing script
│   ├── build_index.py         # Script to build FAISS index
│   └── data/
│       ├── Bitext_Sample_Customer_Support.csv  # Sample customer support data (assuming)
│       ├── customer_support.index               # FAISS index file
│       ├── customer_support_cleaned_data.csv    # Cleaned data file
│       └── customer_support_df.pkl               # Knowledge base DataFrame
├── frontend/
│   └── app.py                 # Streamlit frontend app
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
</code></pre>


<hr />

<h2>Setup Instructions</h2>

<h3>1. Clone the repository</h3>
<pre><code>git clone https://github.com/your-username/customer_support_chatbot.git
cd customer_support_chatbot
</code></pre>

<h3>2. Create and activate a virtual environment (optional but recommended)</h3>
<pre><code>python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
</code></pre>

<h3>3. Install dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3>4. Prepare the data</h3>
<ul>
  <li>Ensure your FAISS index file (<code>customer_support.index</code>) and knowledge base DataFrame (<code>customer_support_df.pkl</code>) are located inside <code>backend/data/</code>.</li>
  <li>These files contain the indexed customer support Q&amp;A pairs used by the retrieval module.</li>
</ul>

<h3>5. Run the FastAPI backend</h3>
<pre><code>uvicorn backend.main:app --reload
</code></pre>
<p>The backend will start on <code>http://127.0.0.1:8000</code>.</p>

<h3>6. Run the Streamlit frontend</h3>
<pre><code>streamlit run frontend/app.py
</code></pre>
<p>The frontend UI will open in your browser where you can chat with the bot.</p>

<hr />

<h2>Usage</h2>
<ul>
  <li>Type your customer support query in the input box.</li>
  <li>Click "Send" to get an answer from the RAG system.</li>
  <li>The detected emotion intensities of your query will also be displayed.</li>
  <li>New messages appear at the top of the chat window.</li>
</ul>

<hr />

<h2>Model Details</h2>
<ul>
  <li><strong>RAG Embeddings:</strong> Uses <code>msmarco-distilbert-base-v4</code> Sentence Transformer model for embedding queries and knowledge base documents.</li>
  <li><strong>Emotion Detection:</strong> Uses a fine-tuned BERT model for multi-label emotion intensity prediction.</li>
  <li><strong>FAISS:</strong> Efficient vector similarity search on the knowledge base.</li>
</ul>

<hr />

<h2>License</h2>
<p>This project is licensed under the MIT License.</p>

<hr />

<h2>Contact</h2>
<p>For questions or contributions, please contact <a href="mailto:rachitguptacse.098@gmail.com">rachitguptacse.098@gmail.com</a>.</p>

</body>
</html>
