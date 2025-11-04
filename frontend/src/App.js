import React, { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [reason, setReason] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text.trim()) {
      alert("Please enter a news article first!");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data.prediction);
      setReason(data.reason);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to connect to backend!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🧠 Fake News Detector</h1>
      <textarea
        placeholder="Paste or type a news article here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      ></textarea>

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Detecting..." : "Detect Fake News"}
      </button>

      {result && (
        <div className="result">
          <h2 className={result.includes("Fake") ? "fake" : "real"}>
            {result}
          </h2>
          <p>{reason}</p>
        </div>
      )}
    </div>
  );
}

export default App;
