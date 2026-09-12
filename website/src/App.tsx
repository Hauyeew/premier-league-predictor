import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { MatchCard } from "./components/MatchCard";
import type { Prediction } from "./types";
import "./App.css";

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/predictions")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch predictions");
        }

        return response.json();
      })
      .then((data: Prediction[]) => {
        setPredictions(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Failed to load predictions:", error);
        setError("Unable to load predictions.");
        setLoading(false);
      });
  }, []);

  return (
    <>
      <Header />

      <main className="container">
        <section className="hero">
          <h2>Premier League Predictions</h2>

          <p>
            AI-powered predictions for the upcoming Premier League
            fixtures.
          </p>

          {!loading && !error && (
            <div className="match-count">
              {predictions.length} upcoming matches
            </div>
          )}
        </section>

        {loading && (
          <div className="loading">
            Loading predictions...
          </div>
        )}

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {!loading && !error && (
          <section className="matches-grid">
            {predictions.map((match, index) => (
              <MatchCard
                key={`${match.Date}-${match.HomeTeam}-${match.AwayTeam}-${index}`}
                match={match}
              />
            ))}
          </section>
        )}
      </main>
    </>
  );
}

export default App;