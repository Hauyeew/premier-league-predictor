import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { MatchCard } from "./components/MatchCard";
import type { Prediction } from "./types";
import "./App.css";

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

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
        console.error(
          "Failed to load predictions:",
          error
        );
  
        setLoading(false);
      });
  }, []);

  return (
    <div className="app">
      <Header />

      <main className="main">
        <section className="hero">
          <div>
            <p className="eyebrow">
              NEXT MATCHWEEK
            </p>

            <h2>
              Premier League Predictions
            </h2>
          </div>

          <div className="match-count">
            <strong>
              {predictions.length}
            </strong>

            <span>Matches</span>
          </div>
        </section>

        {loading ? (
          <div className="loading">
            Loading predictions...
          </div>
        ) : predictions.length === 0 ? (
          <div className="empty">
            No upcoming predictions found.
          </div>
        ) : (
          <section className="matches">
            {predictions.map((match, index) => (
              <MatchCard
                key={`${match.HomeTeam}-${match.AwayTeam}-${index}`}
                match={match}
              />
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;