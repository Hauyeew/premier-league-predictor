import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { MatchCard } from "./components/MatchCard";
import type { Prediction } from "./types";
import "./App.css";

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/predictions.csv")
      .then((response) => response.text())
      .then((csv) => {
        const lines = csv.trim().split("\n");

        const headers = lines[0].split(",");

        const rows: Prediction[] = lines
          .slice(1)
          .map((line) => {
            const values = line.split(",");

            const row: Record<string, string> = {};

            headers.forEach((header, index) => {
              row[header] = values[index];
            });

            return {
              Date: row.Date,
              HomeTeam: row.HomeTeam,
              AwayTeam: row.AwayTeam,

              Probability_H: Number(
                row.Probability_H
              ),

              Probability_D: Number(
                row.Probability_D
              ),

              Probability_A: Number(
                row.Probability_A
              ),

              Prediction: row.Prediction as
                | "H"
                | "D"
                | "A",
            };
          });

        setPredictions(rows);
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