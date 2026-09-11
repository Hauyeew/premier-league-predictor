import type { Prediction } from "../types";
import { ProbabilityBar } from "./ProbabilityBar";

type MatchCardProps = {
  match: Prediction;
};

function getPredictionText(prediction: "H" | "D" | "A") {
  if (prediction === "H") {
    return "Home Win";
  }

  if (prediction === "D") {
    return "Draw";
  }

  return "Away Win";
}

export function MatchCard({ match }: MatchCardProps) {
  const date = new Date(match.Date);

  const formattedDate = date.toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });

  return (
    <div className="match-card">
      <div className="match-header">
        <span>{formattedDate}</span>
        <span className="prediction-badge">
          {getPredictionText(match.Prediction)}
        </span>
      </div>

      <div className="teams">
        <div className="team">
          <span className="team-name">
            {match.HomeTeam}
          </span>

          <span className="venue">
            HOME
          </span>
        </div>

        <div className="vs">
          VS
        </div>

        <div className="team">
          <span className="team-name">
            {match.AwayTeam}
          </span>

          <span className="venue">
            AWAY
          </span>
        </div>
      </div>

      <div className="probabilities">
        <ProbabilityBar
          label="Home"
          probability={match.Probability_H}
        />

        <ProbabilityBar
          label="Draw"
          probability={match.Probability_D}
        />

        <ProbabilityBar
          label="Away"
          probability={match.Probability_A}
        />
      </div>

      <div className="prediction">
        <span>Model prediction</span>

        <strong>
          {getPredictionText(match.Prediction)}
        </strong>
      </div>
    </div>
  );
}