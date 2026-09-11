type ProbabilityBarProps = {
    label: string;
    probability: number;
  };
  
  export function ProbabilityBar({
    label,
    probability,
  }: ProbabilityBarProps) {
    const percentage = probability * 100;
  
    return (
      <div className="probability-row">
        <div className="probability-label">
          <span>{label}</span>
          <span>{percentage.toFixed(1)}%</span>
        </div>
  
        <div className="probability-track">
          <div
            className="probability-fill"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  }