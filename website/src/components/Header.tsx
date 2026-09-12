export function Header() {
    return (
      <header className="header">
        <div className="header-content">
          <div>
            <h1>PL Predictor</h1>
            <p>Premier League Match Predictions</p>
          </div>
  
          <div className="model-status">
            <span className="status-dot" />
            Model Online
          </div>
        </div>
      </header>
    );
  }