export type Prediction = {
    Date: string;
    HomeTeam: string;
    AwayTeam: string;
  
    Probability_H: number;
    Probability_D: number;
    Probability_A: number;
  
    Prediction: "H" | "D" | "A";
  };