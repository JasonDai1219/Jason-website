export type ValidSkills =
  | "Python"
  | "SQL"
  | "JavaScript"
  | "TypeScript"
  | "PyTorch"
  | "TensorFlow"
  | "Scikit-learn"
  | "NumPy"
  | "Pandas"
  | "Spark"
  | "Airflow"
  | "MLflow"
  | "LangChain"
  | "FastAPI"
  | "Flask"
  | "PostgreSQL"
  | "MySQL"
  | "MongoDB"
  | "Redis"
  | "AWS"
  | "Docker"
  | "Kubernetes"
  | "Git"
  | "Linux"
  | "REST APIs"
  | "Distributed Systems"
  | "Data Pipelines"
  | "ETL"
  | "Feature Engineering"
  | "Model Evaluation"
  | "Uncertainty Estimation"
  | "Time Series"
  | "Anomaly Detection"
  | "Computer Vision"
  | "NLP"
  | "Multimodal Models"
  | "Transformers"
  | "Diffusion Models"
  | "React"
  | "Next.js"
  | "Tailwind CSS"
  | "Vercel"
  | "Human-Computer Interaction"
  | "Conformal Prediction"
  | "Causal Inference"
  | "Statistical Modeling"
  | "Healthcare AI"
  | "Experiment Design"
  | "A/B Testing"
  | "Recommender Systems"
  | "Graph Neural Networks"
  | "Reinforcement Learning"
  | "Federated Learning"
  | "AutoML"
  | "MLOps"
  | "Data Visualization"
  | "Big Data"
  | "Cloud Computing"
  | "Software Engineering"
  | "Agile Methodologies"
  | "Project Management"
  | "Tableau";

// =============================
// Pages (for pagesConfig typing)
// =============================
export const ValidPages = [
  "home",
  "projects",
  "experience",
  "skills",
  "contributions",
  "resume",
  "contact",
] as const;

export type ValidPages = (typeof ValidPages)[number];