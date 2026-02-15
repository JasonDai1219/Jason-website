import { Icons } from "@/components/common/icons";

export interface skillsInterface {
  name: string;
  description: string;
  rating: number;
  icon: any;
}

export const skillsUnsorted: skillsInterface[] = [
  {
    name: "Python",
    description:
      "Data processing, machine learning, and end-to-end experimentation pipelines.",
    rating: 5,
    icon: Icons.python,
  },
  {
    name: "SQL",
    description:
      "Analytical querying and performance-minded database work (indexing, query rewrites).",
    rating: 5,
    icon: Icons.sql,
  },
  {
    name: "PyTorch",
    description:
      "Training and evaluating deep learning models for forecasting and multimodal learning.",
    rating: 5,
    icon: Icons.pytorch,
  },
  {
    name: "Machine Learning",
    description:
      "Supervised/unsupervised learning, time series forecasting, and anomaly detection.",
    rating: 5,
    icon: Icons.machineLearning,
  },
  {
    name: "Data Pipelines",
    description:
      "ETL and orchestration workflows for large-scale data processing and model-ready datasets.",
    rating: 4,
    icon: Icons.dataPipeline,
  },
  {
    name: "Spark",
    description:
      "Distributed data processing for ETL and large-scale feature generation.",
    rating: 4,
    icon: Icons.spark,
  },
  {
    name: "Airflow",
    description:
      "Workflow orchestration for scheduled pipelines and automated jobs.",
    rating: 4,
    icon: Icons.airflow,
  },
  {
    name: "PostgreSQL",
    description:
      "Relational data modeling and query optimization for application and analytics workloads.",
    rating: 4,
    icon: Icons.postgresql,
  },
  {
    name: "AWS",
    description: "Cloud infrastructure for data storage, compute, and deployment.",
    rating: 3,
    icon: Icons.amazonaws,
  },
  {
    name: "Docker",
    description: "Containerizing workflows for reproducibility and deployment.",
    rating: 3,
    icon: Icons.docker,
  },
  {
    name: "Kubernetes",
    description:
      "Container orchestration concepts for scalable services and deployments.",
    rating: 3,
    icon: Icons.kubernetes,
  },
  {
    name: "TypeScript",
    description:
      "Type-safe front-end development for full-stack data applications.",
    rating: 3,
    icon: Icons.typescript,
  },
  {
    name: "JavaScript",
    description:
      "Building interactive data applications and visualization-heavy UIs when needed.",
    rating: 3,
    icon: Icons.javascript,
  },
  {
    name: "React",
    description: "Interactive UI for data products and dashboards.",
    rating: 3,
    icon: Icons.react,
  },
  {
    name: "Next.js",
    description: "Full-stack framework for shipping data-driven web apps.",
    rating: 3,
    icon: Icons.nextjs,
  },
  {
    name: "Tailwind CSS",
    description: "Rapid UI styling for clean, responsive interfaces.",
    rating: 3,
    icon: Icons.tailwindcss,
  },
];

export const skills = skillsUnsorted
  .slice()
  .sort((a, b) => b.rating - a.rating);

export const featuredSkills = skills.slice(0, 6);
