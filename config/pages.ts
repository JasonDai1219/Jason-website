import { ValidPages } from "./constants";

type PagesConfig = {
  [key in ValidPages]: {
    title: string;
    description: string;
    metadata: {
      title: string;
      description: string;
    };
  };
};

export const pagesConfig: PagesConfig = {
  home: {
    title: "Home",
    description:
      "Personal portfolio of Ruizhe (Jason) Dai, focused on data science, machine learning, and AI-assisted systems.",
    metadata: {
      title: "Ruizhe (Jason) Dai",
      description:
        "Ruizhe (Jason) Dai is a Data Science and Machine Learning engineer building data-driven systems and AI-assisted tools.",
    },
  },
  skills: {
    title: "Skills",
    description:
      "Core technical skills across data science, machine learning, and ML systems.",
    metadata: {
      title: "Skills | Ruizhe (Jason) Dai",
      description:
        "Technical skills in data science, machine learning, ML engineering, and AI systems.",
    },
  },
  projects: {
    title: "Projects",
    description:
      "Selected projects showcasing machine learning systems, data pipelines, and applied AI research.",
    metadata: {
      title: "Projects | Ruizhe (Jason) Dai",
      description:
        "Projects in machine learning, data science, AI-assisted systems, and applied research by Ruizhe (Jason) Dai.",
    },
  },
  experience: {
    title: "Experience",
    description:
      "Research and industry experience in data science, machine learning, and AI systems.",
    metadata: {
      title: "Experience | Ruizhe (Jason) Dai",
      description:
        "Professional and research experience in data science, machine learning, and AI systems.",
    },
  },
  contributions: {
    title: "Contributions",
    description:
      "Research, systems, and open contributions across AI, data, and machine learning.",
    metadata: {
      title: "Contributions | Ruizhe (Jason) Dai",
      description:
        "Research and system contributions in AI, machine learning, and data-driven systems.",
    },
  },
  contact: {
    title: "Contact",
    description:
      "Get in touch to discuss research, internships, and collaboration opportunities.",
    metadata: {
      title: "Contact | Ruizhe (Jason) Dai",
      description:
        "Contact Ruizhe (Jason) Dai for research collaboration and professional opportunities.",
    },
  },
  resume: {
    title: "Resume",
    description: "Resume of Ruizhe (Jason) Dai.",
    metadata: {
      title: "Resume | Ruizhe (Jason) Dai",
      description:
        "Resume of Ruizhe (Jason) Dai, Data Science and Machine Learning engineer.",
    },
  },
};
