// config/projects.ts
import { ValidSkills } from "./constants";

export type ProjectType = "Personal" | "Academic";

export interface ProjectPageInfo {
  title: string;
  description: string;
  imgArr: string[];
}

export interface ProjectDescriptionDetails {
  paragraphs: string[];
  bullets: string[];
}

export interface ProjectInterface {
  id: string;
  companyName: string;
  type: ProjectType;
  category: string[];
  shortDescription: string;

  websiteLink?: string; // empty string means "no link"
  githubLink?: string;

  techStack: (ValidSkills | string)[];
  startDate: Date;
  endDate: Date;

  companyLogoImg: string;

  pagesInfoArr: ProjectPageInfo[];
  descriptionDetails: ProjectDescriptionDetails;
}

/* ============================= */
/* ✅ Projects Data */
/* ============================= */
export const Projects: ProjectInterface[] = [
  {
    id: "time-machine-earth",
    companyName: "Earth Time Machine",
    type: "Personal",
    category: ["Data Engineering", "Frontend", "AI Systems"],
    shortDescription:
      "Built an interactive spatiotemporal analytics system with a 3D globe interface and a relational (ER) schema; optimized worst-case queries from 15m23s → 1.552s and 39s → 1.628s via indexing, caching, and query rewrites.",
    websiteLink: "", // TODO: add demo link when ready
    githubLink: "https://github.com/JasonDai1219/Earth-Time-Machine",
    techStack: ["React", "TypeScript", "PostgreSQL", "SQL", "Python", "Node.js"],
    startDate: new Date("2024-11-01"),
    endDate: new Date("2024-12-01"),
    companyLogoImg: "/projects/time-machine-earth/logo.png",
    pagesInfoArr: [
      {
        title: "Global 3D Globe Interface",
        description:
          "Interactive 3D globe for country-level exploration across time and development indicators.",
        imgArr: ["/projects/time-machine-earth/globe.png"],
      },
      {
        title: "Relational Schema (ER Diagram)",
        description:
          "ER design connecting countries, indicators, years, and media assets to support join-heavy spatiotemporal analytics.",
        imgArr: ["/projects/time-machine-earth/er_diagram.png"],
      },
      {
        title: "Query Performance Benchmarks",
        description:
          "Pre/post optimization timing results across four complex analytical queries.",
        imgArr: [
          "/projects/time-machine-earth/perf_q2.png",
          "/projects/time-machine-earth/perf_q4.png",
          "/projects/time-machine-earth/perf_q1.png",
          "/projects/time-machine-earth/perf_q3.png",
        ],
      },
    ],
    descriptionDetails: {
      paragraphs: [
        "Earth Time Machine is a spatiotemporal data visualization system for exploring country-level change over time through an interactive 3D globe and analytical queries.",
        "I designed a relational schema (ER-based) to support join-heavy analytics across countries, years, indicators, and media assets, enabling flexible filtering and aggregation over time.",
        "To keep exploration responsive, I optimized multiple complex SQL queries using indexing strategies, caching, and query rewrites, with the largest improvements reducing runtime from 15m23s → 1.552s and 39s → 1.628s.",
        "Benchmark results across four representative queries are included to show both worst-case and steady-state performance gains.",
      ],
      bullets: [
        "Built a full-stack spatiotemporal analytics platform with a 3D globe UI for exploratory analysis.",
        "Designed an ER-based relational schema to support join-heavy queries across countries, years, indicators, and media assets.",
        "Optimized worst-case analytical queries from 15m23s → 1.552s and 39s → 1.628s via indexing, caching, and query rewrites.",
        "Reported pre/post benchmarks across four complex queries to validate performance improvements across query patterns.",
      ],
    },
  },

  {
    id: "foodseg-unet-vgg16",
    companyName: "Beyond UNet: Food Image Semantic Segmentation (FoodSeg103)",
    type: "Academic",
    category: ["Machine Learning", "Computer Vision", "Research"],
    shortDescription:
      "Improved fine-grained food segmentation on FoodSeg103: baseline CNN mIoU 0.0588 → UNet 0.1217 → VGG16 encoder + decoder 0.1559, showing encoder strength dominates performance.",
    websiteLink: "", // TODO: add demo link when ready
    githubLink: "https://github.com/JasonDai1219/Beyond_UNet",
    techStack: [
      "PyTorch",
      "Deep Learning",
      "Semantic Segmentation",
      "Computer Vision",
      "Python",
      "Matplotlib",
      "NumPy",
    ],
    startDate: new Date("2025-10-01"),
    endDate: new Date("2025-12-01"),
    companyLogoImg: "/projects/foodseg/logo.png",
    pagesInfoArr: [
      {
        title: "Dataset EDA: Image Resolution & Input Variability",
        description:
          "Distribution of image widths/heights shows large variance in input resolution, motivating robust preprocessing and resizing strategy.",
        imgArr: ["/projects/foodseg/eda_image_dimensions.png"],
      },
      {
        title: "Dataset EDA: Scene Complexity & Reflective Regions",
        description:
          "Many images contain multiple food categories, and reflective (high-brightness) regions are common—both make boundary learning harder.",
        imgArr: ["/projects/foodseg/eda_items_reflection.png"],
      },
      {
        title: "Dataset EDA: Long-tail Class Imbalance",
        description:
          "Pixel-count distribution across classes is highly imbalanced, indicating a long-tail learning challenge for rare categories.",
        imgArr: ["/projects/foodseg/eda_class_imbalance_top30.png"],
      },
      {
        title: "Baseline Model: CNN Training Curve",
        description:
          "Baseline CNN shows limited optimization progress, motivating a stronger segmentation architecture.",
        imgArr: ["/projects/foodseg/train_cnn_loss.png"],
      },
      {
        title: "UNet: Training Curve",
        description:
          "UNet improves learning dynamics and stability compared to the baseline CNN.",
        imgArr: ["/projects/foodseg/train_unet_loss.png"],
      },
      {
        title: "Ablation: Updated Loss + Weight Map (No Further Gain)",
        description:
          "We tested loss/weight-map design for boundaries and reflective regions; results did not outperform standard UNet, suggesting the bottleneck is not in decoder-side tweaks.",
        imgArr: ["/projects/foodseg/train_unet_loss_weightmap.png"],
      },
      {
        title: "Final Model: VGG16 Encoder + Decoder",
        description:
          "Replacing the encoder with a pretrained VGG16 backbone yields the strongest convergence behavior, supporting the conclusion that encoder representation is the key lever.",
        imgArr: ["/projects/foodseg/train_vgg16_decoder_loss.png"],
      },
    ],
    descriptionDetails: {
      paragraphs: [
        "This project studies fine-grained semantic segmentation on FoodSeg103 (103 food categories + background). The task is challenging due to irregular shapes, blurred boundaries, multi-food overlap, and a long-tailed class distribution.",
        "We started from a lightweight CNN encoder–decoder baseline and observed limited capacity. Switching to a UNet architecture substantially improved boundary quality and overall mIoU. We then tested loss/decoder-side modifications (e.g., weighted loss with boundary/reflection awareness; ASPP bottleneck), which did not provide additional gains.",
        "Finally, replacing the UNet encoder with a pretrained VGG16 backbone produced the best performance with much faster convergence, supporting the conclusion that the primary bottleneck is semantic feature extraction in the encoder rather than decoder tweaks or loss engineering.",
      ],
      bullets: [
        "Baseline CNN reached test mIoU = 0.0588; UNet improved to 0.1217; best model (VGG16 encoder + decoder) achieved 0.1559.",
        "Ran dataset EDA to quantify input-size variability, multi-class complexity per image, long-tail class imbalance, and reflective-area ratio.",
        "Tested UNet variants with composite loss/weight maps and ASPP; observed no improvement over standard UNet, suggesting decoder-side tweaks are not the main lever.",
        "Validated transfer learning: a deep pretrained encoder significantly improved segmentation quality and training efficiency.",
      ],
    },
  },

  {
    id: "causal-stock-return",
    companyName: "Causal Discovery in Stock Return",
    type: "Academic",
    category: ["Machine Learning", "Causal Inference", "Time Series"],
    shortDescription:
      "Built a hybrid stock-return forecasting system combining causal feature selection (PCMCI+/CD-NOD), DeepAR probabilistic forecasting, and FinBERT sentiment signals; added a fusion layer to integrate macro/micro indicators for more interpretable predictions.",
    websiteLink: "https://acai1031.github.io/DSC180B-Capstone-Website",
    githubLink: "https://github.com/VivianZhao12/CAPSTONE-stockreturn",
    techStack: [
      "Python",
      "PyTorch",
      "DeepAR",
      "FinBERT",
      "Random Forest",
      "PCMCI+",
      "CD-NOD",
      "FRED",
      "Yahoo Finance API",
    ],
    startDate: new Date("2024-01-01"),
    endDate: new Date("2024-03-01"),
    companyLogoImg: "/projects/causal-stock/open_bg.jpg",
    pagesInfoArr: [
      {
        title: "System Overview",
        description:
          "End-to-end framework integrating economic indicators, stock time series forecasting, and social sentiment into a fusion layer for final prediction.",
        imgArr: ["/projects/causal-stock/framework_overview.png"],
      },
      {
        title: "Economic Impact Module",
        description:
          "Pipeline to build an economic impact dataframe from macro indicators and company financial reports, then predict external shocks with regression + random forest.",
        imgArr: ["/projects/causal-stock/economic_module_overview.png"],
      },
      {
        title: "Sentiment Analysis Module (FinBERT)",
        description:
          "Workflow from Twitter/Reddit text collection to FinBERT inference, producing a daily sentiment score used as an external signal.",
        imgArr: [
          "/projects/causal-stock/sentiment_workflow.png",
          "/projects/causal-stock/sentiment_distribution.png",
        ],
      },
      {
        title: "Stock Return Forecasting (DeepAR)",
        description:
          "Probabilistic forecasting pipeline using a residually-enhanced LSTM encoder and distributional output for daily return prediction.",
        imgArr: ["/projects/causal-stock/deepar_pipeline.png"],
      },
      {
        title: "Causal Discovery & Interpretability",
        description:
          "PCMCI+ identifies causal predictor structure for stock returns; feature relationship graphs highlight influential drivers for selected tickers.",
        imgArr: [
          "/projects/causal-stock/pcmci_structure.png",
          "/projects/causal-stock/impactful_features_1.png",
          "/projects/causal-stock/impactful_features_2.png",
        ],
      },
      {
        title: "Fusion Layer",
        description:
          "Fusion layer integrates DeepAR predictions with forward-filled quarterly economic data using feature engineering + random forest and adaptive weighting.",
        imgArr: ["/projects/causal-stock/fusion_layer.png"],
      },
      {
        title: "Results & Model Comparison",
        description:
          "Quantitative comparison across tickers using RMSE, MAE, MAPE, and direction accuracy for DeepAR vs Fusion.",
        imgArr: ["/projects/causal-stock/perf_table.png"],
      },
    ],
    descriptionDetails: {
      paragraphs: [
        "Developed a hybrid stock-return prediction framework that integrates public sentiment, historical prices, and macro/microeconomic indicators to improve both predictive performance and interpretability.",
        "Applied PCMCI+ for causal feature selection in multivariate time series, and used CD-NOD to detect causal impacts under non-stationarity for economic indicators.",
        "Modeled return dynamics with an enhanced DeepAR probabilistic forecaster and incorporated FinBERT-based sentiment scores derived from social data; built a Random Forest fusion layer to adaptively blend signals depending on market conditions.",
      ],
      bullets: [
        "Integrated multi-source signals: Yahoo Finance price history, FRED macro indicators, and FinBERT-derived sentiment features",
        "Performed causal feature selection with PCMCI+ and CD-NOD to reduce spurious correlations and improve interpretability",
        "Implemented DeepAR probabilistic forecasting for daily returns and evaluated across multiple tickers",
        "Designed a fusion layer (Random Forest + adaptive blending) to combine DeepAR predictions with economic fundamentals",
      ],
    },
  },

  {
    id: "dsc102-ai-tutor",
    companyName: "AI-assisted Personalized Quiz Generator & Learning Tracker",
    type: "Academic",
    category: ["AI Systems", "Web Dev", "UI/UX"],
    shortDescription:
      "Built an LLM-assisted quiz generator for DSC102 with week- and concept-level customization, concept-based error tracking on AWS, and targeted resource recommendations to reduce answer over-reliance.",
    websiteLink: "https://www.youtube.com/watch?v=2krRz-4Fx3E",
    githubLink: "https://github.com/jeffyuan2022/DSC102_AI_Tutor",
    techStack: ["Python", "Streamlit", "AWS"],
    startDate: new Date("2024-10-01"),
    endDate: new Date("2024-12-01"),
    companyLogoImg: "/projects/dsc102-ai-tutor/logo.png",
    pagesInfoArr: [
      {
        title: "Personalized Quiz Generation",
        description:
          "Generate quizzes by week (Weeks 1–10) or by specific concepts, with configurable question counts.",
        imgArr: ["/projects/dsc102-ai-tutor/quiz-generation.png"],
      },
      {
        title: "Question History & Error Tracking",
        description:
          "Logs each quiz attempt and categorizes results as correct/incorrect for review.",
        imgArr: ["/projects/dsc102-ai-tutor/question-history.png"],
      },
      {
        title: "Concept-Level Error Visualization",
        description:
          "Aggregated error frequencies visualized at concept level to target weak areas.",
        imgArr: ["/projects/dsc102-ai-tutor/error-visualization.png"],
      },
      {
        title: "Targeted Resource Recommendations",
        description:
          "Retrieves curated external learning resources via the SHARP API.",
        imgArr: ["/projects/dsc102-ai-tutor/resource-recommendation.png"],
      },
    ],
    descriptionDetails: {
      paragraphs: [
        "DSC102 AI Tutor shifts the learning workflow toward active practice and self-assessment.",
        "The system reorganizes course materials into week-level and concept-level structures to generate tailored quizzes, and enforces answer submission before solution reveal.",
        "The app records concept-level error frequencies and quiz history per student on AWS, and surfaces external resources via SHARP API for targeted follow-up.",
      ],
      bullets: [
        "Built an LLM-assisted quiz generation and learning tracking tool with week-level and concept-based customization.",
        "Enforced answer submission before solution reveal.",
        "Implemented per-user concept-level error tracking and quiz history logging stored on AWS.",
        "Integrated SHARP API to recommend concept-relevant learning resources.",
      ],
    },
  },

  {
    id: "glaucoma-uncertainty-aware-diagnosis",
    companyName: "Uncertainty-Aware Glaucoma Diagnosis with Conformal Prediction",
    type: "Personal",
    category: ["Machine Learning", "AI Systems", "Healthcare AI"],
    shortDescription:
      "Built an uncertainty-aware medical classification system for glaucoma diagnosis using conformal prediction and an expert-in-the-loop decision pipeline, reducing false negatives in high-risk settings.",
    websiteLink: "",
    githubLink: "https://github.com/JasonDai1219/uncertainty-aware-medAI",
    techStack: [
      "Python",
      "Scikit-learn",
      "Conformal Prediction",
      "Machine Learning",
      "EHR Data Analysis",
    ],
    startDate: new Date("2024-11-01"),
    endDate: new Date("2024-12-01"),
    companyLogoImg: "/projects/glaucoma-cp/hero.png",
    pagesInfoArr: [
      {
        title: "Problem Motivation: Reducing False Negatives in Medical AI",
        description:
          "Reframes glaucoma diagnosis as a decision-safety problem rather than pure classification.",
        imgArr: ["/projects/glaucoma-cp/motivation.png"],
      },
      {
        title: "Baseline Modeling on Glaucoma EHR Data",
        description:
          "Trained a baseline classifier and evaluated AUROC/AUPRC and error profiles.",
        imgArr: ["/projects/glaucoma-cp/baseline-results.png"],
      },
      {
        title: "Conformal Prediction for Sample-Level Uncertainty",
        description:
          "Applied split conformal prediction to convert point predictions into prediction sets.",
        imgArr: ["/projects/glaucoma-cp/conformal-prediction.png"],
      },
      {
        title: "Expert-in-the-Loop Decision Pipeline",
        description:
          "Escalates uncertain cases to an expert to mirror real clinical workflows.",
        imgArr: ["/projects/glaucoma-cp/expert-loop.png"],
      },
      {
        title: "Quantifying False Negative Reduction",
        description:
          "Analyzed how thresholds and expert sensitivity control safety/workload tradeoffs.",
        imgArr: ["/projects/glaucoma-cp/fn-analysis.png"],
      },
    ],
    descriptionDetails: {
      paragraphs: [
        "This project explores uncertainty-aware decision-making for safer medical AI, using glaucoma diagnosis from EHR data as a case study.",
        "Instead of forcing predictions on ambiguous cases, conformal prediction supports abstention with statistical guarantees.",
        "Uncertain cases are escalated to an expert-in-the-loop, simulated via GPT-based judgments or parameterized expert performance.",
      ],
      bullets: [
        "Reframed glaucoma diagnosis as a decision-safety problem rather than a pure classification task.",
        "Applied split conformal prediction to obtain per-sample uncertainty guarantees.",
        "Designed an expert-in-the-loop escalation mechanism for uncertain cases.",
        "Analyzed false-negative reduction under expert intervention and conformal thresholds.",
      ],
    },
  },
];

export const featuredProjects = Projects.slice(0, 3);