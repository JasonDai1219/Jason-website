import { ValidSkills } from "./constants";

export interface ExperienceInterface {
  id: string;
  position: string;
  company: string;
  location: string;
  startDate: Date;
  endDate: Date | "Present";
  description: string[];
  achievements: string[];
  skills: ValidSkills[];
  companyUrl?: string;
  logo?: string;

  // ✅ 方法结构图 / 流程图（放在 public 下面）
  diagram?: string;

  // ✅ 洋葱结构：网站叙事（diagram-aligned）
  narrative?: {
    headline?: string; // 第一层标题（扫一眼）
    subtitle?: string; // 第一层副标题（扫一眼）
    sections: { title: string; body: string[] }[]; // 第三层：分 topic 的旁白
  };
}

export const experiences: ExperienceInterface[] = [
  {
    id: "optic-lab",
    position: "Statistical ML Research Assistant",
    company: "OPTIC Lab, Perelman School of Medicine (UPenn)",
    location: "Philadelphia, PA",
    startDate: new Date("2025-11-01"),
    endDate: "Present",
    description: [
      "Build ML forecasting and validation pipelines for glaucoma progression modeling using longitudinal Humphrey Visual Field (HVF) data.",
      "Develop uncertainty-aware evaluation under dataset shift (e.g., label/distribution shift) to improve reliability for clinical decision support.",
      "Run cross-dataset generalization studies (UW → external cohort) with reproducible experiments and reporting.",
    ],
    achievements: [
      "Built trajectory construction and validation pipelines for multi-visit longitudinal HVF data, generating patient-level sequences for modeling and evaluation.",
      "Implemented uncertainty-aware coverage and calibration analysis under dataset and label shift with automated reporting.",
      "Standardized reproducible cross-cohort evaluation runs (internal → external cohorts) to benchmark robustness across settings.",
    ],
    skills: [
      "Python",
      "PyTorch",
      "Scikit-learn",
      "Model Evaluation",
      "Uncertainty Estimation",
      "Time Series",
    ],
    companyUrl: "https://www.med.upenn.edu/",
    logo: "/experience/optic/logo.png",
    // diagram: "/experience/optic/diagram.png",
    narrative: {
      headline:
        "Uncertainty-Aware Longitudinal Clinical Decision Modeling for Glaucoma Progression",

      subtitle:
        "Framing multi-visit patient trajectories as sequential decision processes to improve reliability under clinical distribution shift.",

      sections: [
        {
          title: "Project Overview (Context)",
          body: [
            "Glaucoma progression modeling requires reasoning over multi-year, irregularly sampled patient visits. Clinical data includes structured visual field measurements, treatment histories, and heterogeneous follow-up intervals.",
            "Standard forecasting models focus on point prediction accuracy, but clinical deployment requires calibrated uncertainty and robustness under dataset shift."
          ],
        },

        {
          title: "The Core Challenge",
          body: [
            "Longitudinal medical data is sparse, non-stationary, and influenced by treatment interventions that alter disease trajectories.",
            "Additionally, cross-cohort generalization is difficult due to label shift, missingness mechanisms, and evolving clinical practice patterns."
          ],
        },

        {
          title: "Modeling Framework",
          body: [
            "Patient histories are structured into trajectory representations, treating each visit as a sequential state within a decision process.",
            "Transformer-based sequence models are used to capture temporal dependencies across visual field progression and intervention signals.",
            "The modeling framework is designed to support both forecasting and downstream decision-oriented evaluation."
          ],
        },

        {
          title: "Uncertainty & Shift-Aware Evaluation",
          body: [
            "Evaluation incorporates conformal calibration techniques to quantify predictive reliability under distributional variation.",
            "Shift-aware validation workflows assess robustness across internal and external cohorts, focusing on coverage stability rather than raw accuracy."
          ],
        },

        {
          title: "Research Status",
          body: [
            "This is ongoing work focused on improving robustness and reproducibility in longitudinal clinical modeling.",
            "Current efforts emphasize reliable evaluation pipelines and systematic analysis of uncertainty under cohort shift."
          ],
        }
      ]
    }
  },

  {
    id: "hdsi",
    position: "Research Assistant",
    company: "Halıcıoglu Data Science Institute (UC San Diego)",
    location: "San Diego, CA",
    startDate: new Date("2024-06-01"),
    endDate: new Date("2025-09-01"),

    // 第一层（简洁一句话 Summary tab 看到的）
    description: [
      "Designed an AI-assisted data collection system that transforms GeoGuessr gameplay into structured multimodal reasoning traces for improving vision-language model grounding."
    ],

    achievements: [
      "Improved GPT-4o visual grounding performance by ~15% through structured human–AI data collection and evaluation pipelines.",
      "Built scalable agentic workflows for annotation, filtering, and automated benchmarking across interface conditions.",
      "Co–first author on CHI 2026 submission evaluating reasoning behavior under time constraints."
    ],

    skills: [
      "Python",
      "LangChain",
      "NLP",
      "Multimodal Models",
      "Model Evaluation",
      "Human-Computer Interaction",
      "Data Pipelines",
    ],

    companyUrl: "https://datascience.ucsd.edu/",
    logo: "/experience/hdsi/logo.png",

    // 第二层（视觉核心）
    diagram: "/experience/hdsi/diagram.png",

    // 第三层（洋葱展开 narrative）
    narrative: {
      headline:
        "GeoGuessr-Based Human–AI Reasoning Data Pipeline for Vision-Language Alignment",

      subtitle:
        "Turned interactive gameplay into structured reasoning data to improve model grounding under realistic cognitive constraints.",

      sections: [
        {
          title: "Project Overview (Context)",
          body: [
            "Vision-language models often struggle with fine-grained geographic cues such as road markings, vegetation, signage language, and architectural style. Improving alignment requires more than labels — it requires capturing the reasoning process behind predictions.",
            "Traditional annotation pipelines collect answers, but rarely capture how humans arrive at those answers."
          ],
        },

        {
          title: "The Core Challenge",
          body: [
            "Large-scale annotation is expensive and often lacks transparency in reasoning quality. Additionally, reasoning depth varies significantly under different cognitive constraints such as time pressure.",
            "The challenge was not only collecting data, but designing a system that produces structured, reproducible reasoning traces at scale."
          ],
        },

        {
          title: "System Design (Read Diagram Left → Right)",
          body: [
            "The system embeds structured interaction prompts directly into GeoGuessr gameplay, guiding players to provide evidence, explanation, and decision signals as part of natural play.",
            "Each session captures image evidence, textual justification, interaction metadata (zoom, search, timing), and final prediction — forming a multimodal reasoning record.",
            "These structured traces are stored, filtered, and processed through automated pipelines to support downstream fine-tuning and evaluation."
          ],
        },

        {
          title: "Deep Dive: Two Design Decisions",
          body: [
            "Behavior-first data design: instead of labeling static images, the system captures temporal interaction signals, enabling analysis of reasoning paths rather than just outcomes.",
            "Time-pressure as experimental variable: study conditions varied time limits to analyze how reasoning quality shifts between intuitive and evidence-driven decision modes."
          ],
        },

        {
          title: "Evaluation & Impact",
          body: [
            "Structured reasoning traces improved GPT-4o visual grounding accuracy by ~15% on held-out evaluation tasks.",
            "Automated evaluation scripts enabled systematic comparison across UI variants and time constraints, turning interaction design into a measurable experimental factor."
          ],
        },

        {
          title: "Reflection",
          body: [
            "One limitation we observed is that outcome correctness does not always imply reasoning quality. Some players arrive at correct guesses with shallow or incomplete explanations, which can distort supervision signals if not filtered carefully.",
            "If rebuilding today, I would introduce reasoning-quality scoring and consistency checks (e.g., explanation–evidence alignment, cross-round stability) to distinguish genuine inference from lucky guesses or low-effort annotations.",
            "I would also formalize lightweight data validation mechanisms to detect adversarial or malicious inputs before they propagate into fine-tuning pipelines."
          ],
        }
      ],
    },
  },

  {
    id: "china-asset-management",
    position: "Data Scientist Intern",
    company: "China Asset Management",
    location: "Beijing, China",
    startDate: new Date("2024-07-01"),
    endDate: new Date("2024-09-01"),
    // ✅ 这里给 Timeline / card 用：保持简洁（不要塞太长）
    description: [
      "Reframed ESG from a static score into a time-aware risk signal by designing behavior features robust to short, missing, industry-heterogeneous histories.",
      "Built an end-to-end feature + modeling pipeline with time-aware validation to avoid optimistic random splits.",
      "Translated model outputs into stakeholder-facing dashboards for investment decision support.",
    ],
    achievements: [
      "Improved classification performance to F1 = 0.83 (+28%) through representation redesign rather than increasing model complexity.",
      "Stabilized cross-year performance under time-aware splits, enabling more reliable risk signal interpretation.",
      "Delivered stakeholder-facing dashboards translating model outputs into actionable investment insights.",
    ],
    skills: [
      "Python",
      "SQL",
      "Pandas",
      "Scikit-learn",
      "Tableau",
      "Model Evaluation",
    ],
    companyUrl: "https://www.chinaamc.com/",
    logo: "/experience/chinaamc/logo.png",

    // ✅ 你 Canva 导出的图放到：public/experience/chinaamc/diagram.png
    diagram: "/experience/chinaamc/diagram.png",

    // ✅ 洋葱结构（只在 Experience detail 的 Summary 里展示）
    narrative: {
      headline: "ESG Risk Modeling (Representation-First)",
      subtitle:
        "Turned short, missing, and industry-heterogeneous ESG histories into time-aware risk signals via behavior features + realistic validation.",
      sections: [
        {
          title: "Project Overview",
          body: [
            "The team evaluated whether ESG data could improve company-level risk identification. In practice, ESG histories were short (often 5–10 years), irregularly missing, and not directly comparable across industries.",
            "A naive baseline (raw ESG levels + a stronger model) did not hold up under time-aware splits, so the focus shifted to representation design under real constraints.",
          ],
        },
        {
          title: "Key Challenge",
          body: [
            "Short and incomplete time series made deeper temporal architectures brittle and overfit-prone. Industry heterogeneity also meant the same ESG level could indicate different behaviors across sectors.",
            "So the problem was reframed: instead of fitting static ESG scores, extract stable behavior signals that survive forward-looking validation.",
          ],
        },
        {
          title: "Design Rationale (Read Diagram Left → Right)",
          body: [
            "The pipeline aligns company-year records onto a consistent time axis and treats missingness as a first-class issue to avoid “false stability” that would understate risk.",
            "ESG is represented as behavior: volatility captures instability, trend captures directionality, and industry-percentile encodings enable within-sector comparability.",
            "These features form a structured tabular representation that supports stable gradient boosting while remaining interpretable to stakeholders.",
          ],
        },
        {
          title: "Deep Dive",
          body: [
            "Representation > model capacity: with limited temporal depth, robust behavior features were more reliable than heavier sequential models.",
            "Validation is built-in: time-aware splits were treated as part of model development (not an afterthought), reflecting realistic deployment.",
          ],
        },
        {
          title: "Outcome",
          body: [
            "The representation redesign improved performance to F1 = 0.83 (+28%) and stabilized cross-year behavior under time-aware evaluation.",
            "Outputs were translated into dashboards/summaries so investment teams could interpret risk signals and act on them.",
          ],
        },
        {
          title: "Reflection",
          body: [
            "Next, I’d add explicit monitoring for distribution shift across industries/years and formalize calibration/uncertainty reporting for borderline cases.",
          ],
        },
      ],
    },
  },

  {
    id: "cec",
    position: "Machine Learning Intern",
    company: "China Electronics Corporation",
    location: "Beijing, China",
    startDate: new Date("2023-07-01"),
    endDate: new Date("2023-09-01"),

    // 🧅 第一层（Summary Tab 简短版本）
    description: [
      "Designed and productionized a multi-signal anomaly detection system for enterprise messaging and access logs under sparse labels and operational noise constraints.",
      "Focused on behavioral representation, score fusion, and alert governance rather than single-model optimization.",
      "Integrated the pipeline into real troubleshooting workflows with prioritization, deduplication, and monitoring support.",
    ],

    achievements: [
      "Achieved anomaly detection performance of F1 = 0.80 with automated orchestration workflows.",
      "Reduced alert noise through multi-signal fusion and cooldown-based suppression logic.",
      "Enabled faster triage by converting raw anomaly scores into prioritized, explainable alerts.",
    ],

    skills: [
      "Python",
      "Spark",
      "Airflow",
      "ETL",
      "Anomaly Detection",
      "Data Pipelines",
    ],

    companyUrl: "https://www.cec.com.cn/",
    logo: "/experience/cec/logo.png",

    // 🖼 放你刚画的结构图
    diagram: "/experience/cec/diagram.png",

    // 🧅 第二层（展开 narrative）
    narrative: {
      headline:
        "China Electronics Corporation — Multi-Signal Enterprise Anomaly Detection",

      subtitle:
        "From noisy operational logs to prioritized security alerts via behavioral aggregation, model fusion, and alert governance.",

      sections: [
        {
          title: "Project Overview (Context)",
          body: [
            "The security team needed to detect abnormal user behavior from large-scale enterprise messaging and access logs. Labels were sparse, behavior patterns were highly user-specific, and raw logs were noisy and inconsistent.",
            "The goal was not just to flag anomalies, but to produce reliable, triage-ready alerts that could support operational decision-making.",
          ],
        },

        {
          title: "The Core Challenge",
          body: [
            "Single-model anomaly detection was insufficient. Global outliers, structural pattern anomalies, and user-specific temporal drift required different detection mechanisms.",
            "Additionally, raw anomaly scores alone were not usable by operations — they needed prioritization, suppression logic, and context.",
          ],
        },

        {
          title: "Why This System Design (Read the Diagram Top → Bottom)",
          body: [
            "Logs were ingested in scheduled batches and normalized through schema alignment, deduplication, and data quality validation. Clean data was aggregated into user–asset time windows to capture behavioral signatures.",
            "Multiple detectors were deployed in parallel: DBSCAN for rare global outliers, Autoencoders for structural inconsistencies, and LSTM models for behavioral drift over time.",
            "Outputs were fused in a centralized scoring layer where signals were normalized, combined, and mapped into severity tiers (P0 / P1 / P2) before entering the alert processing layer.",
          ],
        },

        {
          title: "Deep Dive: Two Design Decisions",
          body: [
            "Multi-signal over single model: rather than relying on a single anomaly detector, we explicitly separated detection responsibilities to reduce blind spots across anomaly types.",
            "Operational-first design: alert deduplication, cooldown suppression, and contextual enrichment were treated as first-class system components, not post-processing hacks.",
          ],
        },

        {
          title: "Outcome",
          body: [
            "The system achieved F1 = 0.80 on labeled subsets while significantly reducing alert spam through risk-tier assignment and suppression logic.",
            "Security analysts could inspect ranked alerts, drill into behavior timelines, and provide feedback, closing the loop between modeling and operations.",
          ],
        },

        {
          title: "Reflection",
          body: [
            "If rebuilding today, I would make the sequence modeling more explicit: use a Transformer-style encoder over time-window events with masking for missingness, then calibrate and monitor the risk score distribution online. I’d also separate offline training (representation + calibration) from online scoring (low-latency inference + alert governance) with drift dashboards tied to retraining triggers.",
          ],
        },
      ],
    },
  }
];