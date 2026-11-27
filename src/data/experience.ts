export type Track = 'data-science' | 'backend';

export interface WorkItem {
  client: string;
  title: string;
  track: Track | 'both';
  summary: string;
  metric?: { value: string; label: string };
  stack: string[];
}

export interface Role {
  company: string;
  role: string;
  location: string;
  period: string;
  current?: boolean;
  items: WorkItem[];
}

export const experience: Role[] = [
  {
    company: 'ThoughtClan Technologies',
    role: 'Software Developer',
    location: 'Bengaluru, India',
    period: 'Apr 2025 – Present',
    current: true,
    items: [
      {
        client: 'FoqusMeta · Lumera Systems',
        title: 'AI-powered education support platform',
        track: 'backend',
        summary:
          'Led backend development: scalable services, APIs, data models and application workflows. Integrated AI capabilities into core platform functionality and worked with frontend and product teams to deliver features end to end.',
        stack: ['Backend APIs', 'Data Modelling', 'AI Integration'],
      },
      {
        client: 'Vedanta',
        title: 'Bath temperature prediction for aluminium plants',
        track: 'data-science',
        summary:
          'Built a predictive modelling system for bath temperature in aluminium production, with error lower than the natural spread of the temperature itself.',
        metric: { value: '6°C', label: 'RMSE, below the bath-temperature std. dev.' },
        stack: ['Predictive Modelling', 'Industrial Data'],
      },
      {
        client: 'Vedanta',
        title: 'Real-time perimeter intrusion detection',
        track: 'data-science',
        summary:
          'Developed a real-time detection system that monitors factory perimeters with YOLO-based object detection and raises alerts as events happen.',
        metric: { value: '0.68', label: 'mAP' },
        stack: ['YOLO', 'Computer Vision', 'Real-time Alerts'],
      },
      {
        client: 'Yokogawa',
        title: 'Agentic recycling marketplace',
        track: 'both',
        summary:
          'Contributed to a marketplace where AI agents represent companies and take part in auctions and negotiations over recyclable materials. Worked on agent development and prompt engineering.',
        stack: ['AI Agents', 'Prompt Engineering'],
      },
      {
        client: 'Yokogawa',
        title: 'Renewable energy price prediction',
        track: 'both',
        summary:
          'Built an ML application for renewable energy prices in the European market, including the model integration, inference pipeline and FastAPI backend. Extended it with a domain-specific GPT-4o chatbot over application and market data.',
        metric: { value: '82%', label: 'prediction accuracy' },
        stack: ['XGBoost', 'FastAPI', 'LangChain', 'LangGraph', 'GPT-4o'],
      },
    ],
  },
  {
    company: 'ThoughtClan Technologies',
    role: 'Data Science Intern',
    location: 'Remote',
    period: 'Feb 2025 – Mar 2025',
    items: [
      {
        client: 'Internal R&D',
        title: 'LLM applications and RAG',
        track: 'data-science',
        summary:
          'Prototyped AI-powered applications on open-source LLMs and built a retrieval-augmented generation system with Ollama embedding models and LangChain over domain-specific information.',
        stack: ['Ollama', 'LangChain', 'RAG', 'Open-source LLMs'],
      },
    ],
  },
];

export const education = [
  {
    school: 'University College Cork',
    place: 'Cork, Ireland',
    degree: 'MSc, Data Science and Analytics',
    period: 'Sept 2023 – Sept 2024',
    grade: 'Grade 1.1',
    courses: ['Data Mining', 'Deep Learning', 'Statistical Data Analytics', 'Generalised Linear Models', 'Optimisation'],
  },
  {
    school: 'Vellore Institute of Technology',
    place: 'Chennai, India',
    degree: 'B.Tech, Electronics & Communication',
    period: 'July 2019 – May 2023',
    grade: 'CGPA 8.74',
    courses: ['Digital Signal Processing', 'Machine Learning', 'Neural Networks', 'Digital Image Processing'],
  },
];

export const publications = [
  {
    title: 'A hybrid model for the classification of Autism Spectrum Disorder using Mu rhythm in EEG',
    date: 'July 2024',
  },
  {
    title:
      'Computational Structural Validation of CYP2C9 Mutations and Evaluation of Machine Learning Algorithms in Predicting the Therapeutic Outcomes of Warfarin',
    date: 'July 2023',
  },
  {
    title: 'Squeeze Excitation Embedded Attention U-Net for Brain Tumor Segmentation',
    date: 'May 2023',
    href: 'https://arxiv.org/abs/2305.07850',
  },
];
