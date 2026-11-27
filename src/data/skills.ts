export const skillGroups = [
  {
    name: 'Data Science & ML',
    track: 'data-science',
    blurb: 'Modelling, vision and statistics',
    items: ['Python', 'R', 'TensorFlow', 'PyTorch', 'Computer Vision', 'Deep Learning', 'Statistical Analysis', 'XGBoost'],
  },
  {
    name: 'GenAI & Agents',
    track: 'both',
    blurb: 'LLM applications that do real work',
    items: ['RAG', 'LangChain', 'LangGraph', 'PydanticAI', 'Ollama', 'Prompt Engineering', 'Claude Code', 'Cursor'],
  },
  {
    name: 'Backend & Systems',
    track: 'backend',
    blurb: 'APIs and services that hold up',
    items: ['FastAPI', 'WebSockets', 'Redis', 'SQL / NoSQL', 'Go', 'Kotlin', 'JavaScript'],
  },
  {
    name: 'Infrastructure & Tooling',
    track: 'backend',
    blurb: 'Shipping and running it',
    items: ['Docker', 'Git', 'Linux', 'Azure Cloud', 'FFmpeg', 'MediaMTX', 'GPU Inference'],
  },
] as const;
