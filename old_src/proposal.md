Project Proposal: LLM Reasoning for Intelligent DJ Set Planning and Transition Suggestion
Sean Theuri, Nyasha Makaya, Abdulkarim Mugisha
1. Introduction: Research Area and Motivation
DJing requires sophisticated reasoning beyond mere beatmatching, involving sequencing tracks and choosing transitions to craft a desired mood, energy flow, and narrative. Human DJs integrate diverse cues: rhythm, harmony, lyrics, energy, transition variety, and audience feedback. While automatic DJ systems handle acoustic alignment, they often lack the higher-level semantic coherence, context-aware transition choices, and goal-driven planning of experts.
This project investigates applying Large Language Models (LLMs) to these DJ reasoning tasks. We explore LLM-driven planning of coherent track sequences from playlists based on user specifications and multimodal features. We also investigate the LLM's ability to suggest appropriate transition types contextually, adding intentionality missing in many automated systems. This presents a unique domain for LLM reasoning, integrating symbolic musical features (tempo, key) with semantic content (lyrics, mood) and contextual awareness (mix history, goals).
The core challenge is representing multimodal information and mixing goals effectively for LLM processing and eliciting sophisticated planning. Though LLMs don't directly process audio, we hypothesize they can reason effectively over structured feature representations (extracted via specialized tools and LLM pre-processing). Focusing on the LLM as a high-level "orchestrator," we aim to generate DJ set plans (sequences with transition suggestions) that are technically sound, semantically coherent, and aligned with user intent, showcasing LLM reasoning in creative sequential decision-making.
2. Overview of Previous Work
Previous automatic DJ systems primarily fall into three categories:
Rule-based systems: Employed hard-coded rules based on music theory and DJ practice (e.g., BPM matching, key compatibility) for sequencing and often used default transition mechanisms (e.g., constant-power crossfade) (e.g., Ishizaki et al., 2009). These are often rigid and lack adaptability or nuanced transition choices.
Feature-based optimization: Formulated track selection and cue point identification as optimization problems minimizing cost functions based on extracted audio features (timbre, harmony, loudness) (e.g., Bittner et al., 2017). These offer more flexibility but rely on hand-engineered features and cost functions, often ignoring semantics or sophisticated transition strategies.
Learning-based methods: Used neural networks (CNNs, GANs, sequence models) to learn transition aesthetics or detect structural elements directly from audio data (e.g., Huang et al., 2017; Chen et al., 2022). These capture subtle nuances but may struggle with explicit goal satisfaction, semantic coherence, or explainable context-aware transition selection.
A key limitation across these approaches is the difficulty in integrating high-level semantic understanding and complex, user-defined goals into the planning process. Furthermore, the choice of transition type is often simplistic or lacks contextual awareness. Traditional methods may optimize for local acoustic smoothness but fail to generate a globally coherent sequence or vary transitions intelligently. This gap presents an opportunity for LLMs, which excel at natural language understanding, context tracking, and complex instruction following, to perform the high-level reasoning required for intelligent setlist planning and context-aware transition suggestion.
3. Research Questions
This project addresses these core questions on LLM reasoning:
Representation: What textual/structured representations of tracks (acoustic features + LLM-generated semantic summaries) best enable LLM reasoning for DJ planning?
Reasoning (Sequencing): How effectively can LLMs plan coherent track sequences via few-shot/zero-shot learning, based on representations and natural language goals?
Reasoning (Transition Suggestion): Can LLMs concurrently suggest appropriate transition types (from a predefined vocabulary) based on musical/semantic context and mix history?
Evaluation: Do LLM-generated plans (sequence + transitions) show improved coherence, goal adherence, or perceived creativity versus baselines?
4. Plan of Attack
Our plan focuses on using an LLM as the core reasoning engine for offline DJ set planning and transition suggestion, scoping out audio generation and real-time interaction.
(a) Problem Formulation:
The task is defined as: Given an unordered list of tracks and a natural language user prompt, the system outputs an ordered playlist, where each step includes the chosen track and a suggested transition type to reach it.
(b) Input Representation:
We will leverage existing tools and LLMs for feature extraction, representing the information textually:
Tracks: A list of available songs.
Acoustic Features: For each track, pre-extract standard features using libraries (e.g., Librosa, Essentia): BPM, key, duration, energy/danceability score, potentially structural markers. Represent concisely (e.g., "Track Name": {BPM: 125, Key: "Am", Energy: "High", Duration: "5:30"}).
Semantic Features: Extract lyrical content (if available) via ASR. Use an LLM in a preprocessing step to generate concise textual summaries of each track's theme, mood, or topic (e.g., Theme: "Uplifting, Love", Mood: "Energetic").
User Prompt: A natural language instruction (e.g., "Create a 30-minute warm-up techno mix, starting mellow around 120 BPM, gradually increasing intensity. Vary the transitions.").
(c) LLM Reasoning Method (Core Investigation):
We will investigate prompt engineering strategies with capable LLMs.
Primary Task (Sequencing): The core goal is for the LLM to select the next track based on compatibility (BPM, key, energy), semantic coherence (themes, mood), and alignment with the user's overall goal (energy curve, genre focus).
Secondary Task (Transition Suggestion): Concurrently, we will prompt the LLM to suggest a transition type from a small, predefined vocabulary (e.g., SimpleCrossfade_16beat, BeatmatchedCut, EQSwap, FilterSweep).
Prompt Design: Craft detailed prompts instructing the LLM to:
Consider all relevant features (acoustic, semantic) for track selection.
Adhere to the user prompt's constraints and goals.
For transition suggestion: Use simple heuristics provided in the prompt (e.g., "prefer smooth fades for harmonic matches," "use cuts for large energy jumps," "avoid repeating the same transition type consecutively if possible").
Reasoning Process Exploration: Experiment primarily with Chain-of-Thought (CoT) prompting, asking the LLM to explain its choice of both the next track and the suggested transition type at each step. This makes the reasoning explicit.
(d) Output:
The primary output will be the ordered list of track names, paired with the LLM-suggested transition type for each step (except the first track). The CoT reasoning trace will also be captured for analysis.
(e) Evaluation:
Evaluation will focus on the quality of the generated plan:
Objective Metrics (Sequence): Adherence to constraints (BPM, key rules, duration), quantifiable semantic coherence, energy curve alignment.
Qualitative Evaluation (Sequence & Transitions):
Human assessment (peers, TAs) of the plans based on: logical flow, creativity, semantic coherence, faithfulness to user prompt, and importantly, the appropriateness and variety of the suggested transition types compared to a default approach.
Analysis of the LLM's CoT reasoning for both sequencing and transition choices.

References
Ishizaki, H., Hoashi, K., & Takishima, Y. (2009, October). Full-Automatic DJ Mixing System with Optimal Tempo Adjustment based on Measurement Function of User Discomfort. In ISMIR (pp. 135-140). https://ismir2009.ismir.net/proceedings/PS1-14.pdf
Bittner, R. M., Gu, M., Hernandez, G., Humphrey, E. J., Jehan, T., McCurry, H., & Montecchio, N. (2017, October). Automatic Playlist Sequencing and Transitions. In ISMIR (pp. 442-448). https://archives.ismir.net/ismir2017/paper/000086.pdf
Huang, Y. S., Chou, S. Y., & Yang, Y. H. (2017). DJnet: A dream for making an automatic DJ. In International Society for Music Information Retrieval Conference, Late-Breaking Paper (pp. 1-2). https://remyhuang.github.io/files/huang17ismir-lbd.pdf
Chen, B. Y., Hsu, W. H., Liao, W. H., Ramírez, M. A. M., Mitsufuji, Y., & Yang, Y. H. (2022, May). Automatic DJ transitions with differentiable audio effects and generative adversarial networks. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 466-470). IEEE. https://ieeexplore.ieee.org/abstract/document/9746663/



