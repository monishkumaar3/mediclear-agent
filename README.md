Problem Statement
Medical records are often a "black box" for patients. When a patient receives a diagnosis or reads their clinical notes, they are confronted with dense jargon like "myocardial infarction" or "acute rhinitis" rather than simple terms like "heart attack" or "stuffy nose."

This gap in Health Literacy leads to three major issues:

Patient Anxiety: Fear of the unknown.

Poor Adherence: Patients don't take medication correctly because they don't understand the "why."

Doctor Burnout: Physicians spend valuable time re-explaining basic concepts.

I built MediClear to solve this by acting as an intelligent bridge between raw medical data and human understanding.

Why agents?
A standard chatbot often hallucinates or mixes up tone. By using a Multi-Agent System, I could separate the concerns into two distinct "brains":

Accuracy (The Researcher): One agent is strictly constrained to facts and data retrieval. It doesn't care about feelings; it cares about precision.

Empathy (The Translator): The second agent is free to focus entirely on tone, analogies, and simplicity, without the burden of searching the database.

This Sequential Agent Architecture ensures that the medical advice is grounded in fact (RAG) before it is ever softened for the patient. A single prompt often struggles to balance these two opposing goals (technical precision vs. simple empathy), but two agents handle it perfectly.

What you created
I built MediClear, a sequential multi-agent application powered by Gemini 2.5 Flash.

The Architecture:

The Brain: The system uses a Sequential Chain.

Step 1: The Researcher Agent uses a custom Tool (Vector Search) to find clinical evidence in the patient's file.

Step 2: The Translator Agent receives the dry facts and the patient's chat history. It rewrites the facts into a 5th-grade reading level using comforting analogies.

The Memory: An in-memory session manager allows the agent to remember context (e.g., if the patient mentioned they hate needles earlier, the agent remembers that context for future responses).

The Knowledge Base: A Retrieval-Augmented Generation (RAG) system using FAISS to index medical text files.


Shutterstock
Demo
(Link your YouTube video here)

In the demo, you can see the agent in action processing a query about "Acute Rhinitis."

Input: The user asks, "What is my diagnosis and what should I do?"

Process: You see the Researcher Agent scan the database and extract technical details about viral loads and hydration.

Output: The Translator Agent converts this into: "Think of your nose like a blocked tunnel. You need rest and water to clear the traffic..."

The Build
This project was built using Python and the Google Generative AI SDK.

Models: gemini-2.5-flash for the agent reasoning (chosen for its speed and long-context window) and text-embedding-004 for the vector search.

Vector Database: FAISS (Facebook AI Similarity Search) to store and retrieve medical chunks efficiently.

Orchestration: I wrote a custom Python orchestration script (llm_agent.py) that handles the hand-off between the Researcher and Translator agents.

Safety: I implemented custom safety settings to allow the model to discuss medical terms without triggering false-positive safety blocks.

If I had more time, this is what I'd do
Add a "Critic" Agent: I would add a third agent to review the Translator's output for safety before showing it to the user, acting as a final "Doctor Sign-off."

Multimodal Input: I would use Gemini's vision capabilities to let patients upload a photo of their prescription bottle or X-ray report directly.

Voice Interaction: I would add a text-to-speech layer so the elderly or visually impaired patients could "talk" to MediClear over the phone.
