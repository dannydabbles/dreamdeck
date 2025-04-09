---

# Dreamdeck Dynamic Personas Roadmap

This document outlines a step-by-step plan to evolve Dreamdeck into a **dynamic, persona-aware, multi-agent storytelling platform**. The goal is to **enhance player immersion** by adapting the AI's persona and tool usage **automatically based on conversation context**, while maintaining a **developer-friendly, modular architecture**.

---

## **Overview**

- **Personas**: Distinct AI "characters" or modes (e.g., Storyteller GM, Therapist, Secretary, Coder, Friend) that influence style, tone, and tool preferences.
- **Dynamic Persona Switching**: The system **detects conversation tone, topic, or user intent** and **suggests or auto-switches** personas accordingly.
- **Persona-Aware Agents & Workflows**: All AI agents (writer, todo, search, knowledge, storyboard) **adapt prompts and behavior** based on the active persona.
- **User Control**: Users can **manually select personas** or **accept persona suggestions**.
- **Config-Driven**: Persona prompts, tool preferences, and behaviors are **configurable** for easy extension.

---

## **Phases**

---

### **Phase 1: Persona Classifier Agent**

**Goal:**  
Create an agent that analyzes recent chat and **suggests the most appropriate persona**.

**Implementation:**

- **New file:** `src/agents/persona_classifier_agent.py`
- **Input:**  
  - `ChatState` (recent chat history, memories, tool results)
- **Output:**  
  - JSON: `{"persona": "therapist", "reason": "User is discussing emotional issues"}`
- **Prompt:**  
  - Use a Jinja2 template (e.g., `persona_classifier_prompt.j2`) instructing the LLM to analyze the conversation and suggest the best persona from a **fixed list**:
    - `"storyteller_gm"`
    - `"therapist"`
    - `"secretary"`
    - `"coder"`
    - `"friend"`
    - `"default"`
- **Logic:**  
  - After each user message, call this agent **asynchronously**.  
  - If the suggested persona **differs** from current, **store suggestion** in session (do **not** auto-switch yet).
- **Config:**  
  - Add prompt file mapping in `config.yaml`:
    ```yaml
    prompts:
      persona_classifier_prompt: "persona_classifier_prompt.j2"
    ```
  - Add default prompt file `src/prompts/persona_classifier_prompt.j2` with clear instructions.

---

### **Phase 2: User Notification & Confirmation**

**Goal:**  
When the classifier suggests a new persona, **notify the user** and **ask for confirmation** before switching.

**Implementation:**

- In `src/event_handlers.py` or `src/workflows.py`, after persona classifier runs:
  - If suggested persona ≠ current persona:
    - Send a message:  
      `"I think switching to **Therapist** mode might help here. Would you like me to switch personas?"`  
      Provide **Yes/No** buttons.
- On **Yes**:
  - Update `cl.user_session["current_persona"]` and `ChatState.current_persona`.
  - Send a confirmation message.
- On **No**:
  - Keep current persona.
- **Fallback:**  
  - If no response, default to **no switch**.

---

### **Phase 3: Persona-Aware Prompt Selection for All Agents**

**Goal:**  
Extend **all agents** to select prompts based on **active persona**.

**Implementation:**

- **Config changes:**

  ```yaml
  agents:
    writer_agent:
      personas:
        storyteller_gm:
          prompt_key: "storyteller_gm_prompt"
        therapist:
          prompt_key: "therapist_writer_prompt"
        coder:
          prompt_key: "coder_writer_prompt"
        secretary:
          prompt_key: "secretary_writer_prompt"
        friend:
          prompt_key: "friend_writer_prompt"
        default:
          prompt_key: "default_writer_prompt"
    todo_agent:
      personas:
        secretary:
          prompt_key: "secretary_todo_prompt"
        default:
          prompt_key: "todo_prompt"
    # Repeat for other agents as needed
  ```

- **Prompt files:**  
  Create new Jinja2 prompt templates for each persona/agent combo (e.g., `therapist_writer_prompt.j2`, `secretary_todo_prompt.j2`).

- **Code changes:**  
  In each agent (writer, todo, knowledge, storyboard, etc.),  
  - Load persona-specific prompt key from config.  
  - Fallback to default if not found.

---

### **Phase 4: Persona-Aware Director Agent**

**Goal:**  
Make the **director agent** consider persona when deciding which tools to invoke.

**Implementation:**

- **Prompt update:**  
  - Add persona context to the director prompt:  
    `"Current persona: {{ persona }}"`
  - Instruct the LLM to **prioritize or skip tools** based on persona.  
    E.g.,  
    - Therapist persona: avoid dice rolls, favor knowledge or conversation.  
    - Secretary persona: favor todo agent.  
    - Coder persona: favor code search or code generation tools.
- **Config:**  
  - Optionally, define persona-specific **tool preferences** in config.
- **Output:**  
  - Director still returns ordered list of actions, but **persona-aware**.

---

### **Phase 5: Persona-Aware Workflows**

**Goal:**  
Adjust the **main chat workflow** to respect persona preferences.

**Implementation:**

- In `src/workflows.py`,  
  - After director returns actions,  
  - Optionally **filter or reorder** actions based on persona preferences.
- Example:  
  - If persona is therapist, **skip dice roll** even if director suggests it.  
  - If secretary, **always call todo agent** after user input.
- **Config:**  
  - Define persona-specific workflow rules if needed.

---

### **Phase 6: New Personas and Capabilities**

**Goal:**  
Add new personas and specialized agents to **enhance player experience**.

**Suggestions:**

- **Therapist Persona:**  
  - Empathetic, supportive tone.  
  - Avoids game mechanics, focuses on conversation.  
  - Uses a `therapist_writer_prompt.j2`.
- **Secretary Persona:**  
  - Manages TODOs, reminders, scheduling.  
  - Calls todo agent proactively.  
  - Summarizes daily notes.
- **Coder Persona:**  
  - Assists with code generation, debugging.  
  - Calls code-related tools (future work).
- **Friend Persona:**  
  - Casual, supportive, social chat.
- **Lorekeeper Persona:**  
  - Focuses on world-building, background info.  
  - Calls knowledge agent more often.
- **Dungeon Master Persona:**  
  - Classic TTRPG style, dice-heavy, rules-focused.

---

### **Phase 7: Daily Report Agent**

**Goal:**  
Aggregate data from the **helper directory** (TODOs, notes) and provide **daily summaries**.

**Implementation:**

- **New agent:** `src/agents/report_agent.py`
- **Trigger:**  
  - Slash command `/report`  
  - Or scheduled (future work)
- **Behavior:**  
  - Reads markdown files from helper directory.  
  - Summarizes completed, in-progress, and remaining tasks.  
  - Optionally, suggests next steps.
- **Persona-aware:**  
  - Secretary persona favors this agent.

---

### **Phase 8: User Experience Enhancements**

**Goal:**  
Make persona switching and multi-agent orchestration **transparent and engaging**.

**Ideas:**

- **Persona switch notifications** with characterful messages.  
  E.g., `"Your AI has donned the Therapist hat to better assist you."`
- **Persona icons or avatars** in UI.
- **Persona-specific voice or style cues**.
- **User override commands:**  
  `/persona therapist` to force switch.  
  `/persona default` to reset.
- **Settings UI:**  
  Allow user to **enable/disable auto persona switching**.

---

## **Guiding Principles**

- **Player-first:**  
  Prioritize immersion, clarity, and helpfulness.
- **Transparency:**  
  Always **notify or confirm** persona switches.
- **Modularity:**  
  Keep agents, prompts, and workflows **configurable and decoupled**.
- **Extensibility:**  
  Make it easy to **add new personas, prompts, and tools**.
- **Fallbacks:**  
  Always have sensible defaults if persona detection fails.

---

## **Summary Table**

| Phase | Description                               | Key Deliverables                                         |
|--------|-------------------------------------------|----------------------------------------------------------|
| 1      | Persona Classifier Agent                 | `persona_classifier_agent.py`, prompt, config            |
| 2      | User Confirmation for Persona Switch     | UI buttons, session update logic                         |
| 3      | Persona-Aware Prompts for All Agents     | Config updates, new prompt files, agent updates          |
| 4      | Persona-Aware Director Agent             | Prompt update, config, director logic                    |
| 5      | Persona-Aware Workflows                  | Workflow logic update, config                            |
| 6      | New Personas & Capabilities              | New prompts, config, optional new agents                 |
| 7      | Daily Report Agent                       | `report_agent.py`, helper dir parsing, slash command     |
| 8      | User Experience Enhancements             | UI tweaks, notifications, override commands              |

---

## **Next Steps**

Start with **Phase 1**:  
Implement the **persona classifier agent** and its prompt.  
Test it by logging suggestions after each user message.

---

# End of Plan

This plan is designed to be **incremental**.  
Each phase can be implemented independently and tested before moving on.  
Focus on **player engagement** and **clear, persona-driven experiences**.

---

**Good luck!**
