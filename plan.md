# Dreamdeck Dynamic Personas & Multi-Agent Roadmap

This plan guides the evolution of Dreamdeck into a **dynamic, persona-aware, multi-agent storytelling platform** that prioritizes **player immersion** and **developer extensibility**.

---

## **Goals**

- **Serve the player first**: adapt tone, style, and tools to player needs.
- **Dynamic personas**: switch AI personas based on conversation context or user choice.
- **Persona-aware agents**: all agents adjust prompts and behavior per persona.
- **Config-driven**: easy to add new personas, prompts, and tool preferences.
- **Incremental, testable phases**.

---

## **Phases**

### **1. Persona Classifier Agent**

- **Create** `src/agents/persona_classifier_agent.py`.
- **Input**: recent chat, memories, tool results.
- **Output**: JSON like `{"persona": "therapist", "reason": "User is discussing emotional issues"}`.
- **Prompt**: instruct LLM to suggest persona from a fixed list:  
  `"storyteller_gm"`, `"therapist"`, `"secretary"`, `"coder"`, `"friend"`, `"lorekeeper"`, `"dungeon_master"`, `"default"`.
- **Run** after each user message, **store suggestion** in `cl.user_session["suggested_persona"]`.
- **Fallback**: if classifier fails, keep current persona.

---

### **2. User Confirmation Flow**

- When a **new persona is suggested**:
  - **Notify user**: _"Switch to Therapist mode? Yes/No"_
  - On **Yes**: update `cl.user_session["current_persona"]` and `ChatState.current_persona`.
  - On **No**: keep current persona.
- Always allow **manual override** via UI or `/persona` command.

---

### **3. Persona-Aware Prompts**

- In `config.yaml`, define **persona-specific prompt keys** for all agents:  
  ```yaml
  agents:
    writer_agent:
      personas:
        therapist: { prompt_key: "therapist_writer_prompt" }
        coder: { prompt_key: "coder_writer_prompt" }
        secretary: { prompt_key: "secretary_todo_prompt" }
        # etc.
    todo_agent:
      personas:
        secretary: { prompt_key: "secretary_todo_prompt" }
    knowledge_agent:
      personas:
        lorekeeper: { prompt_key: "lore_knowledge_prompt" }
  ```
- **Create prompt templates** for each persona/agent combo.
- **Modify agents** to select prompt based on `state.current_persona`.

---

### **4. Persona-Aware Director & Tool Preferences**

- **Update director prompt** to include:  
  `"Current persona: {{ persona }}"`
- **Instruct** LLM to **prioritize or avoid tools** based on persona, e.g.:
  - Therapist: avoid dice, favor conversation.
  - Secretary: favor todo, report.
  - Coder: favor code tools.
- **Config**:  
  ```yaml
  persona_tool_preferences:
    therapist: { avoid: ["roll"], prefer: ["knowledge"] }
    secretary: { prefer: ["todo", "report"] }
    coder: { prefer: ["code_search"] }
  ```

---

### **5. Persona-Aware Workflows**

- In `src/workflows.py`,  
  - **Filter or reorder** director actions based on persona preferences.
  - E.g., skip dice rolls if persona is therapist.
- **Trigger persona-specific agents** (e.g., secretary triggers report agent).

---

### **6. New Personas & Capabilities**

- **Therapist**: empathetic, supportive, no dice.
- **Secretary**: manages TODOs, daily reports.
- **Coder**: code help, debugging.
- **Friend**: casual chat.
- **Lorekeeper**: deep lore dumps.
- **Dungeon Master**: classic TTRPG, dice-heavy.
- **Add prompts** and **config entries** for each.

---

### **7. Daily Report Agent**

- **Create** `src/agents/report_agent.py`.
- **Aggregate** TODOs, notes, calendar.
- **Triggered** by `/report` or secretary persona.
- **Summarize** daily progress, suggest next steps.

---

### **8. User Experience Enhancements**

- **Persona switch notifications**:  
  _"Switching to Therapist persona to better assist you."_
- **Persona icons/avatars** in UI.
- **Settings toggle**: enable/disable auto persona switching.
- **Slash command** `/persona [name]` to force switch.
- **Logging**: track persona suggestions, switches, tool usage.

---

### **9. Testing**

- **Unit tests** for:
  - Persona classifier output.
  - Persona switch confirmation.
  - Persona-aware prompt selection.
  - Director persona context.
  - Workflow filtering.
- **Simulate** conversation scenarios to validate switching.

---

## **Summary**

| Phase | Description                          | Key Deliverables                          |
|--------|--------------------------------------|-------------------------------------------|
| 1      | Persona classifier agent             | Classifier agent, prompt, config          |
| 2      | User confirmation flow               | UI prompt, update persona on confirm      |
| 3      | Persona-aware prompts                | Config, prompt templates, agent updates   |
| 4      | Persona-aware director               | Director prompt update, tool prefs config |
| 5      | Persona-aware workflows              | Workflow filtering, persona triggers      |
| 6      | New personas                         | Prompts, config, optional new agents      |
| 7      | Daily report agent                   | Report agent, slash command               |
| 8      | UX enhancements                      | Notifications, UI, slash commands         |
| 9      | Testing                              | Unit tests, scenario tests                |

---

## **Guiding Principles**

- **Player-first**: adapt to player needs, maximize immersion.
- **Transparency**: notify or confirm persona switches.
- **Modularity**: config-driven, easy to extend.
- **Incremental**: build/test in phases.
- **Extensible**: add new personas, prompts, tools easily.

---

## **Start Here**

Begin with **Phase 1**:  
Implement the **persona classifier agent** and prompt.  
Test persona suggestions and logging.

---

**Dreamdeck will become a truly adaptive, player-centric storytelling platform.**

---
