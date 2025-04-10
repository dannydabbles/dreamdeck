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

### **1. Persona Classifier Agent** ✅ **Completed**

- Implemented `src/agents/persona_classifier_agent.py`.
- Classifies persona after each user message.
- Stores suggestion in `cl.user_session["suggested_persona"]`.
- Falls back to "default" on error.
- Persona classifier prompt moved to `src/prompts/persona_classifier_prompt.j2` and loaded via config.
- **Completed**

---

### **2. User Confirmation Flow** ✅ **Completed**

- When a **new persona is suggested**:
  - **Notify user**: _"Switch to Therapist mode? Yes/No"_
  - On **Yes**: update `cl.user_session["current_persona"]` and `ChatState.current_persona`.
  - On **No**: keep current persona.
- Always allow **manual override** via UI or `/persona` command. (Planned for later phase)
- **Completed**

---

### **3. Persona-Aware Prompts** ✅ **Completed**

- Config now supports persona-specific prompt keys for writer, todo, and knowledge agents.
- Agents dynamically select prompt template based on `ChatState.current_persona`.
- New prompt templates added for therapist, coder, secretary, lorekeeper personas.
- **Completed**

---

### **4. Persona-Aware Director & Tool Preferences** ✅ **Completed**

- **Director prompt updated** to include:  
  `"Current persona: {{ persona }}"`  
  and  
  `"Persona tool preferences: {{ persona_preferences }}"`
- **Config** now supports `persona_tool_preferences` dict with `prefer` and `avoid` lists per persona.
- **Director agent** passes persona and preferences into the prompt, enabling persona-aware tool selection.
- **Completed**

---

### **5. Persona-Aware Workflows** ✅ **Completed**

- In `src/workflows.py`,  
  - Director actions are now **filtered and reordered** dynamically based on persona preferences.
  - E.g., dice rolls are skipped if persona is therapist; preferred tools are prioritized.
- **Note:** Persona-specific agents like report agent will be added in Phase 7.
- **Completed**

---

### **6. New Personas & Capabilities** ✅ **Completed**

- Added **Therapist**, **Secretary**, **Coder**, **Friend**, **Lorekeeper**, **Dungeon Master** personas.
- Created prompt templates for each persona.
- Updated config with persona-specific prompt keys.
- Persona-aware prompts, director, and workflows now fully support these personas.
- **Completed**

---

### **7. Daily Report Agent** ✅ **Completed**

- Implemented `src/agents/report_agent.py`.
- Generates a daily summary report based on the TODO markdown file.
- Triggered by `/report` command or can be invoked programmatically.
- Summarizes progress and suggests next steps.
- Future: can extend to aggregate notes, calendar, or other data sources.
- **Completed**

---

### **8. User Experience Enhancements** ✅ **Completed**

- Persona switch notifications implemented.
- Persona icons/avatars added to chat messages.
- Settings toggle for auto persona switching added.
- Slash command `/persona [name]` implemented.
- Logging added for persona suggestions, switches, and forced changes.
- **Completed**

---

### **9. Testing** ✅ **Completed**

- Unit tests added for:
  - Persona classifier output
  - Persona switch confirmation flow
  - Persona-aware prompt selection
  - Director persona context
  - Workflow filtering and reordering
  - Simulated conversation with persona switching
- Tests use mocks and monkeypatching to isolate components.
- **Completed**

---

## **Summary**

| Phase | Description                          | Key Deliverables                          |
|--------|--------------------------------------|-------------------------------------------|
| 1      | Persona classifier agent             | **✅ Completed**                          |
| 2      | User confirmation flow               | **✅ Completed**                          |
| 3      | Persona-aware prompts                | **✅ Completed**                          |
| 4      | Persona-aware director               | **✅ Completed**                          |
| 5      | Persona-aware workflows              | **✅ Completed**                          |
| 6      | New personas                         | **✅ Completed**                          |
| 7      | Daily report agent                   | **✅ Completed**                          |
| 8      | UX enhancements                      | **✅ Completed**                          |
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
