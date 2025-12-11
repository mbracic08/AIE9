# AI Engineering Onramp

This folder contains materials for the **AI Engineer Onramp** — a 4-week live session series designed to help prepare for the AI Engineering Bootcamp.

---

## 🔗 Quick Access

| 📰 Session Topic | ⏺️ Recording  | 🖼️ Slides     | 👨‍💻 Repo     |
|:-----------------|:-------------|:--------------|:--------------|
| AI Assisted Development | [Recording!](https://us02web.zoom.us/rec/share/ztpkCP9S-eTyVe7CCFLpF2CM3_PWu-P81DBGmcZeYAW7DtSK9VL1elHIoDjdm_oW.RC-nq31aDuoYziOV) (f1#j7Nr^) | [Slides](https://www.canva.com/design/DAG6SNRlYoI/bpELIN03JVB1xNkd9yo8lA/edit?utm_content=DAG6SNRlYoI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | [Repo](https://github.com/AI-Maker-Space/AIEO2/tree/main/Session_01_AI_Assisted_Development) |
| Backend Development & Deployment | [Recording!](https://us02web.zoom.us/rec/share/IXFR3_h72eZ1TbXWWVRiaE4xTSQIPBnpIRZUR-M5450uR8CIo-5kza1ixH9BanrA.KKu-OIqI1YHr3cfZ ) (=7Ld3A2L) | [Slides](https://www.canva.com/design/DAG492HUYsU/d98h86nIBAbpLsJ2TBFriQ/edit?utm_content=DAG492HUYsU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | [Repo](https://github.com/AI-Maker-Space/AIEO2/tree/main/Session_02_Back_End_Web_App_Development_%26_Deployment_of_LLM_Applications) |

## 📚 The stack we'll use is made up of best-practice tools:

- 🐙 Version Control: [GitHub](https://github.com/)
- 🐚 CLI: Shell for Unix-like OS ([WSL](https://ubuntu.com/desktop/wsl))
- 📦 Package & Env Management: [uv](https://docs.astral.sh/uv/)
- 📓 Python Notebooks: [Jupyter](https://jupyter.org/) / [Colab](https://colab.google/)
- 🖱️ Code Editor: [Cursor](https://www.cursor.com/)
- 🤖 CLI Coding Agent: [Cursor CLI](https://cursor.com/docs/cli/overview)
- 🧠 LLM: [OpenAI GPT models](https://platform.openai.com/docs/models)
  - **Security**: Store API keys in a `.env` file (already in [.gitignore](/.gitignore#L138))
- 🎨 User Interface: Vibe-coded with [v0](https://v0.app/)
- ⚡ Web App Framework: [FastAPI](https://fastapi.tiangolo.com/)
- ☁️ Deployment: [Vercel](https://vercel.com/) & [Render](https://render.com/docs)


## 🔍 Setup Checker

Verify your development environment is ready:

Note: run the below commands from the root of the repo.

One off command:
```bash
chmod +x scripts/setup-checker.sh
```

To check the setup each time:
```bash
./scripts/setup-checker.sh
```

This checks your shell, OS tools, Git, Python, IDEs, and API keys. Follow the output instructions to fix any ❌ red X marks, then re-run to verify.

