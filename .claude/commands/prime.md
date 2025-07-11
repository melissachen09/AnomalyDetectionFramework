# Context Window Prime

Always call necessary tool calls in parallel

RUN:

# Basic project analysis

- git ls-files
- find specs -name "_.md" -not -path "specs/completed/_"
- git log --oneline -10
- git branch --show-current
- git status --porcelain
- claude mcp list

READ:

# Core documents (always load)

- README.md
- specs/project-management.md
- specs/code-standards.md
- specs/project_plan.md
- specs/anomaly-detection-design-doc.md
- specs/work_plan.md

Always use conventional commits
Always create PR before making changes 

# Note: JIRA issues should be loaded via MCP on demand when working on specific features/tasks
